"""OpenAI Chat Completions provider：重试、usage 提取、离线重放。

为什么选 Chat Completions 而非官方推荐的 Responses API：中转平台普遍只透传
/v1/chat/completions。代价是 GPT-5 系列的 reasoning items 无法跨轮保留
（每轮工具调用后模型重新推理）。权衡讨论见 DESIGN.md §Provider。

重试策略（自研而非用 SDK 内置，因为要把每次退避写进 trajectory）：
- 可重试：429 / 500 / 502 / 503 / 504 / 连接错误。429 优先遵循 Retry-After 头。
- 不可重试：400 / 401 / 403 / 404 / 413 / 422 —— 重试只会原样重复失败。
- 指数退避 + 全抖动：min(2^attempt + U(0,1), 60)。无抖动会让同时被打回的
  客户端同步重试，复现过载（thundering herd）。
"""
from __future__ import annotations

import random
import time

import openai

from ..telemetry import Usage
from .base import ModelTurn, Provider, RetryCallback, ToolCallRequest

def _retryable(status: int) -> bool:
    # 429 + 全部 5xx：经中转网关时常见 Cloudflare 系 52x（520-526 源站不可达），
    # 与 500/502/503/504 同属瞬时故障；4xx（除 429）重试只会原样重复失败。
    return status == 429 or status >= 500


class OpenAIChatProvider(Provider):
    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None,
                 max_retries: int = 5, reasoning_effort: str | None = None,
                 max_completion_tokens: int | None = None, timeout: float = 600.0):
        # SDK 自带重试关掉（max_retries=0），换成可观测的自研重试
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url,
                                    max_retries=0, timeout=timeout)
        self.model = model
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        # DeepSeek 式思考方言：一旦某次响应携带 reasoning_content，后续每条
        # assistant 历史消息都必须带该字段（缺失补空串），否则 400。
        # 标准 OpenAI 模型永远不会触发此分支，不会被发送未知字段。
        self._thinking_dialect = False

    def complete(self, messages: list[dict], tools: list[dict],
                 on_retry: RetryCallback | None = None) -> ModelTurn:
        if self._thinking_dialect:
            messages = [
                {**m, "reasoning_content": m.get("reasoning_content", "")}
                if m.get("role") == "assistant" else m
                for m in messages
            ]
        kwargs: dict = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        # 仅显式设置时才传，部分中转网关会拒绝不认识的参数
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.max_completion_tokens:
            kwargs["max_completion_tokens"] = self.max_completion_tokens

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            t0 = time.monotonic()
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return self._to_turn(resp, latency_ms=int((time.monotonic() - t0) * 1000))
            except openai.APIConnectionError as e:
                last_exc, status, retry_after = e, None, None
            except openai.APIStatusError as e:  # RateLimitError 等都是其子类
                if not _retryable(e.status_code):
                    raise
                last_exc, status = e, e.status_code
                retry_after = _parse_retry_after(e)
            if attempt >= self.max_retries:
                break
            sleep_s = retry_after if retry_after is not None else min(
                2.0 ** attempt + random.uniform(0, 1), 60.0)
            if on_retry:
                on_retry(attempt + 1, status, str(last_exc), round(sleep_s, 2))
            time.sleep(sleep_s)
        raise last_exc  # type: ignore[misc]

    def _to_turn(self, resp, latency_ms: int) -> ModelTurn:
        choice = resp.choices[0]
        msg = choice.message
        tool_calls = [
            ToolCallRequest.from_raw(tc.id, tc.function.name, tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ]
        u = resp.usage
        # *_details 字段在部分中转/旧网关上缺失，缺省按 0 处理
        prompt_details = getattr(u, "prompt_tokens_details", None)
        completion_details = getattr(u, "completion_tokens_details", None)
        usage = Usage(
            prompt_tokens=u.prompt_tokens,
            cached_tokens=getattr(prompt_details, "cached_tokens", 0) or 0,
            completion_tokens=u.completion_tokens,
            reasoning_tokens=getattr(completion_details, "reasoning_tokens", 0) or 0,
        )
        reasoning_content = getattr(msg, "reasoning_content", None)
        if reasoning_content is not None:
            self._thinking_dialect = True
        return ModelTurn(
            content=msg.content,
            tool_calls=tool_calls,
            # 有 tool_calls 但 finish_reason 异常时以实际内容为准，归一化处理
            finish_reason=choice.finish_reason if not tool_calls else "tool_calls",
            usage=usage,
            request_id=getattr(resp, "_request_id", None),
            latency_ms=latency_ms,
            reasoning_content=reasoning_content,
        )


def _parse_retry_after(e: openai.APIStatusError) -> float | None:
    try:
        value = e.response.headers.get("retry-after")
        return min(float(value), 120.0) if value else None
    except (ValueError, AttributeError):
        return None


class ReplayProvider(Provider):
    """从历史 trajectory 离线重放模型响应：不打 API、零成本、完全确定。

    同时服务三件事：离线协议测试、面谈现场演示、复现历史运行。
    """

    def __init__(self, events: list[dict]):
        self._responses = [e for e in events if e["type"] == "llm_response"]
        self._i = 0

    def complete(self, messages: list[dict], tools: list[dict],
                 on_retry: RetryCallback | None = None) -> ModelTurn:
        if self._i >= len(self._responses):
            raise RuntimeError(
                f"replay exhausted after {self._i} responses; "
                "the conversation diverged from the recorded run")
        e = self._responses[self._i]
        self._i += 1
        u = e.get("usage", {})
        return ModelTurn(
            content=e.get("content"),
            tool_calls=[
                ToolCallRequest.from_raw(tc["id"], tc["name"], tc["arguments"])
                for tc in (e.get("tool_calls") or [])
            ],
            finish_reason=e.get("finish_reason", "stop"),
            usage=Usage(**{k: u.get(k, 0) for k in
                           ("prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens")}),
            request_id=e.get("request_id"),
            latency_ms=0,
            reasoning_content=e.get("reasoning_content"),
        )
