"""Provider 抽象：harness 与模型协议解耦的边界。

loop.py 只认识这里定义的 ModelTurn / ToolCallRequest，不接触任何厂商 SDK 类型。
换协议（Anthropic Messages / OpenAI Responses）只需新增一个 Provider 实现，
协议映射表见 DESIGN.md §Provider。
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

from ..telemetry import Usage

# on_retry(attempt, status, error, sleep_s) —— provider 在每次退避前回调，由 loop 写 telemetry
RetryCallback = Callable[[int, int | None, str, float], None]


@dataclass
class ToolCallRequest:
    """模型发起的一次工具调用。

    arguments_raw 保留原始 JSON 字符串（重放保真 + 回传协议需要原文）；
    arguments 为解析结果，None 表示模型吐出了非法 JSON——这本身是一种
    工具错误，loop 仍必须为该 id 回一条 tool 消息（OpenAI 协议：漏答任何
    一个 tool_call_id，下次请求直接 400）。
    """
    id: str
    name: str
    arguments_raw: str
    arguments: dict | None = None
    parse_error: str | None = None

    @classmethod
    def from_raw(cls, id_: str, name: str, arguments_raw: str) -> "ToolCallRequest":
        try:
            args = json.loads(arguments_raw)
            if not isinstance(args, dict):
                return cls(id_, name, arguments_raw,
                           parse_error=f"arguments must be a JSON object, got {type(args).__name__}")
            return cls(id_, name, arguments_raw, arguments=args)
        except json.JSONDecodeError as e:
            return cls(id_, name, arguments_raw, parse_error=f"invalid JSON in arguments: {e}")


@dataclass
class ModelTurn:
    """一次模型响应的厂商无关表示。"""
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"        # 归一化: "tool_calls" | "stop" | "length"
    usage: Usage = field(default_factory=Usage)
    request_id: str | None = None
    latency_ms: int = 0
    # DeepSeek 等思考模型的协议方言：响应里的 reasoning_content 必须原样回传，
    # 否则后续请求 400（"reasoning_content must be passed back"）。标准 OpenAI
    # 模型没有该字段，None 时不写入消息。
    reasoning_content: str | None = None

    def to_assistant_message(self) -> dict:
        """转回 OpenAI 消息格式以追加进历史（arguments 用原始字符串保真）。"""
        msg: dict = {"role": "assistant", "content": self.content}
        if self.reasoning_content is not None:
            msg["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": tc.arguments_raw}}
                for tc in self.tool_calls
            ]
        return msg


class Provider(ABC):
    @abstractmethod
    def complete(self, messages: list[dict], tools: list[dict],
                 on_retry: RetryCallback | None = None) -> ModelTurn:
        """一次模型调用。重试在 provider 内部完成，每次退避前回调 on_retry。

        抛出异常即为不可恢复错误（4xx 非限流 / 重试耗尽），由 loop 终止运行。
        """
