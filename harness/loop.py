"""Agent loop：整个项目刻意保持最简的部分。

循环只做一件事：调模型 → 按 finish_reason 分派 → 执行工具 → 回传 → 重复。
复杂度全部住在 harness 各层（providers/tools/context/telemetry）里。

协议铁律（tests/test_protocol.py 逐条验证）：
1. assistant 消息里的每一个 tool_call_id 都必须有对应的 role=tool 应答，
   包括工具报错、参数 JSON 非法、被安全策略拒绝的情况——漏一个，下次请求 400。
2. 工具错误回传而非终止：模型拿到错误文本后自纠（OpenAI 无 is_error 标记，
   用 "ERROR: " 前缀给出无歧义信号）。
3. finish_reason == "length" 不是完成：截断后提示模型收敛，再截断才放弃。
终止原因取值：completed | max_turns | max_cost | truncated | interrupted | error
"""
from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor

from .config import Config, load_pricing
from .context import ContextManager, strip_internal_marks
from .hooks import denial_message, gate_tool_call
from .providers.base import ModelTurn, Provider, ToolCallRequest
from .skills import render_skills_section
from .telemetry import CostLedger, RunLogger
from .tools import ToolContext, execute_tool, openai_tool_schemas
from .tools.registry import ToolResult

SYSTEM_PROMPT = """You are a precise, autonomous agent working in a sandboxed workspace.

Workspace root: {workdir}
All file paths are relative to the workspace root, which is the only place you can read or write.

Operating rules:
- Use tools for every action and computation; never invent file contents or numeric results.
- Verify deliverables: after writing an output file, read it back to confirm it is correct.
- Double-check arithmetic with the calculator tool.
- If a tool returns an error, read it carefully and adapt (list files, page through input, \
or switch approach) instead of repeating the same call.
- When the task is fully done, reply with a concise final summary (no tool calls): \
what you did, the key results, and which files you wrote.{skills}"""

LENGTH_NUDGE = ("Your previous reply was cut off by the output token limit. "
                "Continue, but be much more concise; put large content into files "
                "via tools instead of into your reply.")


def build_initial_messages(task: str, cfg: Config) -> list[dict]:
    system = SYSTEM_PROMPT.format(workdir=cfg.workdir,
                                  skills=render_skills_section(cfg.skills))
    return [{"role": "system", "content": system},
            {"role": "user", "content": task}]


def run_agent(task: str | None, cfg: Config, provider: Provider,
              logger: RunLogger, resume_messages: list[dict] | None = None) -> dict:
    """跑一个任务，返回 summary dict（同时落盘 summary.json）。"""
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    messages = resume_messages or build_initial_messages(task or "", cfg)
    schemas = openai_tool_schemas()
    ledger = CostLedger(load_pricing())
    cm = ContextManager(cfg.context_budget, cfg.context_keep_recent)
    tool_ctx = ToolContext(cfg.workdir, cfg.bash_timeout, cfg.tool_output_limit)

    logger.emit("run_start", task=task, model=cfg.model, workdir=str(cfg.workdir),
                config={"max_turns": cfg.max_turns, "max_cost_usd": cfg.max_cost_usd,
                        "context_budget": cfg.context_budget,
                        "reasoning_effort": cfg.reasoning_effort},
                sdk_version=_openai_version(), skills=cfg.skills)

    reason, final, turn, nudged = "max_turns", None, 0, False
    try:
        for turn in range(1, cfg.max_turns + 1):
            if ledger.cost_usd >= cfg.max_cost_usd:
                reason = "max_cost"
                break

            edit = cm.maybe_compact(messages)
            if edit:
                logger.emit("context_edit", turn=turn,
                            prompt_tokens_before=cm.last_prompt_tokens, **edit)

            wire = strip_internal_marks(messages)
            logger.emit("llm_request", turn=turn, model=cfg.model, n_messages=len(wire),
                        messages=wire, tools=[s["function"]["name"] for s in schemas],
                        params={"reasoning_effort": cfg.reasoning_effort,
                                "max_completion_tokens": cfg.max_completion_tokens})

            resp = provider.complete(
                wire, schemas,
                on_retry=lambda attempt, status, err, sleep_s: logger.emit(
                    "retry", turn=turn, attempt=attempt, status=status,
                    error=err[:500], sleep_s=sleep_s))

            cost = ledger.record(cfg.model, resp.usage)
            cm.observe(resp.usage.prompt_tokens)
            logger.emit("llm_response", turn=turn, finish_reason=resp.finish_reason,
                        content=resp.content,
                        tool_calls=[{"id": tc.id, "name": tc.name, "arguments": tc.arguments_raw}
                                    for tc in resp.tool_calls] or None,
                        usage=resp.usage.as_dict(), cost_usd=round(cost, 6),
                        request_id=resp.request_id, latency_ms=resp.latency_ms,
                        reasoning_content=resp.reasoning_content)

            messages.append(resp.to_assistant_message())

            if resp.tool_calls:
                messages.extend(_run_tool_calls(resp.tool_calls, tool_ctx, cfg, logger, turn))
                continue
            if resp.finish_reason == "length":
                if not nudged:
                    nudged = True
                    messages.append({"role": "user", "content": LENGTH_NUDGE})
                    continue
                reason, final = "truncated", resp.content
                break
            reason, final = "completed", resp.content
            break
    except KeyboardInterrupt:
        reason = "interrupted"
    except Exception as e:
        reason, final = "error", f"{type(e).__name__}: {e}"
        logger.emit("error", where="loop", error=final, traceback=traceback.format_exc())

    return logger.finish(reason, turn, ledger, final)


def _run_tool_calls(tool_calls: list[ToolCallRequest], tool_ctx: ToolContext,
                    cfg: Config, logger: RunLogger, turn: int) -> list[dict]:
    """执行一轮的全部工具调用，为每个 tool_call_id 生成应答消息（顺序保持）。"""
    for tc in tool_calls:
        logger.emit("tool_call", turn=turn, tool_call_id=tc.id, name=tc.name,
                    arguments=tc.arguments if tc.arguments is not None else tc.arguments_raw)

    def run_one(tc: ToolCallRequest) -> ToolResult:
        if tc.parse_error:   # 模型吐了非法 JSON：本身就是要回传的错误
            return ToolResult(f"ERROR: {tc.parse_error}. Re-send the tool call with "
                              f"valid JSON arguments. Raw arguments were: {tc.arguments_raw[:300]}",
                              ok=False)
        allowed, why = gate_tool_call(tc.name, tc.arguments, cfg)
        if not allowed:
            return ToolResult(denial_message(tc.name, why), ok=False)
        return execute_tool(tc.name, tc.arguments, tool_ctx)

    if len(tool_calls) > 1:  # 模型并行发起的调用就真并行执行
        with ThreadPoolExecutor(max_workers=min(len(tool_calls), 8)) as pool:
            results = list(pool.map(run_one, tool_calls))
    else:
        results = [run_one(tool_calls[0])]

    out = []
    for tc, r in zip(tool_calls, results):
        logger.emit("tool_result", turn=turn, tool_call_id=tc.id, name=tc.name,
                    ok=r.ok, result=r.text, duration_ms=r.duration_ms, truncated=r.truncated)
        out.append(tool_message(tc.id, r.ok, r.text))
    return out


def tool_message(tool_call_id: str, ok: bool, text: str) -> dict:
    content = text if (ok or text.startswith("ERROR:")) else f"ERROR: {text}"
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def build_resume_messages(events: list[dict]) -> list[dict]:
    """从 trajectory 重建消息历史：最后一次 llm_request 的完整 messages
    + 其后的 assistant 响应与工具应答。这就是 run_id 可复现承诺的兑现。"""
    last_req_idx = max(i for i, e in enumerate(events) if e["type"] == "llm_request")
    messages: list[dict] = [dict(m) for m in events[last_req_idx]["messages"]]
    for e in events[last_req_idx + 1:]:
        if e["type"] == "llm_response":
            msg: dict = {"role": "assistant", "content": e.get("content")}
            if e.get("reasoning_content") is not None:
                msg["reasoning_content"] = e["reasoning_content"]
            if e.get("tool_calls"):
                msg["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in e["tool_calls"]]
            messages.append(msg)
        elif e["type"] == "tool_result":
            messages.append(tool_message(e["tool_call_id"], e.get("ok", True), e["result"]))
    return messages


def _openai_version() -> str:
    try:
        import openai
        return openai.__version__
    except Exception:
        return "unknown"
