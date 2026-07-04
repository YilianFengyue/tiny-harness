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
from dataclasses import dataclass, field
from queue import Empty, Queue
import time

from .cancel import CancelledError, CancellationToken
from .config import Config, load_pricing
from .context import ContextManager, strip_internal_marks
from .hooks import (
    denial_message,
    evaluate_tool_permission,
    resolve_permission_decision,
)
from .features import feature_snapshot
from .memory import memory_summary, render_memory_prompt
from .memory_extract import MemoryExtractionController
from .providers.base import ModelTurn, Provider, ToolCallRequest
from .skills import render_skills_section
from .telemetry import CostLedger, RunLogger
from .tools import ToolContext, execute_tool, openai_tool_schemas
from .tools.registry import (
    ToolResult,
    available_tool_names,
    find_tool_spec,
    tool_property,
    validate_tool_input,
)

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


@dataclass(frozen=True)
class ContinueDecision:
    """Why the loop is continuing into another model turn."""
    reason: str
    detail: dict = field(default_factory=dict)

    def as_event(self, turn: int) -> dict:
        return {"type": "transition", "turn": turn, "kind": "continue",
                "reason": self.reason, **self.detail}


@dataclass(frozen=True)
class TerminalState:
    """Why the loop stopped."""
    reason: str
    turns: int
    final_message: str | None = None


@dataclass
class AgentState:
    """Mutable loop state carried across turns.

    The loop reads a snapshot at the top of each turn and writes one explicit
    transition before continuing. This mirrors the Ch02 State/Continue/Terminal
    model without importing Claude Code's unrelated product machinery.
    """
    messages: list[dict]
    turn: int = 0
    transition: ContinueDecision | None = None
    length_recovery_count: int = 0
    reactive_compact_attempted: bool = False


def build_initial_messages(task: str, cfg: Config) -> list[dict]:
    system = SYSTEM_PROMPT.format(workdir=cfg.workdir,
                                  skills=render_skills_section(cfg.skills))
    system += render_memory_prompt(cfg)
    return [{"role": "system", "content": system},
            {"role": "user", "content": task}]


def run_agent(task: str | None, cfg: Config, provider: Provider,
              logger: RunLogger, resume_messages: list[dict] | None = None) -> dict:
    """跑一个任务，返回 summary dict（同时落盘 summary.json）。"""
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    messages = resume_messages or build_initial_messages(task or "", cfg)
    schemas = openai_tool_schemas()
    ledger = CostLedger(load_pricing())
    cm = ContextManager(cfg.context_budget, cfg.context_keep_recent,
                        cfg.context_hard_limit, cfg.tool_result_budget_chars)
    tool_ctx = ToolContext(cfg.workdir, cfg.bash_timeout, cfg.tool_output_limit)
    tool_ctx.runtime.config = cfg

    events = _run_agent_events(task, cfg, provider, schemas, ledger, cm, tool_ctx, messages)
    terminal: TerminalState | None = None
    while True:
        try:
            event = next(events)
        except StopIteration as done:
            terminal = done.value
            break
        event = dict(event)
        type_ = event.pop("type")
        logger.emit(type_, **event)

    terminal = terminal or TerminalState("error", 0, "loop ended without terminal state")
    if terminal.reason == "completed":
        controller = MemoryExtractionController(cfg)
        controller.extract(
            messages,
            emit=lambda event: _emit_event_dict(logger, event),
        )
    return logger.finish(terminal.reason, terminal.turns, ledger, terminal.final_message)


def _run_agent_events(task: str | None, cfg: Config, provider: Provider,
                      schemas: list[dict], ledger: CostLedger,
                      cm: ContextManager, tool_ctx: ToolContext,
                      messages: list[dict], session_id: str | None = None,
                      cancel_token: CancellationToken | None = None):
    state = AgentState(messages=messages)
    settings_snapshot = cfg.settings_snapshot
    mem_summary = memory_summary(cfg)
    yield {
        "type": "run_start",
        "task": task,
        "model": cfg.model,
        "workdir": str(cfg.workdir),
        "config": {"max_turns": cfg.max_turns, "max_cost_usd": cfg.max_cost_usd,
                   "context_budget": cfg.context_budget,
                   "reasoning_effort": cfg.reasoning_effort,
                   "permission_mode": cfg.permission_mode,
                   "yolo": cfg.yolo},
        "sdk_version": _openai_version(),
        "skills": cfg.skills,
        "session_id": session_id,
        "settings_sources": [
            {"source": layer.source, "path": layer.path, "origin": layer.origin}
            for layer in settings_snapshot.sources
        ] if settings_snapshot else [],
        "settings_policy_origin": (
            settings_snapshot.policy_origin if settings_snapshot else None),
        "settings_errors": [
            {"source": e.source, "path": e.path, "message": e.message}
            for e in settings_snapshot.errors
        ] if settings_snapshot else [],
        "features": feature_snapshot(cfg),
        "memory": mem_summary,
    }
    if mem_summary["count"]:
        yield {"type": "memory_load", **mem_summary}

    try:
        while state.turn < cfg.max_turns:
            if cancel_token:
                cancel_token.throw_if_cancelled()
            if ledger.cost_usd >= cfg.max_cost_usd:
                return TerminalState("max_cost", state.turn)

            state.turn += 1
            yield {"type": "turn_start", "turn": state.turn,
                   "transition": state.transition.reason if state.transition else None,
                   "n_messages": len(state.messages)}

            for edit in (cm.budget_tool_results(state.messages),
                         cm.maybe_compact(state.messages)):
                if edit:
                    yield {"type": "context_edit", "turn": state.turn,
                           "prompt_tokens_before": cm.last_prompt_tokens, **edit}

            hard = cm.hard_limit_exceeded(state.messages)
            if hard:
                yield {"type": "error", "where": "context", "error": "blocking_limit",
                       **hard}
                return TerminalState("blocking_limit", state.turn,
                                     f"Prompt estimate {hard['prompt_tokens_estimate']} "
                                     f"exceeds hard limit {hard['hard_limit_tokens']}.")

            wire = strip_internal_marks(state.messages)
            yield {"type": "llm_request", "turn": state.turn, "model": cfg.model,
                   "n_messages": len(wire), "messages": wire,
                   "tools": [s["function"]["name"] for s in schemas],
                              "params": {"reasoning_effort": cfg.reasoning_effort,
                              "max_completion_tokens": cfg.max_completion_tokens}}

            retry_events: list[dict] = []
            yield {"type": "stream_request_start", "turn": state.turn,
                   "model": cfg.model}
            try:
                stream = provider.stream(
                    wire, schemas,
                    on_retry=lambda attempt, status, err, sleep_s: retry_events.append(
                        {"type": "retry", "turn": state.turn, "attempt": attempt,
                         "status": status, "error": err[:500], "sleep_s": sleep_s}),
                    cancel_token=cancel_token)
                while True:
                    try:
                        event = next(stream)
                    except StopIteration as done:
                        resp: ModelTurn = done.value
                        break
                    for retry_event in retry_events:
                        yield retry_event
                    retry_events.clear()
                    if event.get("type") == "assistant_delta":
                        yield {"turn": state.turn, **event}
                    else:
                        yield {"type": "provider_event", "turn": state.turn, **event}
                for retry_event in retry_events:
                    yield retry_event
            except CancelledError:
                return TerminalState("aborted_streaming", state.turn)
            except KeyboardInterrupt:
                if cancel_token:
                    cancel_token.cancel()
                return TerminalState("aborted_streaming", state.turn)
            except Exception as e:
                if _is_prompt_too_long_error(e):
                    edit = None if state.reactive_compact_attempted else cm.reactive_compact(state.messages)
                    if edit:
                        state.reactive_compact_attempted = True
                        yield {"type": "context_edit", "turn": state.turn,
                               "prompt_tokens_before": cm.last_prompt_tokens, **edit}
                        state.transition = ContinueDecision(
                            "reactive_compact_retry", {"error": str(e)[:500]})
                        yield state.transition.as_event(state.turn)
                        continue
                    return TerminalState("prompt_too_long", state.turn, str(e)[:1000])
                raise

            cost = ledger.record(cfg.model, resp.usage)
            cm.observe(resp.usage.prompt_tokens)
            yield {"type": "llm_response", "turn": state.turn,
                   "finish_reason": resp.finish_reason, "content": resp.content,
                   "tool_calls": [{"id": tc.id, "name": tc.name,
                                   "arguments": tc.arguments_raw}
                                  for tc in resp.tool_calls] or None,
                   "usage": resp.usage.as_dict(), "cost_usd": round(cost, 6),
                   "request_id": resp.request_id, "latency_ms": resp.latency_ms,
                   "reasoning_content": resp.reasoning_content}

            state.messages.append(resp.to_assistant_message())

            if resp.tool_calls:
                tool_messages = []
                try:
                    for event in _run_tool_call_events(resp.tool_calls, tool_ctx, cfg,
                                                       state.turn, cancel_token):
                        if event["type"] == "tool_result_message":
                            tool_messages.append(event["message"])
                            continue
                        yield event
                except CancelledError:
                    return TerminalState("aborted_tools", state.turn)
                state.messages.extend(tool_messages)
                if state.turn >= cfg.max_turns:
                    return TerminalState("max_turns", state.turn)
                state.transition = ContinueDecision(
                    "next_turn", {"tool_calls": len(resp.tool_calls)})
                yield state.transition.as_event(state.turn)
                continue

            if resp.finish_reason == "length":
                if state.length_recovery_count < 1:
                    state.length_recovery_count += 1
                    state.messages.append({"role": "user", "content": LENGTH_NUDGE})
                    state.transition = ContinueDecision(
                        "output_recovery", {"attempt": state.length_recovery_count})
                    yield state.transition.as_event(state.turn)
                    continue
                return TerminalState("truncated", state.turn, resp.content)

            return TerminalState("completed", state.turn, resp.content)
        return TerminalState("max_turns", state.turn)
    except KeyboardInterrupt:
        return TerminalState("interrupted", state.turn)
    except CancelledError:
        return TerminalState("interrupted", state.turn)
    except Exception as e:
        final = f"{type(e).__name__}: {e}"
        yield {"type": "error", "where": "loop", "error": final,
               "traceback": traceback.format_exc()}
        return TerminalState("model_error", state.turn, final)


def _run_tool_call_events(tool_calls: list[ToolCallRequest], tool_ctx: ToolContext,
                          cfg: Config, turn: int,
                          cancel_token: CancellationToken | None = None):
    """Execute tool calls while yielding live lifecycle/progress events."""
    for batch in _partition_tool_calls(tool_calls):
        yield from _run_tool_batch_events(batch, tool_ctx, cfg, turn, cancel_token)


def _partition_tool_calls(tool_calls: list[ToolCallRequest]) -> list[list[ToolCallRequest]]:
    batches: list[list[ToolCallRequest]] = []
    batch_safe = False
    for tc in tool_calls:
        spec = find_tool_spec(tc.name)
        safe = (
            tc.parse_error is None
            and tc.arguments is not None
            and tool_property(spec, "concurrency_safe", tc.arguments)
        )
        if safe and batches and batch_safe:
            batches[-1].append(tc)
        else:
            batches.append([tc])
            batch_safe = safe
    return batches


def _run_tool_batch_events(batch: list[ToolCallRequest], tool_ctx: ToolContext,
                           cfg: Config, turn: int,
                           cancel_token: CancellationToken | None = None):
    progress_events: Queue[dict] = Queue()
    prepared: list[tuple[ToolCallRequest, ToolResult | None]] = []

    for tc in batch:
        for event in _preflight_tool_events(tc, tool_ctx, cfg, turn):
            if event["type"] == "tool_preflight_result":
                prepared.append((tc, event["result"]))
                break
            yield event
        else:
            prepared.append((tc, None))

    runnable = [(tc, result) for tc, result in prepared if result is None]
    finished: dict[str, ToolResult] = {
        tc.id: result for tc, result in prepared if result is not None
    }

    if runnable:
        workers = min(len(runnable), 8)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_by_id = {}
            for tc, _ in runnable:
                def on_progress(payload: dict, tc=tc) -> None:
                    progress_events.put({"type": "tool_progress", "turn": turn,
                                         "tool_call_id": tc.id, "name": tc.name,
                                         **payload})

                child_ctx = ToolContext(
                    tool_ctx.workdir, tool_ctx.bash_timeout, tool_ctx.output_limit,
                    cancel_token=cancel_token, progress_callback=on_progress,
                    runtime=tool_ctx.runtime)

                def run_one(tc=tc, child_ctx=child_ctx) -> ToolResult:
                    if cancel_token:
                        cancel_token.throw_if_cancelled()
                    return execute_tool(tc.name, tc.arguments or {}, child_ctx)

                yield {"type": "tool_start", "turn": turn, "tool_call_id": tc.id,
                       "name": tc.name, "arguments": tc.arguments}
                future_by_id[tc.id] = pool.submit(run_one)

            try:
                while any(not f.done() for f in future_by_id.values()):
                    if cancel_token and cancel_token.is_cancelled:
                        raise CancelledError("tool execution cancelled")
                    yield from _drain_progress(progress_events)
                    time.sleep(0.05)
                yield from _drain_progress(progress_events)
                for tc, _ in runnable:
                    finished[tc.id] = future_by_id[tc.id].result()
            except KeyboardInterrupt:
                if cancel_token:
                    cancel_token.cancel()
                raise CancelledError("tool execution interrupted")

    for tc, _ in prepared:
        yield from _finish_tool_event(turn, tc, finished[tc.id], tool_ctx)


def _preflight_tool_events(tc: ToolCallRequest, tool_ctx: ToolContext,
                           cfg: Config, turn: int):
    arguments = tc.arguments if tc.arguments is not None else tc.arguments_raw
    yield {"type": "tool_call", "turn": turn, "tool_call_id": tc.id,
           "name": tc.name, "arguments": arguments}
    yield {"type": "tool_queued", "turn": turn, "tool_call_id": tc.id,
           "name": tc.name, "arguments": arguments}

    if tc.parse_error:
        yield {"type": "tool_validate", "turn": turn, "tool_call_id": tc.id,
               "name": tc.name, "ok": False, "error": tc.parse_error}
        result = ToolResult(
            f"ERROR: {tc.parse_error}. Re-send the tool call with valid JSON "
            f"arguments. Raw arguments were: {tc.arguments_raw[:300]}",
            ok=False, error_kind="invalid_json")
        yield {"type": "tool_preflight_result", "result": result}
        return

    spec = find_tool_spec(tc.name)
    if spec is None:
        yield {"type": "tool_validate", "turn": turn, "tool_call_id": tc.id,
               "name": tc.name, "ok": False, "error": "unknown_tool"}
        result = ToolResult(
            f"Unknown tool '{tc.name}'. Available tools: {available_tool_names()}.",
            ok=False, error_kind="unknown_tool")
        yield {"type": "tool_preflight_result", "result": result}
        return

    try:
        validate_tool_input(spec.name, tc.arguments or {}, tool_ctx)
    except Exception as e:
        yield {"type": "tool_validate", "turn": turn, "tool_call_id": tc.id,
               "name": spec.name, "ok": False, "error": str(e)}
        yield {"type": "tool_preflight_result",
               "result": ToolResult(str(e), ok=False, error_kind="validation")}
        return

    yield {"type": "tool_validate", "turn": turn, "tool_call_id": tc.id,
           "name": spec.name, "ok": True,
           "read_only": tool_property(spec, "read_only", tc.arguments or {}),
           "concurrency_safe": tool_property(spec, "concurrency_safe", tc.arguments or {}),
           "destructive": tool_property(spec, "destructive", tc.arguments or {})}

    decision = evaluate_tool_permission(spec.name, tc.arguments or {}, cfg, tool_ctx)
    yield {"type": "tool_permission", "turn": turn, "tool_call_id": tc.id,
           "name": spec.name, "ok": decision.allowed,
           "decision": decision.behavior, "reason": decision.message,
           "reason_type": decision.reason_type, "rule": decision.rule,
           "source": decision.source, "mode": decision.mode,
           "safety_check": decision.safety_check,
           "suggestions": list(decision.suggestions)}
    if decision.behavior == "ask":
        decision, permission_events = resolve_permission_decision(
            spec.name, tc.arguments or {}, cfg, tool_ctx, decision)
        for event in permission_events:
            event = dict(event)
            type_ = event.pop("type")
            yield {"type": type_, "turn": turn, "tool_call_id": tc.id,
                   "name": spec.name, **event}
    if not decision.allowed:
        yield {"type": "tool_preflight_result",
               "result": ToolResult(denial_message(spec.name, decision), ok=False,
                                    error_kind="permission_denied")}


def _drain_progress(progress_events: Queue[dict]):
    while True:
        try:
            yield progress_events.get_nowait()
        except Empty:
            return


def _finish_tool_event(turn: int, tc: ToolCallRequest, r: ToolResult,
                       tool_ctx: ToolContext):
    if r.persisted_path:
        yield {"type": "tool_result_persisted", "turn": turn,
               "tool_call_id": tc.id, "name": tc.name,
               "path": r.persisted_path}
    yield {"type": "tool_result", "turn": turn, "tool_call_id": tc.id,
           "name": tc.name, "ok": r.ok, "result": r.text,
           "duration_ms": r.duration_ms, "truncated": r.truncated,
           "persisted_path": r.persisted_path, "error_kind": r.error_kind}
    yield {"type": "tool_end", "turn": turn, "tool_call_id": tc.id,
           "name": tc.name, "ok": r.ok, "duration_ms": r.duration_ms,
           "truncated": r.truncated}
    if r.context_modifier:
        payload = r.context_modifier(tool_ctx.runtime) or {}
        r.context_modified = payload
        yield {"type": "tool_context_modified", "turn": turn,
               "tool_call_id": tc.id, "name": tc.name, **payload}
    yield {"type": "tool_result_message", "message": tool_message(tc.id, r.ok, r.text)}


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


def _is_prompt_too_long_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    hints = ("context length", "maximum context", "prompt too long",
             "too many tokens", "tokens exceeds", "context_length_exceeded")
    return status in (400, 413, 422) and any(h in text for h in hints)


def _openai_version() -> str:
    try:
        import openai
        return openai.__version__
    except Exception:
        return "unknown"


def _emit_event_dict(logger: RunLogger, event: dict) -> None:
    payload = dict(event)
    type_ = payload.pop("type")
    logger.emit(type_, **payload)
