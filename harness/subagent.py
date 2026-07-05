"""Sub-agent runner built on the same loop/runtime as the main agent."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from .agents import AgentDefinition, agent_tool_names, agent_tool_schemas
from .cancel import CancellationToken
from .config import Config, load_pricing
from .context import ContextManager
from .loop import TerminalState, _run_agent_events, build_initial_messages
from .providers.base import Provider
from .telemetry import CostLedger, RunLogger, Usage
from .tools.registry import ToolContext, ToolRuntimeState

SubagentEventCallback = Callable[[dict], None]


@dataclass
class SubagentRunResult:
    agent_id: str
    agent_type: str
    run_id: str
    status: str
    final_message: str | None
    turns: int
    tool_count: int
    usage: Usage
    cost_usd: float
    trajectory_path: str

    def as_tool_text(self) -> str:
        message = (self.final_message or "").strip() or "(no final message)"
        return (
            f"Subagent {self.agent_type} completed with status={self.status}\n"
            f"agent_id: {self.agent_id}\n"
            f"run_id: {self.run_id}\n"
            f"turns: {self.turns}\n"
            f"tool_count: {self.tool_count}\n"
            f"cost_usd: {self.cost_usd:.6f}\n"
            f"trajectory: {self.trajectory_path}\n\n"
            f"{message}"
        )


def run_subagent(agent: AgentDefinition, prompt: str, parent_ctx: ToolContext,
                 *, description: str,
                 emit: SubagentEventCallback | None = None,
                 cancel_token: CancellationToken | None = None,
                 fork_messages: list[dict] | None = None) -> SubagentRunResult:
    cfg = parent_ctx.runtime.config
    provider = parent_ctx.runtime.provider
    if not isinstance(cfg, Config):
        raise RuntimeError("agent tool requires runtime.config")
    if not isinstance(provider, Provider):
        raise RuntimeError("agent tool requires runtime.provider")

    child_provider = provider.spawn_child()
    child_cfg = _child_config(cfg, agent)
    logger = RunLogger(child_cfg.runs_dir)
    agent_id = logger.run_id

    def forward(event: dict) -> None:
        if emit:
            emit(event)
        if parent_ctx.runtime.event_callback:
            parent_ctx.runtime.event_callback(event)

    forward({
        "type": "agent_start",
        "agent_id": agent_id,
        "agent_type": agent.agent_type,
        "description": description,
        "prompt": prompt,
        "run_id": logger.run_id,
        "fork": bool(fork_messages),
        "tools": sorted(agent_tool_names(agent)),
    })

    child_runtime = ToolRuntimeState(
        permission_context=parent_ctx.runtime.permission_context,
        permission_resolver=parent_ctx.runtime.permission_resolver,
        config=child_cfg,
        provider=child_provider,
        agent_id=agent_id,
        agent_type=agent.agent_type,
        agent_depth=parent_ctx.runtime.agent_depth + 1,
        allowed_tools=agent_tool_names(agent),
        disallowed_tools=set(agent.disallowed_tools) | {"agent"},
        require_read_only_tools=agent.require_read_only_tools,
        event_callback=forward,
    )
    child_ctx = ToolContext(
        child_cfg.workdir,
        child_cfg.bash_timeout,
        child_cfg.tool_output_limit,
        cancel_token=cancel_token,
        runtime=child_runtime,
    )
    messages = _initial_messages(prompt, child_cfg, fork_messages)
    messages[0]["content"] = (
        str(messages[0].get("content") or "")
        + "\n\n# Subagent role\n"
        + agent.system_prompt
        + "\n\nYou are running as a sub-agent. Do not call the agent tool. "
          "Return a concise final report for the parent agent."
        + (
            "\n\n# Fork context\nYou inherited a snapshot of the parent conversation. "
            "Treat inherited content as historical context, not ground truth. "
            "Before editing or verifying current files, re-read them from disk."
            if fork_messages else ""
        )
    )

    ledger = CostLedger(load_pricing())
    cm = ContextManager(
        child_cfg.context_budget,
        child_cfg.context_keep_recent,
        child_cfg.context_hard_limit,
        child_cfg.tool_result_budget_chars,
        child_cfg.max_completion_tokens or 20_000,
    )
    schemas = agent_tool_schemas(agent)
    events = _run_agent_events(
        prompt,
        child_cfg,
        child_provider,
        schemas,
        ledger,
        cm,
        child_ctx,
        messages,
        session_id=parent_ctx.runtime.agent_id,
        cancel_token=cancel_token,
    )

    terminal: TerminalState | None = None
    tool_count = 0
    while True:
        try:
            event = next(events)
        except StopIteration as done:
            terminal = done.value
            break
        event = dict(event)
        if event.get("type") == "tool_start":
            tool_count += 1
            forward({
                "type": "agent_progress",
                "agent_id": agent_id,
                "agent_type": agent.agent_type,
                "phase": "tool_start",
                "tool_name": event.get("name"),
            })
        payload = dict(event)
        type_ = payload.pop("type")
        logger.emit(type_, **payload)

    terminal = terminal or TerminalState("error", 0, "subagent ended without terminal state")
    summary = logger.finish(terminal.reason, terminal.turns, ledger, terminal.final_message)
    usage = _usage_from_dict(summary["usage_total"])
    result = SubagentRunResult(
        agent_id=agent_id,
        agent_type=agent.agent_type,
        run_id=logger.run_id,
        status=summary["reason"],
        final_message=terminal.final_message,
        turns=terminal.turns,
        tool_count=tool_count,
        usage=usage,
        cost_usd=float(summary["cost_usd"]),
        trajectory_path=str(child_cfg.runs_dir / logger.run_id / "trajectory.jsonl"),
    )
    forward({
        "type": "agent_done" if result.status == "completed" else "agent_error",
        "agent_id": agent_id,
        "agent_type": agent.agent_type,
        "run_id": logger.run_id,
        "status": result.status,
        "fork": bool(fork_messages),
        "turns": result.turns,
        "tool_count": result.tool_count,
        "cost_usd": round(result.cost_usd, 6),
        "trajectory_path": result.trajectory_path,
        "final_message": result.final_message,
    })
    return result


def _child_config(cfg: Config, agent: AgentDefinition) -> Config:
    max_turns = agent.max_turns if agent.max_turns is not None else cfg.max_turns
    model = agent.model or cfg.model
    return replace(cfg, model=model, max_turns=min(max_turns, cfg.max_turns))


def _initial_messages(prompt: str, cfg: Config,
                      fork_messages: list[dict] | None) -> list[dict]:
    if not fork_messages:
        return build_initial_messages(prompt, cfg)
    messages = [_copy_message(message) for message in fork_messages]
    if not messages or messages[0].get("role") != "system":
        messages = build_initial_messages("", cfg)[:1] + messages
    messages.append({
        "role": "user",
        "content": (
            "[Fork subagent directive]\n"
            "You are a forked worker, not the main agent. Work only on this directive:\n"
            f"{prompt}"
        ),
    })
    return messages


def _copy_message(message: dict) -> dict:
    copied = dict(message)
    if isinstance(copied.get("tool_calls"), list):
        copied["tool_calls"] = [dict(item) for item in copied["tool_calls"]]
    return copied


def _usage_from_dict(raw: dict) -> Usage:
    return Usage(
        prompt_tokens=int(raw.get("prompt_tokens", 0)),
        cached_tokens=int(raw.get("cached_tokens", 0)),
        completion_tokens=int(raw.get("completion_tokens", 0)),
        reasoning_tokens=int(raw.get("reasoning_tokens", 0)),
    )
