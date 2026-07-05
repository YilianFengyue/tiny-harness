"""Agent tool: delegate scoped work to built-in and project sub-agents."""
from __future__ import annotations

from ..coordinator import is_coordinator_mode
from ..agents import format_agent_listing, get_agent_definition
from ..background_agents import BackgroundAgentManager
from .registry import ToolContext, ToolResult, tool


def agent_tool_description(workdir=None, coordinator_mode: bool = False) -> str:
    if coordinator_mode:
        return (
            "Coordinator-only worker launcher. Use this to spawn asynchronous "
            "workers for research, implementation, or verification, or to send "
            "a follow-up message to an existing stopped worker by passing its "
            "agent_id. In Coordinator mode use subagent_type=worker (or null, "
            "which defaults to worker); workers run in the background and "
            "report back through <task-notification>. Worker prompts must be "
            "self-contained with specific files, facts, and acceptance criteria."
        )
    return (
        "Launch a specialized sub-agent for complex multi-step work. "
        "Available subagent_type values for the current workspace:\n"
        f"{format_agent_listing(workdir)}\n"
        "Project agents are loaded from .tiny-harness/agents/*.md. "
        "Use explore/plan/verify for read-only investigation or validation, "
        "custom project agents for domain reviews, and general for scoped "
        "implementation. Set fork=true when the child needs a snapshot of the "
        "parent conversation. Set run_in_background=true for long verification "
        "or parallel review; the parent will receive a completion notification. "
        "The sub-agent returns a concise result to you; detailed trajectory is "
        "saved separately."
    )


AGENT_TOOL_DESCRIPTION = (
    "Launch a specialized sub-agent for complex multi-step work. "
    "Built-in subagent_type values are explore, plan, general, and verify. "
    "Project agents are also available from .tiny-harness/agents/*.md. "
    "Use fork=true to pass a parent context snapshot; use "
    "run_in_background=true for long verification or parallel review."
)


@tool(
    name="agent",
    description=AGENT_TOOL_DESCRIPTION,
    parameters={
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Short 3-5 word description of the delegated task",
            },
            "prompt": {
                "type": "string",
                "description": "Complete task instructions for the sub-agent",
            },
            "subagent_type": {
                "type": ["string", "null"],
                "description": "explore, plan, general, or verify (null = general)",
            },
            "run_in_background": {
                "type": ["boolean", "null"],
                "description": (
                    "When true, run the sub-agent in the background. The parent "
                    "conversation receives a completion notification on a later "
                    "turn and /agents shows status."
                ),
            },
            "fork": {
                "type": ["boolean", "null"],
                "description": (
                    "When true, pass a snapshot of the parent conversation to "
                    "the sub-agent before the delegated prompt."
                ),
            },
            "agent_id": {
                "type": ["string", "null"],
                "description": (
                    "Coordinator mode only: existing worker id to resume via "
                    "SendMessage. Use null to create a new worker."
                ),
            },
        },
    },
    destructive=False,
)
def agent_tool(ctx: ToolContext, description: str, prompt: str,
               subagent_type: str | None = None,
               run_in_background: bool | None = None,
               fork: bool | None = None,
               agent_id: str | None = None) -> ToolResult:
    if ctx.runtime.agent_depth > 0:
        return ToolResult(
            "Sub-agents cannot launch other sub-agents in this milestone. "
            "Complete the delegated task directly with your available tools.",
            ok=False,
            error_kind="agent_recursion_blocked",
        )
    cfg = ctx.runtime.config
    workdir = getattr(cfg, "workdir", None)
    coordinator = is_coordinator_mode(cfg)
    if coordinator:
        if subagent_type not in {None, "worker"}:
            return ToolResult(
                "Coordinator mode can only launch subagent_type='worker'. "
                "Rewrite the worker prompt as a self-contained task and call "
                "agent again with subagent_type='worker'.",
                ok=False,
                error_kind="coordinator_worker_required",
            )
        subagent_type = "worker"
        run_in_background = True
        if agent_id:
            manager = _background_manager(ctx)
            record, error = manager.send_message(
                agent_id,
                prompt,
                ctx,
                description=description,
                emit=None,
            )
            if error:
                return ToolResult(
                    f"ERROR: {error}. Use /agents or the latest "
                    "<task-notification><task-id> value to choose a completed "
                    "resumable worker, or call agent with agent_id=null to "
                    "launch a new worker.",
                    ok=False,
                    error_kind="coordinator_send_message_failed",
                )
            if record is None:
                return ToolResult(
                    f"ERROR: unknown worker agent_id {agent_id!r}.",
                    ok=False,
                    error_kind="coordinator_send_message_failed",
                )
            ctx.progress(
                type="agent_background_start",
                agent_id=record.agent_id,
                agent_type=record.agent_type,
                description=description,
                fork=record.fork,
                mode="coordinator",
                resumed=True,
                resume_count=record.resume_count,
            )
            return ToolResult(
                "Worker resumed with Coordinator SendMessage.\n"
                f"agent_id: {record.agent_id}\n"
                f"agent_type: {record.agent_type}\n"
                f"resume_count: {record.resume_count}\n"
                "The parent conversation will receive a task-notification "
                "when this resumed run completes.",
            )
    elif agent_id:
        return ToolResult(
            "agent_id resume is only available in Coordinator mode. "
            "Use run_in_background/fork for CH09-style sub-agents.",
            ok=False,
            error_kind="agent_resume_requires_coordinator",
        )
    agent = get_agent_definition(subagent_type, workdir)
    if agent is None:
        return ToolResult(
            f"Unknown subagent_type {subagent_type!r}. Available agents:\n"
            f"{format_agent_listing(workdir)}",
            ok=False,
            error_kind="unknown_agent",
        )

    from ..subagent import run_subagent

    fork_messages = _fork_messages(ctx) if fork else None
    if fork and not fork_messages:
        return ToolResult(
            "fork=true requested, but no parent conversation snapshot is available.",
            ok=False,
            error_kind="fork_context_missing",
        )

    if run_in_background or agent.background:
        manager = _background_manager(ctx)
        record = manager.start(
            agent,
            prompt,
            ctx,
            description=description,
            fork=bool(fork),
            fork_messages=fork_messages,
            emit=None,
        )
        ctx.progress(
            type="agent_background_start",
            agent_id=record.agent_id,
            agent_type=agent.agent_type,
            description=description,
            fork=bool(fork),
            mode="coordinator" if coordinator else "normal",
            resumed=False,
            resume_count=record.resume_count,
        )
        mode_line = "mode: coordinator\n" if coordinator else ""
        return ToolResult(
            f"Background subagent launched.\n"
            f"agent_id: {record.agent_id}\n"
            f"agent_type: {agent.agent_type}\n"
            f"fork: {bool(fork)}\n"
            f"{mode_line}"
            "The parent conversation will receive a notification when it completes. "
            "Use /agents in the TUI to inspect background agents.",
        )

    result = run_subagent(
        agent,
        prompt,
        ctx,
        description=description,
        emit=lambda event: ctx.progress(**event),
        cancel_token=ctx.cancel_token,
        fork_messages=fork_messages,
    )
    return ToolResult(result.as_tool_text(), ok=result.status == "completed")


def _fork_messages(ctx: ToolContext) -> list[dict] | None:
    if ctx.runtime.messages is None:
        return None
    return [dict(message) for message in ctx.runtime.messages]


def _background_manager(ctx: ToolContext) -> BackgroundAgentManager:
    manager = ctx.runtime.background_agents
    if isinstance(manager, BackgroundAgentManager):
        return manager
    manager = BackgroundAgentManager()
    ctx.runtime.background_agents = manager
    return manager
