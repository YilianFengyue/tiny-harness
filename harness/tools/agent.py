"""Agent tool: delegate scoped work to built-in sub-agents."""
from __future__ import annotations

from ..agents import format_agent_listing, get_agent_definition
from ..background_agents import BackgroundAgentManager
from .registry import ToolContext, ToolResult, tool


AGENT_TOOL_DESCRIPTION = (
    "Launch a specialized sub-agent for complex multi-step work. "
    "Available subagent_type values:\n"
    f"{format_agent_listing()}\n"
    "Use explore/plan/verify for read-only investigation or validation. "
    "Use general for scoped implementation. Set fork=true only when the child "
    "needs a snapshot of the parent conversation. The sub-agent returns a "
    "concise result to you; detailed trajectory is saved separately."
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
                "description": "Reserved for the next milestone; currently must be false/null",
            },
            "fork": {
                "type": ["boolean", "null"],
                "description": "Reserved for the next milestone; currently must be false/null",
            },
        },
    },
    destructive=False,
)
def agent_tool(ctx: ToolContext, description: str, prompt: str,
               subagent_type: str | None = None,
               run_in_background: bool | None = None,
               fork: bool | None = None) -> ToolResult:
    if ctx.runtime.agent_depth > 0:
        return ToolResult(
            "Sub-agents cannot launch other sub-agents in this milestone. "
            "Complete the delegated task directly with your available tools.",
            ok=False,
            error_kind="agent_recursion_blocked",
        )
    cfg = ctx.runtime.config
    workdir = getattr(cfg, "workdir", None)
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
        )
        return ToolResult(
            f"Background subagent launched.\n"
            f"agent_id: {record.agent_id}\n"
            f"agent_type: {agent.agent_type}\n"
            f"fork: {bool(fork)}\n"
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
