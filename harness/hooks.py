"""Permission gate for tool calls.

Ch04 splits the gate into explicit rules, tool-level safety checks, permission
modes, and final interactive resolution. This file wires that core into the
current non-interactive loop while keeping the decision rich enough for TUI and
viewer rendering.
"""
from __future__ import annotations

import sys

from .config import Config
from .permissions import (
    PermissionContext,
    PermissionDecision,
    find_matching_rule,
    is_sensitive_path,
    load_permission_context,
    permission_rule_value_to_string,
)
from .tools.registry import ToolContext, find_tool_spec, tool_property


def gate_tool_call(name: str, arguments: dict, cfg: Config,
                   tool_ctx: ToolContext | None = None) -> PermissionDecision:
    """Return a structured permission decision for a tool call."""
    spec = find_tool_spec(name)
    context = _permission_context(cfg, tool_ctx)
    args = arguments or {}

    for behavior in ("deny", "ask"):
        rule = find_matching_rule(context, behavior, name, args)
        if rule:
            return _rule_decision(behavior, rule, context.mode)

    safety = _safety_decision(name, args, context.mode)
    if safety:
        return safety

    why = spec.dangerous_check(args) if spec and spec.dangerous_check else None
    if why:
        if cfg.yolo:
            return PermissionDecision(
                "allow", why, reason_type="yolo", mode=context.mode,
                suggestions=("yolo bypassed a legacy dangerous_check",))
        if sys.stdin.isatty():
            try:
                answer = input(f"\n[hook] dangerous operation ({why}): {args}\nAllow? [y/N] ")
                if answer.strip().lower() == "y":
                    return PermissionDecision("allow", why, reason_type="interactive",
                                              mode=context.mode)
            except EOFError:
                pass
        return PermissionDecision(
            "deny", why, reason_type="tool_policy", mode=context.mode,
            suggestions=("use dedicated file tools", "avoid system-level destructive commands"))

    read_only = tool_property(spec, "read_only", args)
    destructive = tool_property(spec, "destructive", args)
    concurrency_safe = tool_property(spec, "concurrency_safe", args)

    if context.mode == "plan" and not read_only:
        return PermissionDecision(
            "deny", "plan mode only allows read-only tools", reason_type="mode",
            mode=context.mode, suggestions=("switch to acceptEdits or bypass after planning",))

    if cfg.yolo:
        return PermissionDecision("allow", "allowed by --yolo", reason_type="yolo",
                                  mode=context.mode)
    if context.mode == "bypass":
        return PermissionDecision("allow", "allowed by bypass mode", reason_type="mode",
                                  mode=context.mode)

    allow_rule = find_matching_rule(context, "allow", name, args)
    if allow_rule:
        return _rule_decision("allow", allow_rule, context.mode)

    if read_only:
        return PermissionDecision(
            "allow", "read-only tool", reason_type="tool_metadata", mode=context.mode,
            suggestions=(f"concurrency_safe={concurrency_safe}",))

    if context.mode == "acceptEdits" and destructive and _is_file_edit_tool(name):
        return PermissionDecision("allow", "file edit allowed by acceptEdits mode",
                                  reason_type="mode", mode=context.mode)

    if not destructive and name != "bash":
        return PermissionDecision("allow", "tool has no destructive metadata",
                                  reason_type="passthrough", mode=context.mode)

    ask = PermissionDecision(
        "ask", "permission required before running this tool",
        reason_type="passthrough_to_ask", mode=context.mode,
        suggestions=("add an allow rule for repeated trusted commands",
                     "use --permission-mode acceptEdits for coding edits"))
    if context.mode == "dontAsk" or not sys.stdin.isatty():
        return PermissionDecision(
            "deny", ask.message, reason_type=ask.reason_type, mode=context.mode,
            suggestions=ask.suggestions)
    try:
        answer = input(f"\n[hook] {ask.message}: {name} {args}\nAllow? [y/N] ")
        if answer.strip().lower() == "y":
            return PermissionDecision("allow", "allowed interactively",
                                      reason_type="interactive", mode=context.mode)
    except EOFError:
        pass
    return PermissionDecision("deny", ask.message, reason_type=ask.reason_type,
                              mode=context.mode, suggestions=ask.suggestions)


def denial_message(name: str, decision: PermissionDecision | str | None) -> str:
    why = decision.message if isinstance(decision, PermissionDecision) else decision
    why = why or "permission denied"
    return (f"ERROR: tool call blocked by safety policy ({why}). "
            "Do NOT retry the same command rephrased. Choose a safer approach: "
            "operate only on files inside the workspace, avoid destructive/system-level "
            "commands, or accomplish the goal with the dedicated file tools.")


def _permission_context(cfg: Config, tool_ctx: ToolContext | None) -> PermissionContext:
    if tool_ctx is None:
        return load_permission_context(cfg.workdir, cfg.permission_mode)
    cached = tool_ctx.runtime.permission_context
    if isinstance(cached, PermissionContext):
        return cached
    context = load_permission_context(cfg.workdir, cfg.permission_mode)
    tool_ctx.runtime.permission_context = context
    return context


def _rule_decision(behavior: str, rule, mode: str) -> PermissionDecision:
    text = permission_rule_value_to_string(rule.value)
    return PermissionDecision(
        behavior, f"matched {rule.source} {behavior} rule {text}",
        reason_type="rule", rule=text, source=rule.source, mode=mode)


def _safety_decision(name: str, arguments: dict, mode: str) -> PermissionDecision | None:
    if not _is_file_edit_tool(name):
        return None
    path = arguments.get("path")
    if is_sensitive_path(str(path) if path is not None else None):
        return PermissionDecision(
            "deny", f"sensitive path requires explicit human review: {path}",
            reason_type="safety_check", mode=mode, safety_check=True,
            suggestions=("avoid editing secrets, git internals, and workflow files via automation",))
    return None


def _is_file_edit_tool(name: str) -> bool:
    return name in {"write_file", "edit_file"}
