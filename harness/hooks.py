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
    PermissionUpdate,
    ResolveOnce,
    apply_permission_update,
    find_matching_rule,
    is_sensitive_path,
    load_permission_context,
    persist_permission_updates,
    permission_rule_value_to_string,
    suggest_permission_rule_value,
    summarize_permission_update,
)
from .tools.registry import ToolContext, find_tool_spec, tool_property


def evaluate_tool_permission(name: str, arguments: dict, cfg: Config,
                             tool_ctx: ToolContext | None = None) -> PermissionDecision:
    """Evaluate rules and policy without resolving an ask prompt."""
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
        return PermissionDecision(
            "ask", why, reason_type="tool_policy", mode=context.mode,
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

    return PermissionDecision(
        "ask", "permission required before running this tool",
        reason_type="passthrough_to_ask", mode=context.mode,
        suggestions=("add an allow rule for repeated trusted commands",
                     "use --permission-mode acceptEdits for coding edits"))


def resolve_permission_decision(name: str, arguments: dict, cfg: Config,
                                tool_ctx: ToolContext | None,
                                decision: PermissionDecision) -> tuple[PermissionDecision, list[dict]]:
    """Resolve an ask decision, returning the final decision plus lifecycle events."""
    if decision.behavior != "ask":
        return decision, []

    events: list[dict] = [{
        "type": "tool_permission_wait",
        "decision": decision.behavior,
        "reason": decision.message,
        "reason_type": decision.reason_type,
        "mode": decision.mode,
        "rule": decision.rule,
        "source": decision.source,
        "safety_check": decision.safety_check,
        "suggestions": list(decision.suggestions),
    }]

    resolver_fn = getattr(tool_ctx.runtime, "permission_resolver", None) if tool_ctx else None
    if callable(resolver_fn):
        resolver = ResolveOnce()
        try:
            choice = resolver_fn(name, arguments, decision, cfg, tool_ctx)
            resolver.claim("tui", choice)
        except Exception as e:
            resolver.claim("tui", f"deny:{e}")
        claimed = str(resolver.value or "deny")
        final, update = _decision_from_choice(claimed, name, arguments, decision, cfg, tool_ctx)
        if update:
            events.append({"type": "tool_permission_update",
                           "persisted": update.destination in {"local", "project"},
                           "summary": summarize_permission_update(update),
                           "destination": update.destination,
                           "behavior": update.behavior,
                           "mode": update.mode})
        events.append(_resolved_event(final, resolver.claimed_by or "unknown"))
        return final, events

    if decision.mode == "dontAsk" or not sys.stdin.isatty():
        final = PermissionDecision(
            "deny", decision.message, reason_type=decision.reason_type,
            mode=decision.mode, rule=decision.rule, source=decision.source,
            safety_check=decision.safety_check, suggestions=decision.suggestions)
        events.append(_resolved_event(final, "noninteractive"))
        return final, events

    resolver = ResolveOnce()
    choice = _prompt_for_permission(name, arguments, decision)
    resolver.claim("stdin", choice)
    claimed = str(resolver.value or "deny")
    final, update = _decision_from_choice(claimed, name, arguments, decision, cfg, tool_ctx)
    if update:
        events.append({"type": "tool_permission_update",
                       "persisted": update.destination in {"local", "project"},
                       "summary": summarize_permission_update(update),
                       "destination": update.destination,
                       "behavior": update.behavior,
                       "mode": update.mode})
    events.append(_resolved_event(final, resolver.claimed_by or "unknown"))
    return final, events


def gate_tool_call(name: str, arguments: dict, cfg: Config,
                   tool_ctx: ToolContext | None = None) -> PermissionDecision:
    """Compatibility wrapper returning only the final decision."""
    decision = evaluate_tool_permission(name, arguments, cfg, tool_ctx)
    final, _ = resolve_permission_decision(name, arguments, cfg, tool_ctx, decision)
    return final


def denial_message(name: str, decision: PermissionDecision | str | None) -> str:
    why = decision.message if isinstance(decision, PermissionDecision) else decision
    why = why or "permission denied"
    return (f"ERROR: tool call blocked by safety policy ({why}). "
            "Do NOT retry the same command rephrased. Choose a safer approach: "
            "operate only on files inside the workspace, avoid destructive/system-level "
            "commands, or accomplish the goal with the dedicated file tools.")


def _prompt_for_permission(name: str, arguments: dict,
                           decision: PermissionDecision) -> str:
    rule = permission_rule_value_to_string(suggest_permission_rule_value(name, arguments))
    print(f"\n[permission] {decision.message}: {name} {arguments}")
    print(f"[permission] suggested rule: {rule}")
    print("[permission] y=allow once, n=deny, s=allow session, l=allow local, p=allow project")
    try:
        answer = input("Allow? [y/N/s/l/p] ").strip().lower()
    except EOFError:
        return "deny"
    return answer or "deny"


def _decision_from_choice(choice: str, name: str, arguments: dict,
                          decision: PermissionDecision, cfg: Config,
                          tool_ctx: ToolContext | None) -> tuple[PermissionDecision, PermissionUpdate | None]:
    if choice in {"y", "yes"}:
        return PermissionDecision("allow", "allowed once by user",
                                  reason_type="interactive", mode=decision.mode), None
    if choice in {"s", "session", "l", "local", "p", "project"}:
        destination = {"s": "session", "session": "session",
                       "l": "local", "local": "local",
                       "p": "project", "project": "project"}[choice]
        update = PermissionUpdate(
            "addRules", destination, "allow",
            (suggest_permission_rule_value(name, arguments),))
        if tool_ctx is not None:
            context = _permission_context(cfg, tool_ctx)
            tool_ctx.runtime.permission_context = apply_permission_update(context, update)
        if destination in {"local", "project"}:
            persist_permission_updates(cfg.workdir, (update,))
        return PermissionDecision(
            "allow", f"allowed by interactive {destination} rule",
            reason_type="interactive", mode=decision.mode), update
    return PermissionDecision("deny", decision.message, reason_type=decision.reason_type,
                              mode=decision.mode, rule=decision.rule,
                              source=decision.source, safety_check=decision.safety_check,
                              suggestions=decision.suggestions), None


def _resolved_event(decision: PermissionDecision, resolver: str) -> dict:
    return {
        "type": "tool_permission_resolved",
        "resolver": resolver,
        "ok": decision.allowed,
        "decision": decision.behavior,
        "reason": decision.message,
        "reason_type": decision.reason_type,
        "rule": decision.rule,
        "source": decision.source,
        "mode": decision.mode,
        "safety_check": decision.safety_check,
        "suggestions": list(decision.suggestions),
    }


def _permission_context(cfg: Config, tool_ctx: ToolContext | None) -> PermissionContext:
    if tool_ctx is None:
        return load_permission_context(cfg.workdir, _mode_override(cfg))
    cached = tool_ctx.runtime.permission_context
    if isinstance(cached, PermissionContext):
        return cached
    context = load_permission_context(cfg.workdir, _mode_override(cfg))
    tool_ctx.runtime.permission_context = context
    return context


def _mode_override(cfg: Config) -> str | None:
    return None if cfg.permission_mode == "default" else cfg.permission_mode


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
