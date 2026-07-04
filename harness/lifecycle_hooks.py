"""Settings-backed lifecycle hooks for CH08.

Tiny Harness starts with command hooks: deterministic scripts that receive a
JSON payload on stdin and may return structured JSON on stdout. More expensive
prompt/agent/http hooks can be layered on this contract later.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .permissions import (
    display_tool_name,
    permission_rule_value_from_string,
    rule_matches_tool,
)
from .settings import nested_get, trusted_security_settings

LifecycleEvent = Literal[
    "SessionStart",
    "SessionEnd",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "PreCompact",
    "PostCompact",
    "Stop",
]

SUPPORTED_LIFECYCLE_EVENTS: tuple[LifecycleEvent, ...] = (
    "SessionStart",
    "SessionEnd",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "PreCompact",
    "PostCompact",
    "Stop",
)


@dataclass(frozen=True)
class LifecycleHookDiagnostic:
    hook_id: str
    event: LifecycleEvent
    hook_type: str
    matcher: str
    source: str
    ok: bool
    blocked: bool = False
    reason: str | None = None
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_ms: int = 0

    def as_event(self, phase: str) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": f"hook_{phase}",
            "hook_id": self.hook_id,
            "hook_event": self.event,
            "hook_type": self.hook_type,
            "matcher": self.matcher,
            "source": self.source,
        }
        if phase == "end":
            data.update({
                "ok": self.ok,
                "blocked": self.blocked,
                "reason": self.reason,
                "stdout": self.stdout[:2000],
                "stderr": self.stderr[:2000],
                "exit_code": self.exit_code,
                "duration_ms": self.duration_ms,
            })
        return data


@dataclass(frozen=True)
class LifecycleHookResult:
    event: LifecycleEvent
    blocked: bool = False
    reason: str | None = None
    updated_input: dict[str, Any] | None = None
    additional_context: str | None = None
    updated_output: str | None = None
    diagnostics: list[LifecycleHookDiagnostic] = field(default_factory=list)


def dispatch_lifecycle_hooks(event: LifecycleEvent,
                             payload: dict[str, Any] | None = None,
                             cfg: Any | None = None) -> LifecycleHookResult:
    """Dispatch matching lifecycle hooks and merge their structured results."""
    if event not in SUPPORTED_LIFECYCLE_EVENTS:
        raise ValueError(f"unsupported lifecycle event: {event}")
    payload = payload or {}
    hooks = _matching_hooks(event, payload, cfg)
    blocked = False
    reason: str | None = None
    updated_input: dict[str, Any] | None = None
    additional: list[str] = []
    updated_output: str | None = None
    diagnostics: list[LifecycleHookDiagnostic] = []

    for hook_id, source, matcher, hook in hooks:
        started = LifecycleHookDiagnostic(
            hook_id, event, str(hook.get("type") or "command"), matcher, source,
            ok=True)
        diagnostics.append(started)
        if hook.get("type", "command") != "command":
            diagnostics.append(LifecycleHookDiagnostic(
                hook_id, event, str(hook.get("type") or "unknown"), matcher, source,
                ok=False, reason="unsupported hook type"))
            continue
        diag, parsed = _run_command_hook(hook_id, event, source, matcher, hook, payload, cfg)
        diagnostics.append(diag)
        if diag.blocked:
            blocked = True
            reason = diag.reason or reason
        if parsed:
            hook_output = parsed.get("hookSpecificOutput")
            if not isinstance(hook_output, dict):
                hook_output = {}
            decision = parsed.get("decision") or hook_output.get("permissionDecision")
            if decision in {"block", "deny"}:
                blocked = True
                reason = str(parsed.get("reason")
                             or hook_output.get("permissionDecisionReason")
                             or diag.reason
                             or "blocked by lifecycle hook")
            if parsed.get("continue") is False:
                blocked = True
                reason = str(parsed.get("stopReason")
                             or parsed.get("reason")
                             or "stopped by lifecycle hook")
            candidate_input = parsed.get("updatedInput", hook_output.get("updatedInput"))
            if isinstance(candidate_input, dict):
                updated_input = candidate_input
            candidate_context = parsed.get(
                "additionalContext", hook_output.get("additionalContext"))
            if candidate_context:
                additional.append(str(candidate_context))
            candidate_output = parsed.get(
                "updatedOutput", hook_output.get("updatedMCPToolOutput"))
            if candidate_output is not None:
                updated_output = str(candidate_output)
        if blocked:
            break

    return LifecycleHookResult(
        event=event,
        blocked=blocked,
        reason=reason,
        updated_input=updated_input,
        additional_context="\n\n".join(additional) if additional else None,
        updated_output=updated_output,
        diagnostics=diagnostics,
    )


def lifecycle_hook_events(result: LifecycleHookResult, **extra: Any):
    last_end = -1
    for index, diag in enumerate(result.diagnostics):
        if diag.duration_ms or diag.exit_code is not None or diag.reason:
            last_end = index
    for index, diag in enumerate(result.diagnostics):
        phase = "start" if diag.duration_ms == 0 and diag.exit_code is None and not diag.reason else "end"
        event = diag.as_event(phase)
        if phase == "end" and index == last_end and result.blocked:
            event["blocked"] = True
            event["reason"] = result.reason or event.get("reason")
        yield {**event, **extra}


def hooks_status(cfg: Any) -> dict[str, Any]:
    snapshot = getattr(cfg, "settings_snapshot", None)
    if snapshot is None:
        return {"enabled": False, "trusted": False, "count": 0, "sources": []}
    trusted = _hooks_trusted(cfg)
    hooks = _hook_layers(cfg, include_untrusted=trusted)
    return {
        "enabled": bool(hooks),
        "trusted": trusted,
        "count": sum(len(items) for _source, items in hooks),
        "sources": [source for source, items in hooks if items],
    }


def hooks_status_line(cfg: Any) -> str:
    status = hooks_status(cfg)
    if not status["enabled"]:
        return "hooks:off"
    trust = "trusted" if status["trusted"] else "untrusted"
    return f"hooks:on {trust} count={status['count']}"


def format_hooks_summary(cfg: Any) -> str:
    status = hooks_status(cfg)
    lines = [
        "Lifecycle hooks",
        f"enabled: {status['enabled']}",
        f"trusted: {status['trusted']}",
        f"count: {status['count']}",
    ]
    sources = status.get("sources") or []
    lines.append("sources: " + (", ".join(sources) if sources else "none"))
    if not status["trusted"]:
        lines.append("projectSettings hooks are skipped until hooksTrusted=true is set in a trusted source")
    snapshot = getattr(cfg, "settings_snapshot", None)
    if snapshot is None:
        return "\n".join(lines)
    configured: list[str] = []
    for source, hooks in _hook_layers(cfg, include_untrusted=True):
        events = [name for name in SUPPORTED_LIFECYCLE_EVENTS if isinstance(hooks.get(name), list)]
        if events:
            configured.append(f"- {source}: {', '.join(events)}")
    if configured:
        lines.append("configured events:")
        lines.extend(configured)
    return "\n".join(lines)


def _matching_hooks(event: LifecycleEvent, payload: dict[str, Any],
                    cfg: Any | None) -> list[tuple[str, str, str, dict]]:
    if cfg is None:
        return []
    result: list[tuple[str, str, str, dict]] = []
    for source, hooks_root in _hook_layers(cfg, include_untrusted=_hooks_trusted(cfg)):
        matchers = hooks_root.get(event, [])
        if not isinstance(matchers, list):
            continue
        for matcher_index, matcher_cfg in enumerate(matchers):
            if not isinstance(matcher_cfg, dict):
                continue
            matcher = str(matcher_cfg.get("matcher") or "*")
            if not _matcher_matches(matcher, payload):
                continue
            hooks = matcher_cfg.get("hooks", [])
            if not isinstance(hooks, list):
                continue
            for hook_index, hook in enumerate(hooks):
                if isinstance(hook, dict):
                    hook_id = f"{source}:{event}:{matcher_index}:{hook_index}"
                    result.append((hook_id, source, matcher, hook))
    return result


def _hook_layers(cfg: Any, *, include_untrusted: bool) -> list[tuple[str, dict]]:
    snapshot = getattr(cfg, "settings_snapshot", None)
    if snapshot is None:
        return []
    trusted = trusted_security_settings(snapshot)
    if nested_get(trusted, "disableAllHooks", False):
        return []
    managed_only = bool(nested_get(trusted, "allowManagedHooksOnly", False))
    layers = []
    for layer in snapshot.sources:
        if managed_only and layer.source != "policySettings":
            continue
        if layer.source == "projectSettings" and not include_untrusted:
            continue
        hooks = layer.settings.get("hooks")
        if isinstance(hooks, dict):
            layers.append((layer.source, hooks))
    return layers


def _hooks_trusted(cfg: Any) -> bool:
    if os.environ.get("TINY_HARNESS_TRUST_HOOKS", "").lower() in {"1", "true", "yes", "on"}:
        return True
    snapshot = getattr(cfg, "settings_snapshot", None)
    if snapshot is None:
        return False
    trusted = trusted_security_settings(snapshot)
    return bool(nested_get(trusted, "hooksTrusted", False))


def _matcher_matches(matcher: str, payload: dict[str, Any]) -> bool:
    matcher = matcher.strip()
    if matcher in {"", "*"}:
        return True
    tool_name = str(payload.get("tool_name") or "")
    tool_input = payload.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        tool_input = {}
    for part in matcher.split("|"):
        part = part.strip()
        if not part:
            continue
        if "(" in part:
            try:
                rule = permission_rule_value_from_string(part)
            except ValueError:
                continue
            if rule_matches_tool(rule, tool_name, tool_input):
                return True
            continue
        if part in {tool_name, display_tool_name(tool_name)}:
            return True
    return False


def _run_command_hook(hook_id: str, event: LifecycleEvent, source: str,
                      matcher: str, hook: dict, payload: dict[str, Any],
                      cfg: Any | None) -> tuple[LifecycleHookDiagnostic, dict | None]:
    command = str(hook.get("command") or "").strip()
    started = time.monotonic()
    if not command:
        return LifecycleHookDiagnostic(
            hook_id, event, "command", matcher, source, ok=False,
            reason="missing command"), None
    timeout = float(hook.get("timeout") or 10)
    workdir = Path(getattr(cfg, "workdir", Path.cwd())).resolve()
    input_payload = {
        "hook_event": event,
        "workdir": str(workdir),
        **payload,
    }
    try:
        proc = subprocess.run(
            command,
            input=json.dumps(input_payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            cwd=str(workdir),
            shell=True,
            timeout=timeout,
            env={**os.environ, "TINY_HARNESS_WORKDIR": str(workdir)},
        )
        parsed = _parse_hook_stdout(proc.stdout)
        blocked = proc.returncode == 2
        reason = None
        if blocked:
            reason = proc.stderr.strip() or "hook exited with code 2"
        diag = LifecycleHookDiagnostic(
            hook_id, event, "command", matcher, source,
            ok=proc.returncode == 0,
            blocked=blocked,
            reason=reason,
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        return diag, parsed
    except subprocess.TimeoutExpired as exc:
        return LifecycleHookDiagnostic(
            hook_id, event, "command", matcher, source,
            ok=False, blocked=True,
            reason=f"hook timed out after {timeout:g}s",
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            duration_ms=int((time.monotonic() - started) * 1000),
        ), None
    except Exception as exc:
        return LifecycleHookDiagnostic(
            hook_id, event, "command", matcher, source,
            ok=False, blocked=True,
            reason=f"{type(exc).__name__}: {exc}",
            duration_ms=int((time.monotonic() - started) * 1000),
        ), None


def _parse_hook_stdout(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        text = line.strip()
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        return data if isinstance(data, dict) else None
    return None
