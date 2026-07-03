"""Permission rules and decisions for tool calls.

This module keeps the Ch04 permission core independent from the UI. The loop
can ask it for a deterministic decision today, while a later TUI layer can plug
in interactive confirmation and persistence without rewriting matching logic.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Literal, Mapping

PermissionBehavior = Literal["allow", "deny", "ask", "passthrough"]
PermissionMode = Literal["default", "plan", "acceptEdits", "bypass", "dontAsk"]
PermissionSource = Literal["project", "local", "session", "cli", "policy"]
PermissionUpdateKind = Literal["addRules", "replaceRules", "removeRules", "setMode"]
PERMISSION_MODES: tuple[PermissionMode, ...] = (
    "default", "plan", "acceptEdits", "bypass", "dontAsk")

CANONICAL_TOOL_NAMES = {
    "bash": "bash",
    "Bash": "bash",
    "Read": "read_file",
    "read": "read_file",
    "read_file": "read_file",
    "Write": "write_file",
    "write": "write_file",
    "write_file": "write_file",
    "Edit": "edit_file",
    "edit": "edit_file",
    "edit_file": "edit_file",
    "Glob": "glob_files",
    "glob": "glob_files",
    "glob_files": "glob_files",
    "Grep": "grep",
    "grep": "grep",
    "list_files": "list_files",
    "file_info": "file_info",
    "show_diff": "show_diff",
    "calculator": "calculator",
}

SENSITIVE_PATH_PATTERNS = (
    ".git",
    ".git/**",
    ".env",
    ".env.*",
    ".tiny-harness/settings.json",
    ".tiny-harness/settings.local.json",
    ".github/workflows/**",
    "*.pem",
    "*.key",
)


@dataclass(frozen=True)
class PermissionRuleValue:
    tool_name: str
    rule_content: str | None = None

    @property
    def canonical_tool_name(self) -> str:
        return normalize_tool_name(self.tool_name)


@dataclass(frozen=True)
class PermissionRule:
    source: PermissionSource
    behavior: PermissionBehavior
    value: PermissionRuleValue


@dataclass(frozen=True)
class PermissionDecision:
    behavior: PermissionBehavior
    message: str = ""
    reason_type: str = "none"
    rule: str | None = None
    source: PermissionSource | None = None
    mode: PermissionMode | None = None
    safety_check: bool = False
    suggestions: tuple[str, ...] = ()

    @property
    def allowed(self) -> bool:
        return self.behavior == "allow"


@dataclass(frozen=True)
class PermissionUpdate:
    kind: PermissionUpdateKind
    destination: PermissionSource = "session"
    behavior: PermissionBehavior | None = None
    rules: tuple[PermissionRuleValue, ...] = ()
    mode: PermissionMode | None = None


@dataclass(frozen=True)
class PermissionContext:
    mode: PermissionMode = "default"
    rules: tuple[PermissionRule, ...] = ()


class ResolveOnce:
    """First-wins resolver for racing permission answers.

    The TUI, a script hook, and an AI policy check may all try to answer the
    same prompt. claim() returns True only for the first claimant.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._claimed_by: str | None = None
        self._value: object | None = None

    def claim(self, owner: str, value: object | None = None) -> bool:
        with self._lock:
            if self._claimed_by is not None:
                return False
            self._claimed_by = owner
            self._value = value
            return True

    @property
    def claimed_by(self) -> str | None:
        return self._claimed_by

    @property
    def value(self) -> object | None:
        return self._value


def normalize_tool_name(name: str) -> str:
    return CANONICAL_TOOL_NAMES.get(name, name)


def display_tool_name(name: str) -> str:
    canonical = normalize_tool_name(name)
    return {
        "bash": "Bash",
        "read_file": "Read",
        "write_file": "Write",
        "edit_file": "Edit",
        "glob_files": "Glob",
        "grep": "Grep",
    }.get(canonical, canonical)


def permission_rule_value_from_string(text: str) -> PermissionRuleValue:
    text = text.strip()
    if not text:
        raise ValueError("permission rule cannot be empty")
    if "(" not in text:
        return PermissionRuleValue(text, None)
    if not text.endswith(")"):
        raise ValueError(f"invalid permission rule: {text}")
    tool, _, rest = text.partition("(")
    if not tool:
        raise ValueError(f"invalid permission rule: {text}")
    return PermissionRuleValue(tool, unescape_rule_content(rest[:-1]))


def permission_rule_value_to_string(value: PermissionRuleValue) -> str:
    if value.rule_content is None:
        return value.tool_name
    return f"{value.tool_name}({escape_rule_content(value.rule_content)})"


def suggest_permission_rule_value(tool_name: str,
                                  arguments: Mapping[str, object]) -> PermissionRuleValue:
    display = display_tool_name(tool_name)
    canonical = normalize_tool_name(tool_name)
    if canonical == "bash":
        command = str(arguments.get("command") or "").strip()
        return PermissionRuleValue(display, command or None)
    if canonical in {"read_file", "write_file", "edit_file", "glob_files", "grep"}:
        path = arguments.get("path") or arguments.get("pattern")
        return PermissionRuleValue(display, str(path)) if path else PermissionRuleValue(display)
    return PermissionRuleValue(display)


def format_permission_context(context: PermissionContext) -> str:
    lines = [f"mode: {context.mode}"]
    if not context.rules:
        lines.append("(no permission rules)")
        return "\n".join(lines)
    for behavior in ("deny", "ask", "allow"):
        items = [rule for rule in context.rules if rule.behavior == behavior]
        if not items:
            continue
        lines.append(f"{behavior}:")
        for rule in items:
            value = permission_rule_value_to_string(rule.value)
            lines.append(f"  - [{rule.source}] {value}")
    return "\n".join(lines)


def summarize_permission_update(update: PermissionUpdate) -> str:
    if update.kind == "setMode":
        return f"set {update.destination} mode to {update.mode}"
    values = ", ".join(permission_rule_value_to_string(v) for v in update.rules) or "(none)"
    return f"{update.kind} {update.destination} {update.behavior}: {values}"


def escape_rule_content(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def unescape_rule_content(text: str) -> str:
    out: list[str] = []
    escaped = False
    for ch in text:
        if escaped:
            out.append(ch)
            escaped = False
        elif ch == "\\":
            escaped = True
        else:
            out.append(ch)
    if escaped:
        out.append("\\")
    return "".join(out)


def rule_matches_tool(value: PermissionRuleValue, tool_name: str,
                      arguments: Mapping[str, object]) -> bool:
    if value.canonical_tool_name != normalize_tool_name(tool_name):
        return False
    if value.rule_content is None:
        return True
    if normalize_tool_name(tool_name) == "bash":
        command = str(arguments.get("command") or "")
        return match_bash_rule(value.rule_content, command)
    return _match_non_bash_content(value.rule_content, arguments)


def match_bash_rule(pattern: str, command: str) -> bool:
    pattern = _normalize_space(pattern)
    command = _normalize_space(command)
    if not pattern:
        return not command
    if pattern.endswith(":*") and "*" not in pattern[:-2]:
        base = pattern[:-2].strip()
        return command == base or command.startswith(base + " ")
    if _has_single_unescaped_trailing_space_star(pattern):
        base = pattern[:-2].strip()
        return command == base or command.startswith(base + " ")
    if _has_unescaped_star(pattern):
        return re.fullmatch(_wildcard_regex(pattern), command) is not None
    return command == pattern


def find_matching_rule(context: PermissionContext, behavior: PermissionBehavior,
                       tool_name: str, arguments: Mapping[str, object]) -> PermissionRule | None:
    return next(
        (rule for rule in context.rules
         if rule.behavior == behavior and rule_matches_tool(rule.value, tool_name, arguments)),
        None,
    )


def load_permission_context(workdir: Path, mode_override: str | None = None,
                            cli_allow: Iterable[str] = (),
                            cli_deny: Iterable[str] = (),
                            cli_ask: Iterable[str] = ()) -> PermissionContext:
    rules: list[PermissionRule] = []
    mode: PermissionMode = "default"
    for source, path in (
        ("project", workdir / ".tiny-harness" / "settings.json"),
        ("local", workdir / ".tiny-harness" / "settings.local.json"),
    ):
        data = _read_settings(path)
        permissions = data.get("permissions", {}) if isinstance(data, dict) else {}
        if isinstance(permissions, dict):
            loaded_mode = permissions.get("mode")
            if loaded_mode in PERMISSION_MODES:
                mode = loaded_mode
            rules.extend(_rules_from_settings(source, permissions))
    rules.extend(_cli_rules("cli", "allow", cli_allow))
    rules.extend(_cli_rules("cli", "deny", cli_deny))
    rules.extend(_cli_rules("cli", "ask", cli_ask))
    if mode_override:
        mode = _coerce_mode(mode_override)
    return PermissionContext(mode=mode, rules=tuple(rules))


def apply_permission_update(context: PermissionContext,
                            update: PermissionUpdate) -> PermissionContext:
    if update.kind == "setMode":
        if update.mode is None:
            raise ValueError("setMode update requires mode")
        return replace(context, mode=update.mode)
    if update.behavior is None:
        raise ValueError(f"{update.kind} update requires behavior")
    incoming = tuple(
        PermissionRule(update.destination, update.behavior, value)
        for value in update.rules
    )
    if update.kind == "addRules":
        existing = set(_rule_key(rule) for rule in context.rules)
        appended = tuple(rule for rule in incoming if _rule_key(rule) not in existing)
        return replace(context, rules=context.rules + appended)
    if update.kind == "replaceRules":
        kept = tuple(rule for rule in context.rules
                     if not (rule.source == update.destination
                             and rule.behavior == update.behavior))
        return replace(context, rules=kept + incoming)
    if update.kind == "removeRules":
        remove = set(_rule_key(rule) for rule in incoming)
        kept = tuple(rule for rule in context.rules if _rule_key(rule) not in remove)
        return replace(context, rules=kept)
    raise ValueError(f"unknown permission update kind: {update.kind}")


def apply_permission_updates(context: PermissionContext,
                             updates: Iterable[PermissionUpdate]) -> PermissionContext:
    for update in updates:
        context = apply_permission_update(context, update)
    return context


def persist_permission_updates(workdir: Path, updates: Iterable[PermissionUpdate]) -> None:
    grouped: dict[PermissionSource, list[PermissionUpdate]] = {"project": [], "local": []}
    for update in updates:
        if update.destination in grouped:
            grouped[update.destination].append(update)
    for destination, items in grouped.items():
        if not items:
            continue
        path = _settings_path(workdir, destination)
        data = _read_settings(path)
        permissions = data.setdefault("permissions", {})
        if not isinstance(permissions, dict):
            permissions = {}
            data["permissions"] = permissions
        context = _context_from_permissions(destination, permissions)
        context = apply_permission_updates(context, items)
        permissions["mode"] = context.mode
        for behavior in ("allow", "deny", "ask"):
            permissions[behavior] = [
                permission_rule_value_to_string(rule.value)
                for rule in context.rules
                if rule.source == destination and rule.behavior == behavior
            ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")


def is_sensitive_path(path: str | None) -> bool:
    if not path:
        return False
    normalized = str(Path(path).as_posix())
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return any(_glob_match(pattern, normalized) for pattern in SENSITIVE_PATH_PATTERNS)


def _match_non_bash_content(content: str, arguments: Mapping[str, object]) -> bool:
    path = arguments.get("path")
    if path is None:
        return False
    return _glob_match(content, str(path))


def _glob_match(pattern: str, text: str) -> bool:
    regex = re.escape(pattern).replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
    return re.fullmatch(regex, text.replace("\\", "/")) is not None


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _has_single_unescaped_trailing_space_star(pattern: str) -> bool:
    return pattern.endswith(" *") and _unescaped_star_count(pattern) == 1


def _has_unescaped_star(pattern: str) -> bool:
    return _unescaped_star_count(pattern) > 0


def _unescaped_star_count(pattern: str) -> int:
    count = 0
    escaped = False
    for ch in pattern:
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == "*":
            count += 1
    return count


def _wildcard_regex(pattern: str) -> str:
    out = ["^"]
    escaped = False
    for ch in pattern:
        if escaped:
            out.append(re.escape(ch))
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == "*":
            out.append(".*")
        else:
            out.append(re.escape(ch))
    if escaped:
        out.append(re.escape("\\"))
    out.append("$")
    return "".join(out)


def _read_settings(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _rules_from_settings(source: PermissionSource, permissions: Mapping[str, object]) -> list[PermissionRule]:
    rules: list[PermissionRule] = []
    for behavior in ("allow", "deny", "ask"):
        raw_rules = permissions.get(behavior, [])
        if not isinstance(raw_rules, list):
            continue
        for item in raw_rules:
            if isinstance(item, str):
                rules.append(PermissionRule(
                    source, behavior, permission_rule_value_from_string(item)))
    return rules


def _cli_rules(source: PermissionSource, behavior: PermissionBehavior,
               items: Iterable[str]) -> list[PermissionRule]:
    return [
        PermissionRule(source, behavior, permission_rule_value_from_string(item))
        for item in items
    ]


def _context_from_permissions(source: PermissionSource,
                              permissions: Mapping[str, object]) -> PermissionContext:
    mode = permissions.get("mode")
    return PermissionContext(
        mode=mode if mode in PERMISSION_MODES else "default",
        rules=tuple(_rules_from_settings(source, permissions)),
    )


def _settings_path(workdir: Path, destination: PermissionSource) -> Path:
    name = "settings.local.json" if destination == "local" else "settings.json"
    return workdir / ".tiny-harness" / name


def _coerce_mode(raw: str) -> PermissionMode:
    if raw not in PERMISSION_MODES:
        raise ValueError(f"unknown permission mode: {raw}")
    return raw  # type: ignore[return-value]


def _rule_key(rule: PermissionRule) -> tuple[str, str, str, str | None]:
    return (
        rule.source,
        rule.behavior,
        rule.value.canonical_tool_name,
        rule.value.rule_content,
    )
