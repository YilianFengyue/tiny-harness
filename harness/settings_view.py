"""Human-readable settings and feature summaries for TUI surfaces."""
from __future__ import annotations

import json
from typing import Any

from .features import feature_snapshot
from .settings import nested_get, trusted_security_settings


def format_settings_summary(cfg, section: str = "") -> str:
    snapshot = cfg.settings_snapshot
    if snapshot is None:
        return "Settings snapshot: unavailable"
    section = section.strip().lower()
    if section == "sources":
        return _format_sources(snapshot)
    if section == "effective":
        return _format_effective(cfg)
    if section == "trust":
        return _format_trust(cfg)
    return "\n\n".join([
        _format_sources(snapshot),
        _format_effective(cfg),
        _format_trust(cfg),
    ])


def format_features(cfg) -> str:
    features = feature_snapshot(cfg)
    if not features:
        return "Features: (none enabled)"
    lines = ["Features:"]
    for name in sorted(features):
        lines.append(f"- {name}: {features[name]}")
    return "\n".join(lines)


def settings_status_line(cfg) -> str:
    snapshot = cfg.settings_snapshot
    if snapshot is None or not snapshot.sources:
        return "settings:none"
    names = "+".join(_short_source(layer.source) for layer in snapshot.sources)
    policy = snapshot.policy_origin or "none"
    return f"settings:{names} policy:{policy}"


def managed_permission_rules_only(cfg) -> bool:
    snapshot = cfg.settings_snapshot
    if snapshot is None:
        return False
    policy = next(
        (layer.settings for layer in snapshot.sources
         if layer.source == "policySettings"),
        {},
    )
    return bool(policy.get("allowManagedPermissionRulesOnly"))


def _format_sources(snapshot) -> str:
    lines = ["Settings sources:"]
    if not snapshot.sources:
        lines.append("- none")
    for layer in snapshot.sources:
        path = f" path={layer.path}" if layer.path else ""
        origin = f" origin={layer.origin}" if layer.origin else ""
        lines.append(f"- {layer.source}{origin}{path}")
    lines.append(f"policy_origin: {snapshot.policy_origin or 'none'}")
    if snapshot.errors:
        lines.append("errors:")
        for error in snapshot.errors:
            lines.append(f"- [{error.source}] {error.path}: {error.message}")
    return "\n".join(lines)


def _format_effective(cfg) -> str:
    snapshot = cfg.settings_snapshot
    effective = snapshot.effective if snapshot else {}
    permissions = effective.get("permissions", {})
    features = feature_snapshot(cfg)
    data: dict[str, Any] = {
        "model": cfg.model,
        "max_turns": cfg.max_turns,
        "max_cost_usd": cfg.max_cost_usd,
        "permission_mode": cfg.permission_mode,
        "skills": cfg.skills,
        "features": features,
    }
    if isinstance(permissions, dict):
        data["permissions"] = {
            "mode": permissions.get("mode", permissions.get("defaultMode")),
            "allow": permissions.get("allow", []),
            "ask": permissions.get("ask", []),
            "deny": permissions.get("deny", []),
        }
    return "Effective settings:\n" + json.dumps(
        data, ensure_ascii=False, indent=2, sort_keys=True)


def _format_trust(cfg) -> str:
    snapshot = cfg.settings_snapshot
    trusted = trusted_security_settings(snapshot) if snapshot else {}
    managed_only = managed_permission_rules_only(cfg)
    return "\n".join([
        "Settings trust:",
        "- security-sensitive switches trust user/local/flag/policy settings",
        "- project settings can add normal project permissions, but should not bypass safety prompts",
        f"- managed permission rules only: {managed_only}",
        f"- trusted skipDangerousModePermissionPrompt: "
        f"{nested_get(trusted, 'skipDangerousModePermissionPrompt', False)}",
    ])


def _short_source(source: str) -> str:
    return {
        "pluginSettings": "plugin",
        "userSettings": "user",
        "projectSettings": "project",
        "localSettings": "local",
        "flagSettings": "flag",
        "policySettings": "policy",
    }.get(source, source)
