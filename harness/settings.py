"""Settings loading and merge rules for Tiny Harness.

The shape intentionally mirrors the useful parts of Claude Code's settings
system without importing its product-specific machinery:
- source order is plugin -> user -> project -> local -> flag -> policy
- arrays append and deduplicate, objects deep-merge, scalars override
- policy settings are selected by first non-empty source, then merged last
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping

SettingSource = Literal[
    "pluginSettings",
    "userSettings",
    "projectSettings",
    "localSettings",
    "flagSettings",
    "policySettings",
]

SETTING_SOURCES: tuple[SettingSource, ...] = (
    "pluginSettings",
    "userSettings",
    "projectSettings",
    "localSettings",
    "flagSettings",
    "policySettings",
)

FILE_SETTING_SOURCES: tuple[SettingSource, ...] = (
    "userSettings",
    "projectSettings",
    "localSettings",
    "flagSettings",
    "policySettings",
)

TRUSTED_SECURITY_SOURCES: tuple[SettingSource, ...] = (
    "userSettings",
    "localSettings",
    "flagSettings",
    "policySettings",
)


@dataclass(frozen=True)
class SettingsError:
    source: SettingSource | str
    path: str
    message: str


@dataclass(frozen=True)
class SettingsLayer:
    source: SettingSource
    settings: dict
    path: str | None = None
    origin: str | None = None


@dataclass(frozen=True)
class SettingsSnapshot:
    effective: dict
    sources: tuple[SettingsLayer, ...]
    errors: tuple[SettingsError, ...] = ()
    policy_origin: str | None = None


def load_settings(
    workdir: Path,
    *,
    flag_settings_path: Path | None = None,
    flag_settings: Mapping[str, object] | None = None,
    plugin_settings: Mapping[str, object] | None = None,
    enabled_sources: Iterable[SettingSource] | None = None,
) -> SettingsSnapshot:
    """Load and merge settings from all enabled sources."""
    workdir = Path(workdir).resolve()
    enabled = set(enabled_sources or SETTING_SOURCES)
    effective: dict = {}
    layers: list[SettingsLayer] = []
    errors: list[SettingsError] = []
    policy_origin: str | None = None

    def merge_layer(layer: SettingsLayer) -> None:
        nonlocal effective
        if not layer.settings:
            return
        effective = merge_settings(effective, layer.settings)
        layers.append(layer)

    if "pluginSettings" in enabled and plugin_settings:
        merge_layer(SettingsLayer(
            "pluginSettings", _plain_dict(plugin_settings), origin="plugin"))

    for source in ("userSettings", "projectSettings", "localSettings"):
        if source not in enabled:
            continue
        path = settings_path_for_source(source, workdir)
        data, source_errors = read_settings_file(path, source)
        errors.extend(source_errors)
        merge_layer(SettingsLayer(source, data, str(path)))

    if "flagSettings" in enabled:
        flag_layer: dict = {}
        flag_path_text: str | None = None
        if flag_settings_path:
            data, source_errors = read_settings_file(flag_settings_path, "flagSettings")
            errors.extend(source_errors)
            flag_layer = merge_settings(flag_layer, data)
            flag_path_text = str(flag_settings_path)
        if flag_settings:
            flag_layer = merge_settings(flag_layer, _plain_dict(flag_settings))
        merge_layer(SettingsLayer(
            "flagSettings", flag_layer, flag_path_text,
            "inline" if flag_settings else None))

    if "policySettings" in enabled:
        policy, origin, source_errors = load_policy_settings()
        errors.extend(source_errors)
        if policy:
            policy_origin = origin
            merge_layer(SettingsLayer(
                "policySettings", policy, _policy_origin_path(origin), origin))

    return SettingsSnapshot(
        effective=effective,
        sources=tuple(layers),
        errors=tuple(errors),
        policy_origin=policy_origin,
    )


def merge_settings(base: Mapping[str, object], incoming: Mapping[str, object]) -> dict:
    """Deep merge with array append/deduplicate semantics."""
    result = copy.deepcopy(dict(base))
    for key, value in incoming.items():
        if key in result:
            result[key] = _merge_value(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def read_settings_file(path: Path, source: SettingSource | str) -> tuple[dict, list[SettingsError]]:
    path = Path(path)
    if not path.exists():
        return {}, []
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError as e:
        return {}, [SettingsError(source, str(path), str(e))]
    if not raw.strip():
        return {}, []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return {}, [SettingsError(source, str(path), f"invalid JSON: {e}")]
    if not isinstance(data, dict):
        return {}, [SettingsError(source, str(path), "settings root must be an object")]
    return data, []


def settings_path_for_source(source: SettingSource, workdir: Path) -> Path:
    if source == "userSettings":
        return _user_settings_home() / "settings.json"
    if source == "projectSettings":
        return workdir / ".tiny-harness" / "settings.json"
    if source == "localSettings":
        return workdir / ".tiny-harness" / "settings.local.json"
    if source == "policySettings":
        return _managed_settings_dir() / "managed-settings.json"
    raise ValueError(f"{source} does not have a stable file path")


def parse_setting_sources_flag(text: str) -> tuple[SettingSource, ...]:
    if not text.strip():
        return ()
    aliases = {
        "plugin": "pluginSettings",
        "user": "userSettings",
        "project": "projectSettings",
        "local": "localSettings",
        "flag": "flagSettings",
        "policy": "policySettings",
    }
    result: list[SettingSource] = []
    for raw in text.split(","):
        name = raw.strip()
        source = aliases.get(name, name)
        if source not in SETTING_SOURCES:
            valid = ", ".join(aliases)
            raise ValueError(f"invalid setting source: {name}. valid: {valid}")
        result.append(source)  # type: ignore[arg-type]
    return tuple(result)


def settings_for_source(snapshot: SettingsSnapshot, source: SettingSource) -> dict:
    merged: dict = {}
    for layer in snapshot.sources:
        if layer.source == source:
            merged = merge_settings(merged, layer.settings)
    return merged


def trusted_security_settings(snapshot: SettingsSnapshot) -> dict:
    """Merge only sources trusted for security-sensitive switches."""
    trusted: dict = {}
    for layer in snapshot.sources:
        if layer.source in TRUSTED_SECURITY_SOURCES:
            trusted = merge_settings(trusted, layer.settings)
    return trusted


def nested_get(data: Mapping[str, object], path: str, default=None):
    cur: object = data
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def load_policy_settings() -> tuple[dict, str | None, list[SettingsError]]:
    """Return the first non-empty policy source plus any read errors."""
    errors: list[SettingsError] = []

    raw = os.environ.get("TINY_HARNESS_POLICY_SETTINGS_JSON", "").strip()
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and data:
                return data, "env", errors
            if not isinstance(data, dict):
                errors.append(SettingsError(
                    "policySettings", "TINY_HARNESS_POLICY_SETTINGS_JSON",
                    "settings root must be an object"))
        except json.JSONDecodeError as e:
            errors.append(SettingsError(
                "policySettings", "TINY_HARNESS_POLICY_SETTINGS_JSON",
                f"invalid JSON: {e}"))

    base = _managed_settings_dir() / "managed-settings.json"
    merged, found = {}, False
    data, base_errors = read_settings_file(base, "policySettings")
    errors.extend(base_errors)
    if data:
        merged = merge_settings(merged, data)
        found = True

    dropin = _managed_settings_dir() / "managed-settings.d"
    try:
        files = sorted(
            p for p in dropin.iterdir()
            if p.is_file() and p.suffix == ".json" and not p.name.startswith(".")
        )
    except OSError:
        files = []
    for path in files:
        data, file_errors = read_settings_file(path, "policySettings")
        errors.extend(file_errors)
        if data:
            merged = merge_settings(merged, data)
            found = True
    if found:
        return merged, "file", errors
    return {}, None, errors


def _merge_value(old, new):
    if isinstance(old, dict) and isinstance(new, Mapping):
        return merge_settings(old, new)
    if isinstance(old, list) and isinstance(new, list):
        result = copy.deepcopy(old)
        for item in new:
            if item not in result:
                result.append(copy.deepcopy(item))
        return result
    return copy.deepcopy(new)


def _plain_dict(value: Mapping[str, object]) -> dict:
    return copy.deepcopy(dict(value))


def _user_settings_home() -> Path:
    raw = os.environ.get("TINY_HARNESS_CONFIG_HOME")
    return Path(raw).expanduser() if raw else Path.home() / ".tiny-harness"


def _managed_settings_dir() -> Path:
    raw = os.environ.get("TINY_HARNESS_MANAGED_SETTINGS_PATH")
    if raw:
        return Path(raw).expanduser()
    if os.name == "nt":
        return Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "TinyHarness"
    return Path("/etc/tiny-harness")


def _policy_origin_path(origin: str | None) -> str | None:
    if origin == "env":
        return "TINY_HARNESS_POLICY_SETTINGS_JSON"
    if origin == "file":
        return str(_managed_settings_dir())
    return None
