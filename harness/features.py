"""Feature gates for startup-fixed and settings-driven behavior."""
from __future__ import annotations

import os
from typing import Any

from .settings import nested_get


def feature(name: str, cfg=None) -> bool:
    return bool(feature_value(name, False, cfg))


def feature_value(name: str, default: Any = None, cfg=None) -> Any:
    return feature_snapshot(cfg).get(name, default)


def feature_snapshot(cfg=None) -> dict[str, Any]:
    """Return the effective feature map for this process/session.

    Settings provide the project/session defaults. TINY_HARNESS_FEATURES is a
    startup override so command-line smoke tests can force features without
    editing persistent files. Syntax: "a,b,-c" where -c disables c.
    """
    features: dict[str, Any] = {}
    snapshot = getattr(cfg, "settings_snapshot", None)
    if snapshot is not None:
        raw = nested_get(snapshot.effective, "features", {})
        if isinstance(raw, dict):
            features.update(raw)
    features.update(_env_features())
    return features


def _env_features() -> dict[str, bool]:
    raw = os.environ.get("TINY_HARNESS_FEATURES", "").strip()
    if not raw:
        return {}
    out: dict[str, bool] = {}
    for item in raw.split(","):
        name = item.strip()
        if not name:
            continue
        enabled = True
        if name[0] in {"+", "-"}:
            enabled = name[0] == "+"
            name = name[1:].strip()
        if name:
            out[name] = enabled
    return out
