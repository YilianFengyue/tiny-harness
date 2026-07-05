"""配置层：.env 加载、运行参数、价格表。

不引入 python-dotenv：.env 只需 15 行解析逻辑，省一个依赖也证明我们知道它在做什么。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .settings import (
    SettingsSnapshot,
    load_settings,
    parse_setting_sources_flag,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 价格表（美元 / 每百万 token）。来源: https://openai.com/api/pricing/ (2026-06 核实)
# cached = 命中自动前缀缓存的输入价（gpt-5.5 为 input 的 10%）。
# 未知模型按 0 计价并在 summary 里标记 pricing_unknown，绝不静默编造成本。
DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "gpt-5.5":      {"input": 5.00, "cached": 0.50, "output": 30.00},
    "gpt-5.5-pro":  {"input": 30.00, "cached": 3.00, "output": 180.00},
    "gpt-5.1":      {"input": 1.25, "cached": 0.125, "output": 10.00},
    "gpt-5.1-mini": {"input": 0.25, "cached": 0.025, "output": 2.00},
    "gpt-5.5-mini": {"input": 0.25, "cached": 0.025, "output": 2.00},
}

# gpt-5.5 经济阈值：单次请求 input 超过 272K token，整个请求按 input 2x / output 1.5x 计费。
# 这是 context 管理预算的硬上界依据（见 DESIGN.md §Context）。
LONG_CONTEXT_THRESHOLD = 272_000
LONG_CONTEXT_INPUT_MULT = 2.0
LONG_CONTEXT_OUTPUT_MULT = 1.5


def load_dotenv(path: Path | None = None) -> None:
    """极简 .env 解析：KEY=VALUE，# 开头为注释，不覆盖已有环境变量。"""
    path = path or PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def load_pricing() -> dict[str, dict[str, float]]:
    pricing = dict(DEFAULT_PRICING)
    raw = os.environ.get("TINY_HARNESS_PRICING", "").strip()
    if raw:
        pricing.update(json.loads(raw))
    return pricing


@dataclass
class Config:
    model: str = "gpt-5.5"
    base_url: str | None = None
    api_key: str | None = None
    workdir: Path = field(default_factory=lambda: Path.cwd() / "workspace")
    max_turns: int = 30
    max_cost_usd: float = 1.0
    # context 预算（token）：超过即触发工具结果清理。240K 给 272K 计费阈值留余量。
    context_budget: int = 240_000
    context_keep_recent: int = 3       # 清理时保留最近 N 条工具结果
    context_hard_limit: int = 300_000  # 本地估算或上轮真实 input 超过后阻断
    tool_result_budget_chars: int = 16_000
    reasoning_effort: str | None = None  # none/low/medium/high/xhigh；None=不传，用服务端默认
    max_completion_tokens: int | None = None  # None=服务端默认；推理模型需给推理留 >=25K
    tool_output_limit: int = 20_000    # 工具结果截断阈值（字符）
    bash_timeout: int = 60             # 秒
    max_retries: int = 5
    yolo: bool = False                 # True 时跳过危险命令确认
    permission_mode: str = "default"   # default/plan/acceptEdits/bypass/dontAsk
    coordinator_mode: bool = False     # CH10: main agent orchestrates workers only
    skills: list[str] = field(default_factory=list)
    runs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "runs")
    settings_path: Path | None = None
    setting_sources: tuple[str, ...] | None = None
    settings_snapshot: SettingsSnapshot | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        load_dotenv()
        explicit = {k: v for k, v in overrides.items() if v is not None}
        settings_path = explicit.pop("settings_path", None)
        setting_sources_raw = explicit.pop("setting_sources", None)

        seed_kwargs = {}
        for key in ("workdir", "runs_dir"):
            if key in explicit:
                seed_kwargs[key] = explicit[key]
        cfg = cls(**seed_kwargs)
        cfg.workdir = Path(cfg.workdir).resolve()

        if setting_sources_raw is None:
            setting_sources_raw = os.environ.get("TINY_HARNESS_SETTING_SOURCES")
        setting_sources = None
        if isinstance(setting_sources_raw, str) and setting_sources_raw.strip():
            setting_sources = parse_setting_sources_flag(setting_sources_raw)
        elif setting_sources_raw:
            setting_sources = tuple(setting_sources_raw)

        snapshot = load_settings(
            cfg.workdir,
            flag_settings_path=Path(settings_path).expanduser() if settings_path else None,
            enabled_sources=setting_sources,
        )
        _apply_settings(cfg, snapshot.effective)
        cfg.settings_snapshot = snapshot
        cfg.settings_path = Path(settings_path).expanduser() if settings_path else None
        cfg.setting_sources = tuple(setting_sources) if setting_sources else None

        _apply_env(cfg)

        for key, value in explicit.items():
            if key in {"workdir", "runs_dir"}:
                value = Path(value)
            setattr(cfg, key, value)

        cfg.workdir = Path(cfg.workdir).resolve()
        cfg.runs_dir = Path(cfg.runs_dir).resolve()
        return cfg


def _apply_settings(cfg: Config, settings: dict[str, Any]) -> None:
    mapping = {
        "model": "model",
        "base_url": "base_url",
        "api_key": "api_key",
        "workdir": "workdir",
        "runs_dir": "runs_dir",
        "max_turns": "max_turns",
        "max_cost_usd": "max_cost_usd",
        "context_budget": "context_budget",
        "context_keep_recent": "context_keep_recent",
        "context_hard_limit": "context_hard_limit",
        "tool_result_budget_chars": "tool_result_budget_chars",
        "reasoning_effort": "reasoning_effort",
        "max_completion_tokens": "max_completion_tokens",
        "tool_output_limit": "tool_output_limit",
        "bash_timeout": "bash_timeout",
        "max_retries": "max_retries",
        "yolo": "yolo",
        "permission_mode": "permission_mode",
        "coordinator_mode": "coordinator_mode",
        "skills": "skills",
    }
    aliases = {
        "max_cost": "max_cost_usd",
        "keep_recent": "context_keep_recent",
        "permissionMode": "permission_mode",
        "coordinatorMode": "coordinator_mode",
    }
    for key, attr in {**mapping, **aliases}.items():
        if key in settings:
            _set_config_value(cfg, attr, settings[key])

    permissions = settings.get("permissions")
    if isinstance(permissions, dict):
        mode = permissions.get("mode", permissions.get("defaultMode"))
        if mode is not None:
            cfg.permission_mode = str(mode)


def _apply_env(cfg: Config) -> None:
    cfg.api_key = os.environ.get("OPENAI_API_KEY", cfg.api_key)
    cfg.base_url = os.environ.get("OPENAI_BASE_URL") or cfg.base_url
    cfg.model = os.environ.get("TINY_HARNESS_MODEL", cfg.model)
    cfg.permission_mode = os.environ.get("TINY_HARNESS_PERMISSION_MODE",
                                         cfg.permission_mode)
    if os.environ.get("TINY_HARNESS_COORDINATOR_MODE"):
        cfg.coordinator_mode = _as_bool(os.environ["TINY_HARNESS_COORDINATOR_MODE"])
    if os.environ.get("TINY_HARNESS_MAX_TURNS"):
        cfg.max_turns = int(os.environ["TINY_HARNESS_MAX_TURNS"])
    if os.environ.get("TINY_HARNESS_MAX_COST"):
        cfg.max_cost_usd = float(os.environ["TINY_HARNESS_MAX_COST"])


def _set_config_value(cfg: Config, attr: str, value: Any) -> None:
    if value is None or not hasattr(cfg, attr):
        return
    current = getattr(cfg, attr)
    if isinstance(current, Path):
        setattr(cfg, attr, Path(str(value)).expanduser())
    elif isinstance(current, bool):
        setattr(cfg, attr, _as_bool(value))
    elif isinstance(current, int) and not isinstance(current, bool):
        setattr(cfg, attr, int(value))
    elif isinstance(current, float):
        setattr(cfg, attr, float(value))
    elif isinstance(current, list):
        setattr(cfg, attr, list(value) if isinstance(value, list) else [str(value)])
    else:
        setattr(cfg, attr, value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
