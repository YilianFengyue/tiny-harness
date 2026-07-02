"""配置层：.env 加载、运行参数、价格表。

不引入 python-dotenv：.env 只需 15 行解析逻辑，省一个依赖也证明我们知道它在做什么。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

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
    skills: list[str] = field(default_factory=list)
    runs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "runs")

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        load_dotenv()
        cfg = cls(**overrides)
        if cfg.api_key is None:
            cfg.api_key = os.environ.get("OPENAI_API_KEY")
        if cfg.base_url is None:
            cfg.base_url = os.environ.get("OPENAI_BASE_URL") or None
        if "model" not in overrides:
            cfg.model = os.environ.get("TINY_HARNESS_MODEL", cfg.model)
        cfg.workdir = Path(cfg.workdir).resolve()
        return cfg
