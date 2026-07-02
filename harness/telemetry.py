"""可观测性层：JSONL trajectory + run_id + 成本台账。

设计依据（DESIGN.md §Observability）：
- JSONL 一行一个自包含事件：append-only、可流式 tail、崩溃只丢最后一行。
- 每条事件带 run_id + 单调递增 step + ISO 8601 UTC 时间戳。
- llm_request 记录完整 messages，llm_response 记录完整产出与 usage——
  这是"记录而非控制"的可复现策略：LLM 推理本质不确定（服务端动态 batching），
  完整请求/响应是唯一可靠的重放凭据。

事件类型（viewer/index.html 与 tests/ 共同依赖的契约，改动需同步三处）：
  run_start     {task, model, workdir, config, sdk_version, skills}
  turn_start    {turn, transition, n_messages}
  llm_request   {turn, model, n_messages, messages, tools, params}
  stream_request_start {turn, model}
  assistant_delta {turn, content?, reasoning_content?}
  retry         {turn, attempt, status, error, sleep_s}
  llm_response  {turn, finish_reason, content, tool_calls, usage, cost_usd,
                 request_id, latency_ms}
  tool_call     {turn, tool_call_id, name, arguments}
  tool_start    {turn, tool_call_id, name, arguments}
  tool_progress {turn, tool_call_id, name, ...progress}
  tool_result   {turn, tool_call_id, name, ok, result, duration_ms, truncated}
  tool_end      {turn, tool_call_id, name, ok, duration_ms, truncated}
  transition    {turn, kind, reason, ...details}
  context_edit  {turn, cleared_messages, est_tokens_freed, prompt_tokens_before}
  error         {where, error}
  run_end       {reason, turns, usage_total, cost_usd, pricing_unknown,
                 duration_s, final_message}
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import (
    LONG_CONTEXT_INPUT_MULT,
    LONG_CONTEXT_OUTPUT_MULT,
    LONG_CONTEXT_THRESHOLD,
)


def new_run_id() -> str:
    """时间戳 + 短 uuid：可读、可排序、唯一。"""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


@dataclass
class Usage:
    """单次/累计 token 用量。OpenAI 语义：

    - prompt_tokens 已包含 cached_tokens（cached 是其子集，计价打 1 折）
    - completion_tokens 已包含 reasoning_tokens（reasoning 不可见但按 output 计费、占上下文）
    """
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.cached_tokens += other.cached_tokens
        self.completion_tokens += other.completion_tokens
        self.reasoning_tokens += other.reasoning_tokens

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


@dataclass
class CostLedger:
    """成本台账。公式（gpt-5.5，5m/24h 自动缓存）：

      cost = (prompt - cached) * P_in + cached * P_cached + completion * P_out

    单次请求 prompt > 272K 时触发长上下文加价（input x2 / output x1.5），
    这也是 context 预算默认 240K 的经济学依据。
    """
    pricing: dict[str, dict[str, float]]
    total: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    pricing_unknown: bool = False
    long_context_hits: int = 0

    def record(self, model: str, usage: Usage) -> float:
        """记账一次请求，返回该次成本（美元）。"""
        self.total.add(usage)
        price = self.pricing.get(model)
        if price is None:
            self.pricing_unknown = True
            return 0.0
        in_mult, out_mult = 1.0, 1.0
        if usage.prompt_tokens > LONG_CONTEXT_THRESHOLD:
            in_mult, out_mult = LONG_CONTEXT_INPUT_MULT, LONG_CONTEXT_OUTPUT_MULT
            self.long_context_hits += 1
        cost = (
            (usage.prompt_tokens - usage.cached_tokens) * price["input"] * in_mult
            + usage.cached_tokens * price["cached"] * in_mult
            + usage.completion_tokens * price["output"] * out_mult
        ) / 1_000_000
        self.cost_usd += cost
        return cost


class RunLogger:
    """每次运行一个目录：runs/<run_id>/trajectory.jsonl + summary.json。"""

    def __init__(self, runs_dir: Path, run_id: str | None = None):
        self.run_id = run_id or new_run_id()
        self.dir = Path(runs_dir) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "trajectory.jsonl"
        self._fh = open(self.path, "a", encoding="utf-8")
        self._step = 0
        self._t0 = time.monotonic()

    def emit(self, type_: str, **payload) -> None:
        event = {
            "run_id": self.run_id,
            "step": self._step,
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "type": type_,
            **payload,
        }
        self._fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()  # 崩溃时最多丢正在写的一行
        self._step += 1

    def finish(self, reason: str, turns: int, ledger: CostLedger,
               final_message: str | None = None) -> dict:
        summary = {
            "run_id": self.run_id,
            "reason": reason,
            "turns": turns,
            "usage_total": ledger.total.as_dict(),
            "cost_usd": round(ledger.cost_usd, 6),
            "pricing_unknown": ledger.pricing_unknown,
            "long_context_hits": ledger.long_context_hits,
            "duration_s": round(time.monotonic() - self._t0, 2),
            "final_message": final_message,
        }
        self.emit("run_end", **summary)
        (self.dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self._fh.close()
        return summary


def read_trajectory(runs_dir: Path, run_id: str) -> list[dict]:
    """读取一次运行的全部事件（resume / replay / 测试断言共用）。"""
    path = Path(runs_dir) / run_id / "trajectory.jsonl"
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events
