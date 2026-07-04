"""Context 管理：基于真实 usage 的预算跟踪 + 工具结果清理。

策略选择（DESIGN.md §Context 有完整权衡）：
- 采用【清除旧工具结果】而非摘要压缩：这是改动最小、无信息再加工失真的
  压缩方式（Anthropic context editing API 的默认策略同此，trigger=100K、
  keep=3 对）。摘要压缩作为第二层手段在文档中讨论，本实现刻意不做——
  它引入额外 LLM 调用、摘要失真风险，且会打破前缀缓存。
- 触发依据用 API 返回的真实 prompt_tokens（上一轮），而非本地估算：
  本地 tokenizer 对不上服务端（尤其经中转），真实账单数字才是 ground truth。
- 预算默认 240K：留出余量避开 gpt-5.5 的 272K 长上下文加价线（input x2）。
- 只改写 role=tool 消息的 content，不动消息结构——tool_call_id 链必须完整，
  否则违反 OpenAI 协议。
"""
from __future__ import annotations

from dataclasses import dataclass

MAX_RESERVED_OUTPUT_TOKENS = 20_000
AUTOCOMPACT_BUFFER_TOKENS = 13_000
WARNING_THRESHOLD_BUFFER_TOKENS = 20_000
ERROR_THRESHOLD_BUFFER_TOKENS = 20_000
MANUAL_COMPACT_BUFFER_TOKENS = 3_000

PLACEHOLDER = "[tool result cleared to save context: ~{est} tokens. Re-run the tool if needed.]"
MANUAL_PLACEHOLDER = "[tool result manually compacted: ~{est} tokens cleared. Re-run the tool if exact content is needed.]"


@dataclass(frozen=True)
class ContextBudgetState:
    raw_window_tokens: int
    reserved_output_tokens: int
    effective_window_tokens: int
    auto_compact_threshold: int
    warning_threshold: int
    error_threshold: int
    blocking_limit: int
    summary_compact_threshold: int
    usage_tokens: int
    usage_source: str
    last_prompt_tokens: int
    estimated_tokens: int
    percent_used: int
    percent_left: int
    level: str
    is_above_warning_threshold: bool
    is_above_error_threshold: bool
    is_above_auto_compact_threshold: bool
    is_at_blocking_limit: bool
    consecutive_auto_compact_failures: int = 0
    last_compact_kind: str | None = None

    def as_dict(self) -> dict:
        return {
            "raw_window_tokens": self.raw_window_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "effective_window_tokens": self.effective_window_tokens,
            "auto_compact_threshold": self.auto_compact_threshold,
            "summary_compact_threshold": self.summary_compact_threshold,
            "warning_threshold": self.warning_threshold,
            "error_threshold": self.error_threshold,
            "blocking_limit": self.blocking_limit,
            "usage_tokens": self.usage_tokens,
            "usage_source": self.usage_source,
            "last_prompt_tokens": self.last_prompt_tokens,
            "estimated_tokens": self.estimated_tokens,
            "percent_used": self.percent_used,
            "percent_left": self.percent_left,
            "level": self.level,
            "is_above_warning_threshold": self.is_above_warning_threshold,
            "is_above_error_threshold": self.is_above_error_threshold,
            "is_above_auto_compact_threshold": self.is_above_auto_compact_threshold,
            "is_at_blocking_limit": self.is_at_blocking_limit,
            "consecutive_auto_compact_failures": self.consecutive_auto_compact_failures,
            "last_compact_kind": self.last_compact_kind,
        }


@dataclass
class ContextManager:
    budget_tokens: int = 240_000
    keep_recent: int = 3          # 保留最近 N 条工具结果不清理
    hard_limit_tokens: int = 300_000
    tool_result_budget_chars: int = 16_000
    reserved_output_tokens: int = MAX_RESERVED_OUTPUT_TOKENS
    last_prompt_tokens: int = 0   # 上一轮真实 input 规模（由 loop 在每次响应后更新）
    consecutive_auto_compact_failures: int = 0
    last_compact_kind: str | None = None

    def observe(self, prompt_tokens: int) -> None:
        self.last_prompt_tokens = prompt_tokens

    @property
    def effective_window_tokens(self) -> int:
        reserved = min(max(int(self.reserved_output_tokens), 0), MAX_RESERVED_OUTPUT_TOKENS)
        return max(int(self.hard_limit_tokens) - reserved, 1)

    @property
    def auto_compact_threshold(self) -> int:
        return min(
            int(self.budget_tokens),
            max(self.effective_window_tokens - AUTOCOMPACT_BUFFER_TOKENS, 1),
        )

    @property
    def blocking_limit(self) -> int:
        return max(self.effective_window_tokens - MANUAL_COMPACT_BUFFER_TOKENS, 1)

    @property
    def summary_compact_threshold(self) -> int:
        return max(self.effective_window_tokens - AUTOCOMPACT_BUFFER_TOKENS, 1)

    def status(self, messages: list[dict] | None = None) -> ContextBudgetState:
        estimated = estimate_tokens(messages or [])
        usage = max(self.last_prompt_tokens, estimated)
        source = "api" if self.last_prompt_tokens >= estimated and self.last_prompt_tokens > 0 else "estimate"
        threshold = self.auto_compact_threshold
        warning_threshold = max(threshold - WARNING_THRESHOLD_BUFFER_TOKENS, 1)
        error_threshold = max(threshold - ERROR_THRESHOLD_BUFFER_TOKENS, 1)
        blocking_limit = self.blocking_limit
        percent_used = min(999, max(0, round((usage / self.effective_window_tokens) * 100)))
        percent_left = max(0, 100 - min(percent_used, 100))
        above_warning = usage >= warning_threshold
        above_error = usage >= error_threshold
        above_auto = usage >= threshold
        at_blocking = usage >= blocking_limit
        if at_blocking:
            level = "blocking"
        elif above_auto:
            level = "auto_compact"
        elif above_error:
            level = "error"
        elif above_warning:
            level = "warning"
        else:
            level = "safe"
        return ContextBudgetState(
            raw_window_tokens=int(self.hard_limit_tokens),
            reserved_output_tokens=min(max(int(self.reserved_output_tokens), 0),
                                       MAX_RESERVED_OUTPUT_TOKENS),
            effective_window_tokens=self.effective_window_tokens,
            auto_compact_threshold=threshold,
            summary_compact_threshold=self.summary_compact_threshold,
            warning_threshold=warning_threshold,
            error_threshold=error_threshold,
            blocking_limit=blocking_limit,
            usage_tokens=usage,
            usage_source=source,
            last_prompt_tokens=self.last_prompt_tokens,
            estimated_tokens=estimated,
            percent_used=percent_used,
            percent_left=percent_left,
            level=level,
            is_above_warning_threshold=above_warning,
            is_above_error_threshold=above_error,
            is_above_auto_compact_threshold=above_auto,
            is_at_blocking_limit=at_blocking,
            consecutive_auto_compact_failures=self.consecutive_auto_compact_failures,
            last_compact_kind=self.last_compact_kind,
        )

    def maybe_compact(self, messages: list[dict]) -> dict | None:
        """超预算时就地清理旧 tool 消息，返回清理统计（无动作返回 None）。"""
        if not self.status(messages).is_above_auto_compact_threshold:
            return None
        return self.clear_old_tool_results(messages, reason="microcompact")

    def should_summary_compact(self, messages: list[dict]) -> bool:
        return self.status(messages).usage_tokens >= self.summary_compact_threshold

    def budget_tool_results(self, messages: list[dict]) -> dict | None:
        """Before each request, shrink individual oversized tool results."""
        changed, freed_chars = 0, 0
        for m in messages:
            if m.get("role") != "tool" or m.get("_budgeted"):
                continue
            content = m.get("content") or ""
            if len(content) <= self.tool_result_budget_chars:
                continue
            head = content[: int(self.tool_result_budget_chars * 0.7)]
            tail = content[-int(self.tool_result_budget_chars * 0.15):]
            omitted = len(content) - len(head) - len(tail)
            m["content"] = (
                f"{head}\n... [tool result budgeted before model request: "
                f"{omitted} chars omitted; re-run the tool with a narrower "
                f"request if exact omitted content is needed] ...\n{tail}"
            )
            m["_budgeted"] = True
            changed += 1
            freed_chars += omitted
        if not changed:
            return None
        return {"kind": "tool_result_budget", "budgeted_messages": changed,
                "est_tokens_freed": freed_chars // 4}

    def hard_limit_exceeded(self, messages: list[dict]) -> dict | None:
        """Return stats when the next request is likely over the hard limit."""
        state = self.status(messages)
        if not state.is_at_blocking_limit:
            return None
        return {"prompt_tokens_estimate": state.usage_tokens,
                "hard_limit_tokens": state.blocking_limit,
                "effective_window_tokens": state.effective_window_tokens,
                "percent_used": state.percent_used}

    def reactive_compact(self, messages: list[dict]) -> dict | None:
        """Emergency compaction after a provider says the prompt is too long."""
        return self.clear_old_tool_results(messages, reason="reactive_compact",
                                           keep_recent=0, min_chars=1)

    def manual_compact(self, messages: list[dict], note: str = "") -> dict | None:
        """User-triggered deterministic compaction for old tool results."""
        edit = self.clear_old_tool_results(
            messages, reason="manual_compact", keep_recent=self.keep_recent,
            min_chars=1, placeholder=MANUAL_PLACEHOLDER)
        if edit is not None:
            edit["note"] = note
            edit["status"] = self.status(messages).as_dict()
        return edit

    def clear_old_tool_results(self, messages: list[dict], reason: str,
                               keep_recent: int | None = None,
                               min_chars: int = 200,
                               placeholder: str = PLACEHOLDER) -> dict | None:
        """Replace older tool contents with placeholders while preserving IDs."""
        keep = self.keep_recent if keep_recent is None else keep_recent

        tool_indices = [i for i, m in enumerate(messages)
                        if m.get("role") == "tool" and not m.get("_cleared")]
        clearable = tool_indices[:-keep] if keep else tool_indices
        if not clearable:
            return None

        cleared, freed_chars = 0, 0
        for i in clearable:
            content = messages[i].get("content") or ""
            if len(content) < min_chars:    # 太短的清了也省不出什么，还破坏可读性
                continue
            est = max(len(content) // 4, 1)   # chars/4 粗估，只用于占位符提示
            messages[i]["content"] = placeholder.format(est=est)
            messages[i]["_cleared"] = True    # 发送前由 loop 剥除的内部标记
            cleared += 1
            freed_chars += len(content)

        if cleared == 0:
            return None
        # 乐观下调水位，避免下一轮重复触发；真实值由下一次 API 响应矫正
        est_freed = freed_chars // 4
        self.last_prompt_tokens = max(self.last_prompt_tokens - est_freed, 0)
        self.last_compact_kind = reason
        return {"kind": reason, "cleared_messages": cleared,
                "est_tokens_freed": est_freed}

    def record_auto_compact_failure(self) -> int:
        self.consecutive_auto_compact_failures += 1
        return self.consecutive_auto_compact_failures

    def record_auto_compact_success(self) -> None:
        self.consecutive_auto_compact_failures = 0


def strip_internal_marks(messages: list[dict]) -> list[dict]:
    """发送给 API 前剥除内部标记字段（协议不认识多余字段）。"""
    return [
        {k: v for k, v in m.items() if not k.startswith("_")}
        for m in messages
        if m.get("_kind") != "compact_boundary"
    ]


def estimate_tokens(messages: list[dict]) -> int:
    """Coarse prompt estimate for local hard-limit checks."""
    total = 0
    for m in messages:
        total += 4
        for value in m.values():
            if isinstance(value, str):
                total += max(len(value) // 4, 1)
            elif isinstance(value, list):
                total += len(str(value)) // 4
    return total
