"""Human-readable context status for TUI surfaces."""
from __future__ import annotations

def context_status_line(status: dict[str, object]) -> str:
    level = str(status.get("level") or "safe")
    used = int(status.get("percent_used") or 0)
    left = int(status.get("percent_left") or 0)
    usage = _fmt_tokens(int(status.get("usage_tokens") or 0))
    if level == "blocking":
        label = "blocking"
    elif level == "auto_compact":
        label = "compact"
    elif level == "error":
        label = "low"
    elif level == "warning":
        label = "warn"
    else:
        label = "safe"
    return f"ctx:{label} {used}% used {left}% free ({usage})"


def context_pill(status: dict[str, object]) -> str:
    used = int(status.get("percent_used") or 0)
    level = str(status.get("level") or "safe")
    symbol = "◯"
    if level in {"warning", "error"}:
        symbol = "◐"
    elif level in {"auto_compact", "blocking"}:
        symbol = "●"
    return f"{symbol} context {used}%"


def format_context_summary(status: dict[str, object]) -> str:
    lines = [
        "## Context Usage",
        "",
        f"- level: {status.get('level', 'safe')}",
        f"- usage: {_fmt_tokens(int(status.get('usage_tokens') or 0))} "
        f"({status.get('percent_used', 0)}% used, {status.get('percent_left', 0)}% free)",
        f"- source: {status.get('usage_source', 'estimate')}",
        f"- effective window: {_fmt_tokens(int(status.get('effective_window_tokens') or 0))}",
        f"- reserved output: {_fmt_tokens(int(status.get('reserved_output_tokens') or 0))}",
        f"- auto compact threshold: {_fmt_tokens(int(status.get('auto_compact_threshold') or 0))}",
        f"- warning threshold: {_fmt_tokens(int(status.get('warning_threshold') or 0))}",
        f"- blocking limit: {_fmt_tokens(int(status.get('blocking_limit') or 0))}",
    ]
    last = status.get("last_compact_kind")
    if last:
        lines.append(f"- last compact: {last}")
    failures = int(status.get("consecutive_auto_compact_failures") or 0)
    if failures:
        lines.append(f"- auto compact failures: {failures}")
    return "\n".join(lines)


def format_compact_result(edit: dict | None, status: dict[str, object]) -> str:
    if not edit:
        return "No old tool results were compacted. Context is already lean enough."
    if edit.get("kind") == "manual_summary_compact":
        return (
            "[Context Summarized]\n"
            f"- trigger: {edit.get('trigger')}\n"
            f"- summarized messages: {edit.get('messages_summarized', 0)}\n"
            f"- kept recent messages: {edit.get('messages_kept', 0)}\n"
            f"- estimated tokens: {edit.get('pre_tokens', 0)} -> {edit.get('post_tokens', 0)}\n"
            f"- current: {context_status_line(status)}"
        )
    error = edit.get("summary_error")
    prefix = "[Context Compacted]"
    if error:
        prefix += f"\n- summary fallback: {error}"
    return (
        f"{prefix}\n"
        f"- kind: {edit.get('kind')}\n"
        f"- cleared messages: {edit.get('cleared_messages', 0)}\n"
        f"- estimated tokens freed: {edit.get('est_tokens_freed', 0)}\n"
        f"- current: {context_status_line(status)}"
    )


def _fmt_tokens(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)
