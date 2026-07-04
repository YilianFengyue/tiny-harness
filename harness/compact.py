"""LLM-backed context compaction for CH07.

The raw transcript stays in trajectory files. This module rewrites only the
active in-memory message view by inserting a local compact boundary plus a
summary message, then preserving the most recent messages verbatim.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .config import Config
from .context import estimate_tokens
from .providers.base import ModelTurn, Provider

NO_TOOLS_PREAMBLE = """CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

- Do NOT use Read, Bash, Grep, Glob, Edit, Write, or ANY other tool.
- You already have all the context you need in the conversation below.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.
"""

COMPACT_PROMPT = """Your task is to create a detailed summary of the conversation so far.
This summary will replace older context so development work can continue without losing important state.

In <analysis>, inspect the conversation chronologically and identify:
- the user's explicit requests and feedback
- key decisions and constraints
- files, functions, commands, and test results that mattered
- errors encountered and how they were handled
- the exact current work and next step

Then write <summary> with these sections:
1. Primary Request and Intent
2. Key Technical Concepts
3. Files and Code Sections
4. Errors and Fixes
5. Problem Solving
6. User Messages and Feedback
7. Pending Tasks
8. Current Work
9. Next Step

Do not include the <analysis> content inside the summary.
"""

SUMMARY_PREFIX = (
    "This session is being continued from a previous conversation that was "
    "compacted to free context. Treat this summary as a guide, not truth; "
    "verify current files and code before acting.\n\n"
)


class CompactError(RuntimeError):
    pass


@dataclass
class CompactResult:
    trigger: str
    boundary_message: dict
    summary_message: dict
    messages_summarized: int
    messages_kept: int
    pre_tokens: int
    post_tokens: int
    summary: str
    usage: object | None = None
    request_id: str | None = None

    def as_event(self) -> dict:
        return {
            "trigger": self.trigger,
            "messages_summarized": self.messages_summarized,
            "messages_kept": self.messages_kept,
            "pre_tokens": self.pre_tokens,
            "post_tokens": self.post_tokens,
            "summary_chars": len(self.summary),
            "request_id": self.request_id,
        }


def compact_conversation(
    messages: list[dict],
    provider: Provider,
    cfg: Config,
    *,
    trigger: str,
    custom_instructions: str = "",
    keep_recent_messages: int = 8,
    max_transcript_chars: int = 80_000,
    on_retry: Callable | None = None,
) -> CompactResult:
    """Summarize older active messages and rewrite `messages` in place."""
    if len(messages) < 4:
        raise CompactError("not enough messages to compact")

    system_prefix = _system_prefix(messages)
    active_start = _last_boundary_index(messages) + 1
    if active_start <= 0:
        active_start = len(system_prefix)
    active = messages[active_start:]
    if len(active) <= keep_recent_messages + 1:
        raise CompactError("not enough active history to compact")

    keep_start = max(0, len(active) - keep_recent_messages)
    keep_start = _adjust_keep_start(active, keep_start)
    to_summarize = active[:keep_start]
    to_keep = active[keep_start:]
    if len(to_summarize) < 2:
        raise CompactError("not enough older history to compact")

    pre_tokens = estimate_tokens(messages)
    transcript = _render_transcript(to_summarize, max_chars=max_transcript_chars)
    prompt = _build_prompt(transcript, custom_instructions)
    resp = provider.complete(
        [{"role": "system", "content": NO_TOOLS_PREAMBLE},
         {"role": "user", "content": prompt}],
        [],
        on_retry=on_retry,
    )
    if resp.tool_calls:
        raise CompactError("compact model attempted a tool call")
    summary = format_compact_summary(resp.content or "")
    if not summary.strip():
        raise CompactError("compact model returned an empty summary")

    boundary = compact_boundary_message(
        trigger=trigger,
        pre_tokens=pre_tokens,
        messages_summarized=len(to_summarize),
        user_context=custom_instructions,
    )
    summary_message = compact_summary_message(summary, trigger=trigger)
    new_messages = system_prefix + [boundary, summary_message] + [dict(m) for m in to_keep]
    messages[:] = new_messages
    post_tokens = estimate_tokens(messages)
    return CompactResult(
        trigger=trigger,
        boundary_message=boundary,
        summary_message=summary_message,
        messages_summarized=len(to_summarize),
        messages_kept=len(to_keep),
        pre_tokens=pre_tokens,
        post_tokens=post_tokens,
        summary=summary,
        usage=getattr(resp, "usage", None),
        request_id=getattr(resp, "request_id", None),
    )


def compact_boundary_message(*, trigger: str, pre_tokens: int,
                             messages_summarized: int,
                             user_context: str = "") -> dict:
    return {
        "role": "system",
        "content": "[compact boundary: older conversation summarized locally]",
        "_kind": "compact_boundary",
        "_compact_metadata": {
            "trigger": trigger,
            "pre_tokens": pre_tokens,
            "messages_summarized": messages_summarized,
            "user_context": user_context,
        },
    }


def compact_summary_message(summary: str, *, trigger: str) -> dict:
    return {
        "role": "user",
        "content": SUMMARY_PREFIX + summary.strip(),
        "_kind": "compact_summary",
        "_compact_trigger": trigger,
    }


def is_compact_boundary(message: dict) -> bool:
    return message.get("_kind") == "compact_boundary"


def strip_compact_boundaries(messages: list[dict]) -> list[dict]:
    return [m for m in messages if not is_compact_boundary(m)]


def format_compact_summary(raw: str) -> str:
    text = re.sub(r"<analysis>[\s\S]*?</analysis>", "", raw, flags=re.IGNORECASE)
    match = re.search(r"<summary>([\s\S]*?)</summary>", text, flags=re.IGNORECASE)
    if match:
        text = match.group(1)
    text = re.sub(r"</?summary>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _build_prompt(transcript: str, custom_instructions: str) -> str:
    prompt = COMPACT_PROMPT
    if custom_instructions.strip():
        prompt += f"\n\nAdditional instructions from the user:\n{custom_instructions.strip()}\n"
    prompt += "\n\nConversation to summarize:\n\n" + transcript
    prompt += "\n\nREMINDER: Respond with <analysis>...</analysis> then <summary>...</summary>. Do not call tools."
    return prompt


def _render_transcript(messages: list[dict], *, max_chars: int) -> str:
    parts: list[str] = []
    for index, message in enumerate(messages, start=1):
        role = message.get("role", "unknown")
        content = message.get("content")
        if content is None and message.get("tool_calls"):
            content = "[assistant requested tool calls]"
        line = f"## Message {index} ({role})\n{_stringify_content(content)}"
        if message.get("tool_calls"):
            line += "\nTool calls: " + _stringify_content(message.get("tool_calls"))
        if message.get("tool_call_id"):
            line += f"\nTool call id: {message.get('tool_call_id')}"
        parts.append(line)
    transcript = "\n\n".join(parts)
    if len(transcript) <= max_chars:
        return transcript
    head = int(max_chars * 0.65)
    tail = max_chars - head - 80
    return (
        transcript[:head]
        + "\n\n...[middle of transcript omitted before compaction summary request]...\n\n"
        + transcript[-tail:]
    )


def _stringify_content(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return repr(value)


def _system_prefix(messages: list[dict]) -> list[dict]:
    prefix: list[dict] = []
    for message in messages:
        if message.get("role") == "system" and not is_compact_boundary(message):
            prefix.append(dict(message))
            continue
        break
    return prefix


def _last_boundary_index(messages: list[dict]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if is_compact_boundary(messages[index]):
            return index
    return -1


def _adjust_keep_start(active: list[dict], keep_start: int) -> int:
    while keep_start > 0 and active[keep_start].get("role") == "tool":
        keep_start -= 1
    return keep_start
