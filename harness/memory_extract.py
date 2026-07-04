"""Automatic durable memory extraction.

This is the deterministic first stage of the CH06 extractor. It watches the
completed conversation, extracts explicit durable memory signals, and writes
validated records through the local memory subsystem. The controller keeps the
same coordination shape as a future forked-agent extractor: throttling,
direct-write mutual exclusion, in-progress coalescing, and trailing extraction.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .memory import (
    MEMORY_TYPES,
    load_memory_records,
    memory_path_info,
    write_memory,
)
from .settings import nested_get

MemoryEventSink = Callable[[dict], None]


@dataclass(frozen=True)
class MemoryCandidate:
    type: str
    name: str
    description: str
    content: str


@dataclass
class MemoryExtractionState:
    enabled: bool = True
    extracting: bool = False
    pending: bool = False
    last_status: str = "idle"
    last_saved: list[str] = field(default_factory=list)
    last_error: str | None = None
    turns_since_last_extraction: int = 0
    last_processed_index: int = 0


class MemoryExtractionController:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.state = MemoryExtractionState(enabled=auto_extract_enabled(cfg))
        self._lock = threading.RLock()
        self._pending_messages: list[dict] | None = None

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.state.enabled = enabled
            self.state.last_status = "enabled" if enabled else "disabled"

    def status(self) -> dict:
        with self._lock:
            return {
                "enabled": self.state.enabled,
                "extracting": self.state.extracting,
                "pending": self.state.pending,
                "last_status": self.state.last_status,
                "last_saved": list(self.state.last_saved),
                "last_error": self.state.last_error,
                "turns_since_last_extraction": self.state.turns_since_last_extraction,
                "last_processed_index": self.state.last_processed_index,
            }

    def extract(self, messages: list[dict], *, emit: MemoryEventSink | None = None,
                force: bool = False) -> list[dict]:
        emitted: list[dict] = []

        def send(event: dict) -> None:
            emitted.append(event)
            if emit:
                emit(event)

        with self._lock:
            if self.state.extracting:
                self._pending_messages = [dict(message) for message in messages]
                self.state.pending = True
                self.state.last_status = "pending"
                send({"type": "memory_extract_trailing", "status": "pending"})
                return emitted
            self.state.extracting = True
            self.state.pending = False
            self.state.last_error = None
            self.state.last_status = "extracting"

        try:
            self._run(messages, send, force=force, trailing=False)
        except Exception as exc:
            with self._lock:
                self.state.last_error = f"{type(exc).__name__}: {exc}"
                self.state.last_status = "error"
            send({"type": "memory_extract_error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            trailing_messages: list[dict] | None
            with self._lock:
                self.state.extracting = False
                trailing_messages = self._pending_messages
                self._pending_messages = None
                self.state.pending = False
            if trailing_messages is not None:
                send({"type": "memory_extract_trailing", "status": "running"})
                with self._lock:
                    self.state.extracting = True
                    self.state.last_status = "extracting"
                try:
                    self._run(trailing_messages, send, force=True, trailing=True)
                except Exception as exc:
                    with self._lock:
                        self.state.last_error = f"{type(exc).__name__}: {exc}"
                        self.state.last_status = "error"
                    send({"type": "memory_extract_error", "error": f"{type(exc).__name__}: {exc}"})
                finally:
                    with self._lock:
                        self.state.extracting = False
        return emitted

    def _run(self, messages: list[dict], emit: MemoryEventSink, *,
             force: bool, trailing: bool) -> None:
        info = memory_path_info(self.cfg)
        if not info.enabled:
            self._skip(emit, "memory_disabled", len(messages), emit_event=force)
            return
        if not self.state.enabled:
            self._skip(emit, "auto_extract_disabled", len(messages), emit_event=force)
            return

        start = self.state.last_processed_index
        recent = messages[start:]
        visible_count = sum(1 for message in recent if message.get("role") in {"user", "assistant"})
        if visible_count <= 0:
            self._skip(emit, "no_new_messages", len(messages), emit_event=force)
            return

        if has_direct_memory_write(recent):
            self._skip(emit, "direct_memory_write", len(messages), emit_event=True)
            return

        every = max(1, int(memory_setting(self.cfg, "extractEveryTurns", 1)))
        if not force and not trailing:
            self.state.turns_since_last_extraction += 1
            if self.state.turns_since_last_extraction < every:
                self._skip(emit, "throttled", start, emit_event=False)
                return
        self.state.turns_since_last_extraction = 0

        max_messages = max(1, int(memory_setting(self.cfg, "extractMaxMessages", 20)))
        candidates = extract_memory_candidates(recent[-max_messages:])
        if not candidates:
            self._skip(emit, "no_candidates", len(messages), emit_event=force)
            return
        emit({
            "type": "memory_extract_start",
            "message_count": visible_count,
            "force": force,
            "trailing": trailing,
        })
        saved = []
        skipped_duplicates = 0
        existing = load_memory_records(info.directory)
        seen = {_dedupe_key(record.type or "", record.description, record.content)
                for record in existing}
        for candidate in candidates:
            key = _dedupe_key(candidate.type, candidate.description, candidate.content)
            if key in seen:
                skipped_duplicates += 1
                continue
            record = write_memory(
                info.directory,
                candidate.type,
                candidate.name,
                candidate.description,
                candidate.content,
            )
            seen.add(key)
            saved.append(record.id)

        with self._lock:
            self.state.last_processed_index = len(messages)
            self.state.last_saved = saved
            self.state.last_status = "saved" if saved else "no_candidates"
        if saved:
            emit({
                "type": "memory_extract_saved",
                "count": len(saved),
                "paths": saved,
                "skipped_duplicates": skipped_duplicates,
            })
        else:
            if force:
                emit({
                    "type": "memory_extract_skipped",
                    "reason": "duplicates",
                    "skipped_duplicates": skipped_duplicates,
                })

    def _skip(self, emit: MemoryEventSink, reason: str, next_index: int,
              *, emit_event: bool) -> None:
        with self._lock:
            self.state.last_processed_index = next_index
            self.state.last_status = f"skipped:{reason}"
            self.state.last_saved = []
        if emit_event:
            emit({"type": "memory_extract_skipped", "reason": reason})


def auto_extract_enabled(cfg) -> bool:
    effective = getattr(getattr(cfg, "settings_snapshot", None), "effective", {}) or {}
    default = _as_bool(nested_get(effective, "autoMemoryEnabled", True))
    return _as_bool(nested_get(effective, "memory.autoExtract", default))


def memory_setting(cfg, name: str, default):
    effective = getattr(getattr(cfg, "settings_snapshot", None), "effective", {}) or {}
    return nested_get(effective, f"memory.{name}", default)


def has_direct_memory_write(messages: Iterable[dict]) -> bool:
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            fn = call.get("function", {}) if isinstance(call, dict) else {}
            if fn.get("name") in {"write_memory", "forget_memory"}:
                return True
    return False


def extract_memory_candidates(messages: Iterable[dict]) -> list[MemoryCandidate]:
    candidates: list[MemoryCandidate] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        for line in _candidate_lines(content):
            candidate = _candidate_from_line(line)
            if candidate:
                candidates.append(candidate)
    return candidates


def _candidate_lines(text: str) -> list[str]:
    lines = []
    for raw in re.split(r"[\n。！？!?]+", text):
        line = " ".join(raw.split()).strip(" -:：")
        if len(line) < 8:
            continue
        if line.lower().startswith(("原因", "why:", "why ")):
            continue
        if _looks_durable(line):
            lines.append(line)
    return lines


def _looks_durable(line: str) -> bool:
    lower = line.lower()
    markers = (
        "记住", "请记", "以后", "下次", "偏好", "不要", "别再", "必须",
        "统一", "约定", "规则", "原因是", "文档在", "链接", "地址",
        "remember", "from now on", "next time", "prefer", "always", "never",
        "must", "rule", "convention", "docs are", "dashboard", "http://",
        "https://",
    )
    return any(marker in lower for marker in markers)


def _candidate_from_line(line: str) -> MemoryCandidate | None:
    mem_type = _classify(line)
    if mem_type not in MEMORY_TYPES:
        return None
    cleaned = _strip_memory_request(line)
    name = _title(cleaned)
    description = _description(cleaned)
    content = _content(mem_type, cleaned)
    return MemoryCandidate(mem_type, name, description, content)


def _classify(line: str) -> str:
    lower = line.lower()
    if "http://" in lower or "https://" in lower or any(
        word in lower for word in ("grafana", "slack", "confluence", "linear", "文档在", "链接", "地址")
    ):
        return "reference"
    if any(word in lower for word in ("我是", "我熟悉", "我的角色", "my role", "i am ", "i'm ")):
        return "user"
    if any(word in lower for word in ("这个项目", "本项目", "团队", "架构", "迁移", "约定", "统一", "project", "convention")):
        return "project"
    return "feedback"


def _strip_memory_request(line: str) -> str:
    patterns = (
        r"^请?记住[:：,，\s]*",
        r"^以后[:：,，\s]*",
        r"^下次[:：,，\s]*",
        r"^remember[:：,，\s]*",
        r"^from now on[:：,，\s]*",
    )
    result = line.strip()
    for pattern in patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE).strip()
    return result or line.strip()


def _title(text: str) -> str:
    compact = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text, flags=re.UNICODE).strip()
    words = compact.split()
    if len(words) <= 8:
        return compact[:60] or "memory"
    return " ".join(words[:8])[:60]


def _description(text: str) -> str:
    return text[:180].rstrip()


def _content(mem_type: str, text: str) -> str:
    if mem_type in {"feedback", "project"}:
        return f"Rule/fact: {text}\nWhy: Captured from an explicit user instruction.\nHow to apply: Treat this as guidance, then verify current project state before acting."
    return text


def _dedupe_key(mem_type: str, description: str, content: str) -> str:
    return " ".join(f"{mem_type} {description} {content}".casefold().split())


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
