"""Persistent memory storage for Tiny Harness.

The CH06 memory model is intentionally small: four closed memory types,
one markdown file per memory, and a concise MEMORY.md index loaded into the
agent prompt. Memory is treated as a clue, never as current truth.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .settings import (
    SettingsSnapshot,
    SettingSource,
    nested_get,
    trusted_security_settings,
)

MEMORY_TYPES: tuple[str, ...] = ("user", "feedback", "project", "reference")
MEMORY_INDEX = "MEMORY.md"
DEFAULT_MAX_INDEX_LINES = 200
DEFAULT_MAX_INDEX_BYTES = 25 * 1024

_TRUSTED_PATH_SOURCES: tuple[SettingSource, ...] = (
    "userSettings",
    "localSettings",
    "flagSettings",
    "policySettings",
)


@dataclass(frozen=True)
class MemoryPathInfo:
    directory: Path
    enabled: bool
    source: str
    source_path: str | None = None
    ignored_project_directory: str | None = None
    warning: str | None = None


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    type: str | None
    name: str
    description: str
    path: Path
    content: str

    @property
    def relative_path(self) -> str:
        return self.path.name


def memory_path_info(cfg) -> MemoryPathInfo:
    snapshot: SettingsSnapshot | None = getattr(cfg, "settings_snapshot", None)
    trusted = trusted_security_settings(snapshot) if snapshot else {}
    effective = snapshot.effective if snapshot else {}

    enabled = _setting_bool(effective, "autoMemoryEnabled", True)
    enabled = _setting_bool(effective, "memory.enabled", enabled)

    ignored_project = _project_auto_memory_directory(snapshot)
    configured, source, source_path = _trusted_auto_memory_directory(snapshot)
    if configured:
        resolved, warning = _resolve_configured_memory_dir(configured)
        if warning:
            return MemoryPathInfo(
                default_memory_dir(getattr(cfg, "workdir", Path.cwd())),
                enabled,
                "default",
                ignored_project_directory=ignored_project,
                warning=warning,
            )
        return MemoryPathInfo(
            resolved,
            enabled,
            source,
            source_path=source_path,
            ignored_project_directory=ignored_project,
        )

    # Also support a nested trusted alias for users who prefer grouped memory
    # settings. The top-level name mirrors the reference implementation.
    nested = nested_get(trusted, "memory.directory")
    if isinstance(nested, str) and nested.strip():
        resolved, warning = _resolve_configured_memory_dir(nested)
        if warning:
            return MemoryPathInfo(
                default_memory_dir(getattr(cfg, "workdir", Path.cwd())),
                enabled,
                "default",
                ignored_project_directory=ignored_project,
                warning=warning,
            )
        return MemoryPathInfo(
            resolved,
            enabled,
            "trusted memory.directory",
            ignored_project_directory=ignored_project,
        )

    return MemoryPathInfo(
        default_memory_dir(getattr(cfg, "workdir", Path.cwd())),
        enabled,
        "default",
        ignored_project_directory=ignored_project,
    )


def default_memory_dir(workdir: Path) -> Path:
    raw_home = os.environ.get("TINY_HARNESS_CONFIG_HOME", "").strip()
    home = Path(raw_home).expanduser() if raw_home else Path.home() / ".tiny-harness"
    return home / "memory" / "projects" / sanitize_project_key(Path(workdir)) / "memory"


def ensure_memory_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)


def load_memory_records(directory: Path) -> list[MemoryRecord]:
    if not directory.exists():
        return []
    records: list[MemoryRecord] = []
    for path in sorted(directory.glob("*.md")):
        if path.name == MEMORY_INDEX:
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        meta, content = parse_frontmatter(raw)
        raw_type = meta.get("type")
        mem_type = raw_type if raw_type in MEMORY_TYPES else None
        name = str(meta.get("name") or path.stem.replace("_", " ")).strip()
        description = str(meta.get("description") or first_content_line(content)).strip()
        records.append(MemoryRecord(
            id=path.stem,
            type=mem_type,
            name=name,
            description=description,
            path=path,
            content=content.strip(),
        ))
    return records


def recent_memory_records(directory: Path, limit: int = 5) -> list[MemoryRecord]:
    records = load_memory_records(directory)
    return sorted(
        records,
        key=lambda record: record.path.stat().st_mtime if record.path.exists() else 0,
        reverse=True,
    )[:limit]


def read_memory_index(directory: Path, *, max_lines: int = DEFAULT_MAX_INDEX_LINES,
                      max_bytes: int = DEFAULT_MAX_INDEX_BYTES) -> str:
    path = directory / MEMORY_INDEX
    if not path.exists():
        return ""
    try:
        return truncate_index(path.read_text(encoding="utf-8"), max_lines, max_bytes)
    except OSError:
        return ""


def rebuild_memory_index(directory: Path, *, max_lines: int = DEFAULT_MAX_INDEX_LINES,
                         max_bytes: int = DEFAULT_MAX_INDEX_BYTES) -> str:
    ensure_memory_dir(directory)
    records = load_memory_records(directory)
    lines = ["# Memory Index", ""]
    if not records:
        lines.append("(no memories)")
    for mem_type in MEMORY_TYPES:
        typed = [record for record in records if record.type == mem_type]
        if not typed:
            continue
        lines.extend(["", f"## {mem_type}"])
        for record in typed:
            hook = record.description or first_content_line(record.content)
            lines.append(f"- [{record.name}]({record.relative_path}) - {_one_line(hook, 150)}")
    legacy = [record for record in records if record.type is None]
    if legacy:
        lines.extend(["", "## legacy"])
        for record in legacy:
            hook = record.description or first_content_line(record.content)
            lines.append(f"- [{record.name}]({record.relative_path}) - {_one_line(hook, 150)}")
    text = truncate_index("\n".join(lines).rstrip() + "\n", max_lines, max_bytes)
    (directory / MEMORY_INDEX).write_text(text, encoding="utf-8")
    return text


def write_memory(directory: Path, mem_type: str, name: str, description: str,
                 content: str) -> MemoryRecord:
    if mem_type not in MEMORY_TYPES:
        raise ValueError(f"invalid memory type: {mem_type}")
    ensure_memory_dir(directory)
    safe_name = slugify(name or description or mem_type)
    digest = hashlib.sha1(f"{mem_type}\n{name}\n{content}".encode("utf-8")).hexdigest()[:8]
    path = directory / f"{mem_type}_{safe_name}_{digest}.md"
    body = render_memory_file(mem_type, name, description, content)
    path.write_text(body, encoding="utf-8")
    rebuild_memory_index(directory)
    return MemoryRecord(
        id=path.stem,
        type=mem_type,
        name=name.strip() or path.stem,
        description=description.strip(),
        path=path,
        content=content.strip(),
    )


def read_memory(directory: Path, ident: str) -> MemoryRecord:
    path = resolve_memory_file(directory, ident)
    raw = path.read_text(encoding="utf-8")
    meta, content = parse_frontmatter(raw)
    raw_type = meta.get("type")
    return MemoryRecord(
        id=path.stem,
        type=raw_type if raw_type in MEMORY_TYPES else None,
        name=str(meta.get("name") or path.stem),
        description=str(meta.get("description") or first_content_line(content)),
        path=path,
        content=content.strip(),
    )


def forget_memory(directory: Path, ident: str) -> Path:
    path = resolve_memory_file(directory, ident)
    path.unlink()
    rebuild_memory_index(directory)
    return path


def resolve_memory_file(directory: Path, ident: str) -> Path:
    if not ident or "\x00" in ident:
        raise ValueError("memory id/path is required")
    base = directory.resolve()
    raw = Path(ident)
    candidates = []
    if raw.suffix:
        candidates.append(base / raw.name)
    else:
        candidates.append(base / f"{raw.name}.md")
        candidates.extend(sorted(base.glob(f"{raw.name}*.md")))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved == base or base not in resolved.parents:
            continue
        if resolved.name == MEMORY_INDEX:
            continue
        if resolved.exists() and resolved.is_file():
            return resolved
    raise FileNotFoundError(f"memory not found: {ident}")


def render_memory_file(mem_type: str, name: str, description: str, content: str) -> str:
    return "\n".join([
        "---",
        f"name: {frontmatter_scalar(name.strip())}",
        f"description: {frontmatter_scalar(description.strip())}",
        f"type: {mem_type}",
        f"created_at: {datetime.now(timezone.utc).isoformat()}",
        "---",
        "",
        content.strip(),
        "",
    ])


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    raw_meta = text[4:end].strip()
    body = text[end + 4:].lstrip("\r\n")
    meta: dict[str, str] = {}
    for line in raw_meta.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip("\"'")
    return meta, body


def truncate_index(text: str, max_lines: int = DEFAULT_MAX_INDEX_LINES,
                   max_bytes: int = DEFAULT_MAX_INDEX_BYTES) -> str:
    lines = text.splitlines()[:max_lines]
    truncated = "\n".join(lines).rstrip() + ("\n" if lines else "")
    raw = truncated.encode("utf-8")
    if len(raw) <= max_bytes:
        return truncated
    clipped = raw[:max_bytes].decode("utf-8", errors="ignore")
    return clipped.rstrip() + "\n"


def memory_index_limits(cfg) -> tuple[int, int]:
    effective = getattr(getattr(cfg, "settings_snapshot", None), "effective", {}) or {}
    max_lines = nested_get(effective, "memory.maxIndexLines", DEFAULT_MAX_INDEX_LINES)
    max_bytes = nested_get(effective, "memory.maxIndexBytes", DEFAULT_MAX_INDEX_BYTES)
    try:
        lines = max(1, int(max_lines))
    except (TypeError, ValueError):
        lines = DEFAULT_MAX_INDEX_LINES
    try:
        bytes_ = max(1024, int(max_bytes))
    except (TypeError, ValueError):
        bytes_ = DEFAULT_MAX_INDEX_BYTES
    return lines, bytes_


def render_memory_prompt(cfg) -> str:
    info = memory_path_info(cfg)
    if not info.enabled:
        return ""
    max_lines, max_bytes = memory_index_limits(cfg)
    index = read_memory_index(info.directory, max_lines=max_lines, max_bytes=max_bytes)
    if not index.strip():
        return ""
    return "\n".join([
        "",
        "",
        "# Memory",
        "",
        "Persistent memory is available below. Treat memory as clues, not truth.",
        "Memory types are closed: user, feedback, project, reference.",
        "Do not save code facts, git history, file structure, or temporary task state.",
        "Before acting on a memory that names current files, functions, flags, or behavior, verify the current workspace state with tools.",
        "If the user asks you to ignore memory, proceed as if this section were empty.",
        "",
        index.rstrip(),
    ])


def memory_summary(cfg) -> dict[str, Any]:
    info = memory_path_info(cfg)
    records = load_memory_records(info.directory) if info.enabled else []
    counts = {mem_type: 0 for mem_type in MEMORY_TYPES}
    legacy = 0
    for record in records:
        if record.type in counts:
            counts[record.type] += 1
        else:
            legacy += 1
    return {
        "enabled": info.enabled,
        "directory": str(info.directory),
        "source": info.source,
        "source_path": info.source_path,
        "ignored_project_directory": info.ignored_project_directory,
        "warning": info.warning,
        "count": len(records),
        "counts": counts,
        "legacy_count": legacy,
        "index_exists": (info.directory / MEMORY_INDEX).exists(),
    }


def sanitize_project_key(workdir: Path) -> str:
    resolved = str(Path(workdir).resolve()).replace("\\", "/").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", resolved).strip("-")
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:10]
    return f"{slug[:70]}-{digest}" if slug else digest


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return slug[:48] or "memory"


def frontmatter_scalar(text: str) -> str:
    cleaned = text.replace("\r", " ").replace("\n", " ").strip()
    return '"' + cleaned.replace("\\", "\\\\").replace('"', '\\"') + '"'


def first_content_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _trusted_auto_memory_directory(snapshot: SettingsSnapshot | None) -> tuple[str | None, str, str | None]:
    if snapshot is None:
        return None, "default", None
    chosen: tuple[str, str, str | None] | None = None
    for layer in snapshot.sources:
        if layer.source not in _TRUSTED_PATH_SOURCES:
            continue
        value = layer.settings.get("autoMemoryDirectory")
        if isinstance(value, str) and value.strip():
            chosen = (value, layer.source, layer.path)
    return chosen if chosen else (None, "default", None)


def _project_auto_memory_directory(snapshot: SettingsSnapshot | None) -> str | None:
    if snapshot is None:
        return None
    for layer in snapshot.sources:
        if layer.source == "projectSettings":
            value = layer.settings.get("autoMemoryDirectory")
            if isinstance(value, str) and value.strip():
                return value
    return None


def _resolve_configured_memory_dir(raw: str) -> tuple[Path, str | None]:
    if "\x00" in raw:
        return Path(), "autoMemoryDirectory contains a null byte"
    if raw.startswith("\\\\"):
        return Path(), "autoMemoryDirectory must not be a UNC path"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        return Path(), "autoMemoryDirectory must be absolute"
    try:
        resolved = path.resolve()
    except OSError as e:
        return Path(), f"autoMemoryDirectory cannot be resolved: {e}"
    anchor = Path(resolved.anchor)
    if resolved == anchor:
        return Path(), "autoMemoryDirectory must not be a filesystem root"
    return resolved, None


def _setting_bool(settings: Mapping[str, object], path: str, default: bool) -> bool:
    value = nested_get(settings, path, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _one_line(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(1, limit - 3)].rstrip() + "..."
