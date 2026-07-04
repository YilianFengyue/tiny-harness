"""Human-readable memory summaries for TUI and tests."""
from __future__ import annotations

from .memory import (
    MEMORY_TYPES,
    forget_memory,
    load_memory_records,
    memory_path_info,
    memory_summary,
    read_memory,
    read_memory_index,
    recent_memory_records,
    rebuild_memory_index,
    render_memory_prompt,
    write_memory,
)


def memory_status_line(cfg) -> str:
    summary = memory_summary(cfg)
    state = "on" if summary["enabled"] else "off"
    return f"memory:{state} {summary['count']} entries dir:{summary['source']}"


def format_memory_summary(cfg, section: str = "") -> str:
    section = section.strip()
    if section == "sources":
        return format_memory_sources(cfg)
    if section == "prompt":
        return format_memory_prompt_preview(cfg)
    if section == "status":
        return format_memory_runtime_status(cfg)
    if section.startswith("tail"):
        _, _, raw_limit = section.partition(" ")
        return format_memory_tail(cfg, raw_limit.strip())
    if section.startswith("list"):
        _, _, mem_type = section.partition(" ")
        return format_memory_list(cfg, mem_type.strip() or None)
    if section.startswith("read "):
        return format_memory_read(cfg, section.partition(" ")[2].strip())
    return "\n\n".join([
        _format_memory_header(cfg),
        format_memory_sources(cfg),
        format_memory_list(cfg, None),
    ])


def format_memory_sources(cfg) -> str:
    info = memory_path_info(cfg)
    lines = [
        "Memory sources:",
        f"- enabled: {info.enabled}",
        f"- directory: {info.directory}",
        f"- source: {info.source}",
    ]
    if info.source_path:
        lines.append(f"- source_path: {info.source_path}")
    if info.ignored_project_directory:
        lines.append(
            f"- ignored project autoMemoryDirectory: {info.ignored_project_directory}")
    if info.warning:
        lines.append(f"- warning: {info.warning}")
    return "\n".join(lines)


def format_memory_list(cfg, mem_type: str | None = None) -> str:
    info = memory_path_info(cfg)
    records = load_memory_records(info.directory) if info.enabled else []
    if mem_type:
        if mem_type not in MEMORY_TYPES:
            return f"Invalid memory type: {mem_type}. Valid: {', '.join(MEMORY_TYPES)}"
        records = [record for record in records if record.type == mem_type]
    title = f"Memory list{f' ({mem_type})' if mem_type else ''}:"
    if not records:
        return title + "\n- none"
    lines = [title]
    for record in records:
        kind = record.type or "legacy"
        lines.append(
            f"- {record.id} [{kind}] {record.name} - {record.description}")
    return "\n".join(lines)


def format_memory_read(cfg, ident: str) -> str:
    if not ident:
        return "Usage: /memory read <id|path>"
    info = memory_path_info(cfg)
    try:
        record = read_memory(info.directory, ident)
    except Exception as e:
        return f"Memory read failed: {e}"
    return "\n".join([
        f"Memory: {record.id}",
        f"type: {record.type or 'legacy'}",
        f"name: {record.name}",
        f"description: {record.description}",
        f"path: {record.path}",
        "",
        record.content,
    ]).rstrip()


def format_memory_prompt_preview(cfg) -> str:
    prompt = render_memory_prompt(cfg)
    if not prompt.strip():
        return "Memory prompt: (empty)"
    return "Memory prompt preview:\n" + prompt.strip()


def format_memory_runtime_status(cfg, controller=None) -> str:
    summary = memory_summary(cfg)
    lines = [
        "[◈ Memory Status]",
        f"enabled: {summary['enabled']}",
        f"entries: {summary['count']}",
        f"directory: {summary['directory']}",
    ]
    if controller is not None:
        status = controller.status()
        lines.extend([
            f"auto_extract: {status['enabled']}",
            f"extracting: {status['extracting']}",
            f"pending: {status['pending']}",
            f"last_status: {status['last_status']}",
            f"last_saved: {', '.join(status['last_saved']) if status['last_saved'] else 'none'}",
        ])
        if status["last_error"]:
            lines.append(f"last_error: {status['last_error']}")
    return "\n".join(lines)


def format_memory_tail(cfg, raw_limit: str = "") -> str:
    info = memory_path_info(cfg)
    try:
        limit = int(raw_limit) if raw_limit else 5
    except ValueError:
        limit = 5
    records = recent_memory_records(info.directory, max(1, limit)) if info.enabled else []
    if not records:
        return "Memory tail:\n- none"
    lines = ["Memory tail:"]
    for record in records:
        lines.append(f"- {record.id} [{record.type or 'legacy'}] {record.name} - {record.description}")
    return "\n".join(lines)


def add_memory_from_text(cfg, text: str) -> str:
    mem_type, name, description, content = _parse_add_text(text)
    info = memory_path_info(cfg)
    if not info.enabled:
        return "Memory is disabled."
    try:
        record = write_memory(info.directory, mem_type, name, description, content)
    except Exception as e:
        return f"Memory write failed: {e}"
    return f"[◈ Memory Saved] {record.id} [{record.type}] {record.name}"


def forget_memory_from_text(cfg, text: str) -> str:
    ident = text.strip()
    if not ident:
        return "Usage: /memory forget <id|path>"
    info = memory_path_info(cfg)
    try:
        path = forget_memory(info.directory, ident)
    except Exception as e:
        return f"Memory forget failed: {e}"
    return f"[◈ Memory Removed] {path.name}"


def rebuild_memory_index_for_cfg(cfg) -> str:
    info = memory_path_info(cfg)
    if not info.enabled:
        return "Memory is disabled."
    rebuild_memory_index(info.directory)
    return "[◈ Memory Index Rebuilt]"


def extract_memory_for_session(session) -> str:
    events = session.extract_memory(force=True)
    saved = [event for event in events if event.get("type") == "memory_extract_saved"]
    if saved:
        count = saved[-1].get("count", 0)
        paths = ", ".join(saved[-1].get("paths") or [])
        return f"[◈ Memory Saved] {count}: {paths}"
    skipped = [event for event in events if event.get("type") == "memory_extract_skipped"]
    if skipped:
        return f"◈ Extracting.. skipped: {skipped[-1].get('reason')}"
    errors = [event for event in events if event.get("type") == "memory_extract_error"]
    if errors:
        return f"◈ Extracting.. error: {errors[-1].get('error')}"
    return "◈ Extracting.. done"


def _format_memory_header(cfg) -> str:
    summary = memory_summary(cfg)
    counts = ", ".join(f"{kind}={summary['counts'][kind]}" for kind in MEMORY_TYPES)
    index_note = "yes" if summary["index_exists"] else "no"
    loaded = "Loaded" if summary["count"] else "Empty"
    return "\n".join([
        f"[◈ Memory {loaded}]",
        f"enabled: {summary['enabled']}",
        f"entries: {summary['count']} ({counts}, legacy={summary['legacy_count']})",
        f"index: {index_note}",
    ])


def _parse_add_text(text: str) -> tuple[str, str, str, str]:
    parts = [part.strip() for part in text.split("|")]
    head = parts[0] if parts else ""
    head_parts = head.split(maxsplit=1)
    if len(head_parts) < 2:
        raise ValueError(
            "Usage: /memory add <user|feedback|project|reference> <title> | <description> | <content>")
    mem_type, name = head_parts
    if mem_type not in MEMORY_TYPES:
        raise ValueError(f"invalid memory type: {mem_type}")
    description = parts[1] if len(parts) > 1 and parts[1] else name
    content = parts[2] if len(parts) > 2 and parts[2] else description
    return mem_type, name, description, content
