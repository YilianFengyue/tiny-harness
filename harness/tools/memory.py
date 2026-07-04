"""Persistent memory tools."""
from __future__ import annotations

from .. import memory as mem
from .registry import ToolContext, ToolError, tool


def _cfg(ctx: ToolContext):
    cfg = ctx.runtime.config
    if cfg is not None:
        return cfg

    class MinimalConfig:
        workdir = ctx.workdir
        settings_snapshot = None

    return MinimalConfig()


@tool(
    name="list_memories",
    description=(
        "List persistent memory records available to the agent. Use this when "
        "the user asks what you remember or when memory may contain relevant "
        "long-term preferences. Memory is a clue, not current truth."
    ),
    parameters={
        "type": "object",
        "properties": {
            "type": {
                "type": ["string", "null"],
                "description": "Optional memory type: user, feedback, project, reference, or null for all",
            },
            "query": {
                "type": ["string", "null"],
                "description": "Optional case-insensitive text filter over name and description",
            },
        },
    },
    read_only=True,
    concurrency_safe=True,
)
def list_memories(ctx: ToolContext, type: str | None = None,
                  query: str | None = None) -> str:
    cfg = _cfg(ctx)
    info = mem.memory_path_info(cfg)
    if not info.enabled:
        return "Memory is disabled."
    if type is not None and type not in mem.MEMORY_TYPES:
        raise ToolError(f"invalid memory type '{type}'. valid: {', '.join(mem.MEMORY_TYPES)}")
    records = mem.load_memory_records(info.directory)
    if type:
        records = [record for record in records if record.type == type]
    if query:
        needle = query.casefold()
        records = [
            record for record in records
            if needle in record.name.casefold() or needle in record.description.casefold()
        ]
    if not records:
        return "No matching memories."
    lines = []
    for record in records:
        kind = record.type or "legacy"
        lines.append(f"- {record.id} [{kind}] {record.name}: {record.description}")
    return "\n".join(lines)


@tool(
    name="read_memory",
    description=(
        "Read one persistent memory record by id or filename. Before acting on "
        "repo-specific facts from memory, verify the current workspace with tools."
    ),
    parameters={
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Memory id or filename"},
        },
    },
    read_only=True,
    concurrency_safe=True,
)
def read_memory(ctx: ToolContext, id: str) -> str:
    cfg = _cfg(ctx)
    info = mem.memory_path_info(cfg)
    if not info.enabled:
        return "Memory is disabled."
    try:
        record = mem.read_memory(info.directory, id)
    except Exception as e:
        raise ToolError(str(e))
    return "\n".join([
        f"id: {record.id}",
        f"type: {record.type or 'legacy'}",
        f"name: {record.name}",
        f"description: {record.description}",
        "",
        record.content,
    ]).rstrip()


@tool(
    name="write_memory",
    description=(
        "Save durable, non-derivable information to persistent memory. Only use "
        "this for user profile, feedback, project state, or external reference "
        "information. Do not save code facts, git history, file structure, or "
        "temporary task details."
    ),
    parameters={
        "type": "object",
        "properties": {
            "type": {"type": "string", "description": "user, feedback, project, or reference"},
            "name": {"type": "string", "description": "Short memory title"},
            "description": {"type": "string", "description": "One-line hook for future relevance"},
            "content": {"type": "string", "description": "Memory body, including Why/How to apply when useful"},
        },
    },
    destructive=True,
)
def write_memory(ctx: ToolContext, type: str, name: str,
                 description: str, content: str) -> str:
    cfg = _cfg(ctx)
    info = mem.memory_path_info(cfg)
    if not info.enabled:
        return "Memory is disabled."
    try:
        record = mem.write_memory(info.directory, type, name, description, content)
    except Exception as e:
        raise ToolError(str(e))
    return f"[◈ Memory Saved] {record.id} [{record.type}] {record.name}"


@tool(
    name="forget_memory",
    description="Remove a persistent memory record by id or filename when the user asks you to forget it.",
    parameters={
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Memory id or filename to remove"},
        },
    },
    destructive=True,
)
def forget_memory(ctx: ToolContext, id: str) -> str:
    cfg = _cfg(ctx)
    info = mem.memory_path_info(cfg)
    if not info.enabled:
        return "Memory is disabled."
    try:
        path = mem.forget_memory(info.directory, id)
    except Exception as e:
        raise ToolError(str(e))
    return f"[◈ Memory Removed] {path.name}"
