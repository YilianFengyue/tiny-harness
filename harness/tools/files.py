"""文件读写工具：所有路径锁定在 workdir 沙箱内。

路径校验：resolve()（消解 ..、符号链接）后必须仍在 workdir 下。
字符串层面的过滤（如检查 ".."）不可靠——symlink、大小写、UNC 路径都能绕，
必须在解析后的绝对路径上判定。
"""
from __future__ import annotations

from pathlib import Path

from .registry import ToolContext, ToolError, tool


def resolve_in_workdir(ctx: ToolContext, path: str) -> Path:
    """把（相对）路径解析进 workdir，逃逸即抛可恢复错误。"""
    p = Path(path)
    candidate = (p if p.is_absolute() else ctx.workdir / p).resolve()
    workdir = ctx.workdir.resolve()
    if candidate != workdir and workdir not in candidate.parents:
        raise ToolError(
            f"path '{path}' escapes the workspace. All paths must stay inside "
            f"the workspace root; use paths relative to it, e.g. 'data.csv' or 'out/result.txt'.")
    return candidate


def _dir_listing(d: Path, limit: int = 50) -> str:
    entries = sorted(d.iterdir(), key=lambda e: (e.is_file(), e.name))[:limit]
    if not entries:
        return "(empty directory)"
    return ", ".join(e.name + ("/" if e.is_dir() else "") for e in entries)


@tool(
    name="read_file",
    description=(
        "Read a UTF-8 text file inside the workspace and return its content with "
        "1-based line numbers. Large files are paged: pass offset (start line) and "
        "max_lines to read further chunks. For files too large to page through, or "
        "for binary files, use the bash tool (head, wc -l, awk, python) instead."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the workspace root"},
            "offset": {"type": ["integer", "null"],
                       "description": "1-based line to start from (null = 1)"},
            "max_lines": {"type": ["integer", "null"],
                          "description": "Max lines to return (null = 500)"},
        },
    },
)
def read_file(ctx: ToolContext, path: str, offset: int | None = None,
              max_lines: int | None = None) -> str:
    target = resolve_in_workdir(ctx, path)
    if not target.exists():
        parent = target.parent if target.parent.exists() else ctx.workdir
        raise ToolError(f"file '{path}' not found. Files in '{parent.name or '.'}': "
                        f"{_dir_listing(parent)}")
    if target.is_dir():
        raise ToolError(f"'{path}' is a directory, not a file. It contains: {_dir_listing(target)}")
    offset = max(offset or 1, 1)
    max_lines = max_lines or 500
    try:
        all_lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        raise ToolError(f"cannot read '{path}': {e}")
    chunk = all_lines[offset - 1: offset - 1 + max_lines]
    if not chunk:
        return f"(no content: file has {len(all_lines)} lines, offset {offset} is past the end)"
    body = "\n".join(f"{i:>6}\t{line}" for i, line in enumerate(chunk, start=offset))
    note = ""
    if offset - 1 + len(chunk) < len(all_lines):
        note = (f"\n... [{len(all_lines) - (offset - 1 + len(chunk))} more lines; "
                f"continue with offset={offset + len(chunk)}]")
    return f"{target.name} ({len(all_lines)} lines total)\n{body}{note}"


@tool(
    name="write_file",
    description=(
        "Write UTF-8 text to a file inside the workspace, creating parent directories "
        "as needed and overwriting any existing content. After writing a deliverable, "
        "read it back (read_file) to verify it contains exactly what you intended."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the workspace root"},
            "content": {"type": "string", "description": "Full file content to write"},
        },
    },
)
def write_file(ctx: ToolContext, path: str, content: str) -> str:
    target = resolve_in_workdir(ctx, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"


@tool(
    name="list_files",
    description=(
        "List files and directories inside the workspace (non-recursive). "
        "Use this first when you are unsure what files exist or a filename "
        "from the task doesn't match reality."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": ["string", "null"],
                     "description": "Directory relative to workspace root (null = root)"},
        },
    },
)
def list_files(ctx: ToolContext, path: str | None = None) -> str:
    target = resolve_in_workdir(ctx, path or ".")
    if not target.exists():
        raise ToolError(f"directory '{path}' not found. Workspace root contains: "
                        f"{_dir_listing(ctx.workdir)}")
    if target.is_file():
        size = target.stat().st_size
        return f"'{path}' is a file ({size} bytes), not a directory"
    lines = []
    for e in sorted(target.iterdir(), key=lambda e: (e.is_file(), e.name))[:200]:
        kind = "dir " if e.is_dir() else "file"
        size = "" if e.is_dir() else f"  {e.stat().st_size} bytes"
        lines.append(f"{kind}  {e.name}{size}")
    return "\n".join(lines) if lines else "(empty directory)"
