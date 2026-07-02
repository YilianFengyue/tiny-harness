"""文件读写工具：所有路径锁定在 workdir 沙箱内。

路径校验：resolve()（消解 ..、符号链接）后必须仍在 workdir 下。
字符串层面的过滤（如检查 ".."）不可靠——symlink、大小写、UNC 路径都能绕，
必须在解析后的绝对路径上判定。
"""
from __future__ import annotations

import difflib
import subprocess
from pathlib import Path

from .registry import ToolContext, ToolError, ToolResult, ToolRuntimeState, tool


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


def _validate_path_arg(ctx: ToolContext, arguments: dict) -> None:
    path = arguments.get("path")
    if path is not None and not isinstance(path, str):
        raise ToolError("path must be a string path relative to the workspace root")


def _record_read(path: Path, content: str, offset: int, max_lines: int):
    def modifier(runtime: ToolRuntimeState) -> dict:
        stat = path.stat()
        runtime.read_files[str(path)] = {
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
            "content": content,
            "offset": offset,
            "max_lines": max_lines,
        }
        return {"kind": "read_file_state", "path": str(path), "size": stat.st_size}
    return modifier


def _read_text_for_write(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ToolError(f"'{path.name}' is not valid UTF-8 text; refusing to edit/write it")
    except OSError as e:
        raise ToolError(f"cannot read '{path.name}' before writing: {e}")


def _require_fresh_read(ctx: ToolContext, path: Path) -> str:
    state = ctx.runtime.read_files.get(str(path))
    if state is None:
        raise ToolError(
            f"existing file '{path.name}' must be read with read_file before it is modified. "
            "Read it first, then use edit_file for small changes or write_file for a full rewrite.")
    current = _read_text_for_write(path)
    if current != state.get("content"):
        raise ToolError(
            f"file '{path.name}' has changed since it was read. Read it again before modifying it.")
    return current


def _line_count(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _unified_diff(before: str, after: str, path: str, max_lines: int = 80) -> str:
    lines = list(difflib.unified_diff(
        before.splitlines(), after.splitlines(),
        fromfile=f"a/{path}", tofile=f"b/{path}", lineterm=""))
    if len(lines) > max_lines:
        omitted = len(lines) - max_lines
        lines = lines[:max_lines] + [f"... [diff truncated: {omitted} lines omitted]"]
    return "\n".join(lines) if lines else "(no diff)"


def _record_write(path: Path, action: str, chars: int,
                  before: str | None = None, after: str | None = None):
    def modifier(runtime: ToolRuntimeState) -> dict:
        stat = path.stat()
        runtime.file_history.append({
            "action": action,
            "path": str(path),
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
            "chars": chars,
            "before": before,
            "after": after,
        })
        runtime.read_files[str(path)] = {
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
            "content": after if after is not None else path.read_text(encoding="utf-8", errors="replace"),
            "offset": 1,
            "max_lines": None,
        }
        return {"kind": "file_write_state", "path": str(path), "action": action,
                "chars": chars}
    return modifier


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
    read_only=True,
    concurrency_safe=True,
    validate_input=_validate_path_arg,
)
def read_file(ctx: ToolContext, path: str, offset: int | None = None,
              max_lines: int | None = None) -> ToolResult:
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
        content = target.read_text(encoding="utf-8", errors="replace")
        all_lines = content.splitlines()
    except OSError as e:
        raise ToolError(f"cannot read '{path}': {e}")
    chunk = all_lines[offset - 1: offset - 1 + max_lines]
    if not chunk:
        return ToolResult(
            f"(no content: file has {len(all_lines)} lines, offset {offset} is past the end)",
            context_modifier=_record_read(target, content, offset, max_lines))
    body = "\n".join(f"{i:>6}\t{line}" for i, line in enumerate(chunk, start=offset))
    note = ""
    if offset - 1 + len(chunk) < len(all_lines):
        note = (f"\n... [{len(all_lines) - (offset - 1 + len(chunk))} more lines; "
                f"continue with offset={offset + len(chunk)}]")
    return ToolResult(f"{target.name} ({len(all_lines)} lines total)\n{body}{note}",
                      context_modifier=_record_read(target, content, offset, max_lines))


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
    destructive=True,
    validate_input=_validate_path_arg,
)
def write_file(ctx: ToolContext, path: str, content: str) -> ToolResult:
    target = resolve_in_workdir(ctx, path)
    existed = target.exists()
    before = _require_fresh_read(ctx, target) if existed else None
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    action = "update" if existed else "create"
    line_note = f"{_line_count(content)} lines"
    return ToolResult(f"{action}d {len(content)} chars ({line_note}) to {path}",
                      context_modifier=_record_write(target, action, len(content),
                                                     before, content))


@tool(
    name="edit_file",
    description=(
        "Precisely edit an existing UTF-8 text file inside the workspace by replacing "
        "old_string with new_string. You MUST read_file the file first. By default "
        "old_string must match exactly once; if it matches multiple places, provide "
        "more surrounding context or set replace_all=true for intentional bulk "
        "renames. Prefer this over write_file for small code changes."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the workspace root"},
            "old_string": {"type": "string", "description": "Exact text to replace"},
            "new_string": {"type": "string", "description": "Replacement text"},
            "replace_all": {"type": ["boolean", "null"],
                            "description": "Replace all occurrences (null/false = require unique match)"},
        },
    },
    destructive=True,
    validate_input=_validate_path_arg,
)
def edit_file(ctx: ToolContext, path: str, old_string: str, new_string: str,
              replace_all: bool | None = None) -> ToolResult:
    target = resolve_in_workdir(ctx, path)
    if not target.exists():
        raise ToolError(f"file '{path}' does not exist. Use write_file to create new files.")
    if target.is_dir():
        raise ToolError(f"'{path}' is a directory, not a file")
    if old_string == "":
        raise ToolError("old_string must be non-empty for edit_file")
    if old_string == new_string:
        raise ToolError("old_string and new_string are identical; no edit would be made")

    before = _require_fresh_read(ctx, target)
    matches = before.count(old_string)
    if matches == 0:
        raise ToolError(
            f"String to replace not found in '{path}'. Re-read the file and copy the exact text, "
            "including indentation, but do not include read_file line numbers.")
    replace_all = bool(replace_all)
    if matches > 1 and not replace_all:
        raise ToolError(
            f"Found {matches} matches of old_string in '{path}', but replace_all is false. "
            "Provide a longer unique old_string with surrounding context or set replace_all=true.")

    after = before.replace(old_string, new_string) if replace_all else before.replace(old_string, new_string, 1)
    target.write_text(after, encoding="utf-8")
    diff = _unified_diff(before, after, path)
    count = matches if replace_all else 1
    text = f"edited {path}: replaced {count} occurrence(s)\n{diff}"
    return ToolResult(text, context_modifier=_record_write(target, "edit", len(after), before, after))


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
    read_only=True,
    concurrency_safe=True,
    validate_input=_validate_path_arg,
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


@tool(
    name="file_info",
    description=(
        "Return metadata for a file or directory inside the workspace: kind, size, "
        "mtime, UTF-8 readability, and line count for text files. Use this before "
        "reading very large files."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the workspace root"},
        },
    },
    read_only=True,
    concurrency_safe=True,
    validate_input=_validate_path_arg,
)
def file_info(ctx: ToolContext, path: str) -> str:
    target = resolve_in_workdir(ctx, path)
    if not target.exists():
        return f"{path}: missing"
    stat = target.stat()
    if target.is_dir():
        entries = sum(1 for _ in target.iterdir())
        return f"{path}: directory, entries={entries}, mtime_ns={stat.st_mtime_ns}"
    try:
        text = target.read_text(encoding="utf-8")
        text_note = f"text=utf-8, lines={_line_count(text)}"
    except UnicodeDecodeError:
        text_note = "text=binary-or-non-utf8, lines=unknown"
    return f"{path}: file, size={stat.st_size} bytes, mtime_ns={stat.st_mtime_ns}, {text_note}"


@tool(
    name="show_diff",
    description=(
        "Show changes made in the workspace. In a git repository, returns git diff. "
        "Outside git, returns the most recent file change tracked by tiny-harness."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": ["string", "null"],
                     "description": "Optional path relative to workspace root (null = all changes)"},
        },
    },
    read_only=True,
    concurrency_safe=True,
    validate_input=_validate_path_arg,
)
def show_diff(ctx: ToolContext, path: str | None = None) -> str:
    args = ["git", "diff", "--"]
    if path:
        target = resolve_in_workdir(ctx, path)
        args.append(str(target.relative_to(ctx.workdir.resolve())))
    try:
        proc = subprocess.run(args, cwd=str(ctx.workdir), capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=10)
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout
    except Exception:
        pass

    history = ctx.runtime.file_history
    if not history:
        return "(no tracked file changes)"
    item = next((h for h in reversed(history)
                 if not path or Path(h["path"]).resolve() == resolve_in_workdir(ctx, path)), history[-1])
    before, after = item.get("before"), item.get("after")
    if before is None or after is None:
        return f"latest change: {item.get('action')} {item.get('path')}"
    rel = str(Path(item["path"]).resolve().relative_to(ctx.workdir.resolve()))
    return _unified_diff(before, after, rel, max_lines=200)
