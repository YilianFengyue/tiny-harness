"""代码搜索工具：文件名 glob 与内容 grep。

专用搜索工具比 bash grep/find 更适合 coding agent：只读、可并发、输出稳定、
有结果上限和明确的缩小提示。
"""
from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path

from .files import resolve_in_workdir
from .registry import ToolContext, ToolError, tool

DEFAULT_EXCLUDES = {".git", "__pycache__", ".pytest_cache", ".tiny-harness", "runs"}


def _validate_pattern(ctx: ToolContext, arguments: dict) -> None:
    pattern = arguments.get("pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        raise ToolError("pattern must be a non-empty string")
    path = arguments.get("path")
    if path is not None and not isinstance(path, str):
        raise ToolError("path must be null or a string relative to the workspace root")


def _iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in DEFAULT_EXCLUDES]
        for name in filenames:
            yield Path(dirpath) / name


def _rel(path: Path, workdir: Path) -> str:
    return path.resolve().relative_to(workdir.resolve()).as_posix()


def _limit_note(total: int, shown: int, hint: str) -> str:
    if total <= shown:
        return ""
    return f"\n... [{total - shown} more results omitted; {hint}]"


@tool(
    name="glob_files",
    description=(
        "Find files by glob pattern inside the workspace. Use this to discover "
        "project structure or locate files by name before reading them. Results "
        "are stable-sorted and capped; narrow path/pattern if capped."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py' or 'src/**/test_*.py'"},
            "path": {"type": ["string", "null"], "description": "Directory to search (null = workspace root)"},
            "max_results": {"type": ["integer", "null"], "description": "Max paths to return (null = 200)"},
        },
    },
    read_only=True,
    concurrency_safe=True,
    validate_input=_validate_pattern,
)
def glob_files(ctx: ToolContext, pattern: str, path: str | None = None,
               max_results: int | None = None) -> str:
    root = resolve_in_workdir(ctx, path or ".")
    if not root.exists():
        raise ToolError(f"search path '{path}' does not exist")
    if root.is_file():
        root = root.parent
    max_results = max(1, min(max_results or 200, 1000))
    rg = shutil.which("rg")
    results: list[str] = []
    if rg:
        try:
            proc = subprocess.run([rg, "--files"], cwd=str(root), capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", timeout=20)
            if proc.returncode in (0, 1):
                for line in proc.stdout.splitlines():
                    if any(part in DEFAULT_EXCLUDES for part in Path(line).parts):
                        continue
                    rel_to_root = line.replace("\\", "/")
                    if fnmatch.fnmatch(rel_to_root, pattern) or fnmatch.fnmatch(_rel(root / line, ctx.workdir), pattern):
                        results.append(_rel(root / line, ctx.workdir))
        except Exception:
            results = []
    if not results:
        for file in _iter_files(root):
            rel_root = file.relative_to(root).as_posix()
            rel_work = _rel(file, ctx.workdir)
            if fnmatch.fnmatch(rel_root, pattern) or fnmatch.fnmatch(rel_work, pattern):
                results.append(rel_work)
    results = sorted(dict.fromkeys(results))
    shown = results[:max_results]
    if not shown:
        return f"(no files matched pattern {pattern!r})"
    return "\n".join(shown) + _limit_note(
        len(results), len(shown), "narrow pattern/path or increase max_results")


@tool(
    name="grep",
    description=(
        "Search UTF-8 text files inside the workspace for a regex pattern and "
        "return path:line:content matches. Use grep to locate code before "
        "reading/editing specific files. Supports include glob like '*.py'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression to search for"},
            "path": {"type": ["string", "null"], "description": "File or directory to search (null = workspace root)"},
            "include": {"type": ["string", "null"], "description": "Optional file glob filter, e.g. '*.py'"},
            "max_results": {"type": ["integer", "null"], "description": "Max matches to return (null = 100)"},
            "context_lines": {"type": ["integer", "null"], "description": "Context lines before/after each match (null = 0)"},
        },
    },
    read_only=True,
    concurrency_safe=True,
    validate_input=_validate_pattern,
)
def grep(ctx: ToolContext, pattern: str, path: str | None = None,
         include: str | None = None, max_results: int | None = None,
         context_lines: int | None = None) -> str:
    target = resolve_in_workdir(ctx, path or ".")
    if not target.exists():
        raise ToolError(f"search path '{path}' does not exist")
    max_results = max(1, min(max_results or 100, 1000))
    context_lines = max(0, min(context_lines or 0, 5))

    results = _grep_with_rg(ctx, pattern, target, include, max_results, context_lines)
    if results is None:
        results = _grep_with_python(ctx, pattern, target, include, max_results, context_lines)
    if not results:
        return f"(no matches for pattern {pattern!r})"
    return "\n".join(results) + (
        f"\n... [results capped at {max_results}; narrow pattern/path/include]"
        if len(results) >= max_results else "")


def _grep_with_rg(ctx: ToolContext, pattern: str, target: Path, include: str | None,
                  max_results: int, context_lines: int) -> list[str] | None:
    rg = shutil.which("rg")
    if not rg:
        return None
    args = [rg, "-n", "--no-heading", "--color", "never", "--max-count", str(max_results)]
    if context_lines:
        args += ["-C", str(context_lines)]
    if include:
        args += ["--glob", include]
    target_arg = "."
    try:
        target_arg = target.resolve().relative_to(ctx.workdir.resolve()).as_posix()
    except ValueError:
        target_arg = str(target)
    args += [pattern, target_arg]
    try:
        proc = subprocess.run(args, cwd=str(ctx.workdir), capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=20)
    except Exception:
        return None
    if proc.returncode not in (0, 1):
        return None
    lines = []
    for raw in proc.stdout.splitlines():
        lines.append(_normalize_rg_line(raw, ctx.workdir))
        if len(lines) >= max_results:
            break
    return lines


def _normalize_rg_line(raw: str, workdir: Path) -> str:
    text = raw.replace("\\", "/")
    work = workdir.resolve()
    work_s = str(work).replace("\\", "/")
    if text.startswith(work_s + "/"):
        rest = text[len(work_s) + 1:]
        return rest
    return text


def _grep_with_python(ctx: ToolContext, pattern: str, target: Path, include: str | None,
                      max_results: int, context_lines: int) -> list[str]:
    try:
        rx = re.compile(pattern)
    except re.error as e:
        raise ToolError(f"invalid regex pattern: {e}")
    files = [target] if target.is_file() else list(_iter_files(target))
    results: list[str] = []
    for file in files:
        rel = _rel(file, ctx.workdir)
        if include and not fnmatch.fnmatch(file.name, include) and not fnmatch.fnmatch(rel, include):
            continue
        try:
            lines = file.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(lines, start=1):
            if not rx.search(line):
                continue
            start = max(1, i - context_lines)
            end = min(len(lines), i + context_lines)
            for n in range(start, end + 1):
                prefix = ">" if n == i else " "
                results.append(f"{rel}:{n}:{prefix} {lines[n - 1]}")
                if len(results) >= max_results:
                    return results
    return results
