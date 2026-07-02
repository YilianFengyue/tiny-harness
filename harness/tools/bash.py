"""bash 工具：subprocess 软沙箱。

沙箱诚实声明（DESIGN.md §Sandbox 有完整讨论）：
本实现提供的是【事故防护】而非【对抗防护】——
  1. cwd 锁定 workdir + 超时杀进程树 + 输出截断（防挂起、防上下文爆炸）
  2. 危险命令模式匹配 → 交给 hook 决定（默认拒绝，可 --yolo 放行）
字符串过滤不是安全边界（base64/变量拼接/子 shell 均可绕过）。生产级隔离
应使用 OS 原语（bubblewrap/Seatbelt，参考 anthropic-experimental/sandbox-runtime）
或容器（terminal-bench/inspect_ai 的标准做法）。本考核场景威胁模型是
"模型犯傻"而非"模型作恶"，软沙箱 + 明示边界是匹配该威胁模型的工程选择。
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time

from .registry import ToolContext, ToolError, tool

# 命中即触发 hook 确认/拒绝。注释给人看：为什么是这些模式。
DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"\brm\s+(-[a-z]*[rf][a-z]*\s+)+(/|~|\$HOME|[A-Za-z]:)", "recursive delete near filesystem root"),
    (r"\bmkfs\b|\bdd\s+[^|]*of=/dev/", "raw disk write"),
    (r"\b(shutdown|reboot|halt)\b", "system power control"),
    (r":\(\)\s*\{.*\}\s*;\s*:", "fork bomb"),
    (r"\bsudo\b", "privilege escalation"),
    (r"(curl|wget)\s+[^|;&]*\|\s*(ba)?sh", "pipe remote script into shell"),
    (r">\s*/dev/sd[a-z]", "raw device overwrite"),
    (r"\bgit\s+push\s+.*--force", "force push"),
]

READ_ONLY_COMMANDS = {
    "cat", "dir", "echo", "find", "grep", "head", "ls", "pwd", "rg", "sort",
    "tail", "type", "uniq", "wc", "where", "whereis", "which",
    "Get-ChildItem", "Get-Content", "Select-String", "Measure-Object",
}

READ_ONLY_GIT_PREFIXES = ("git status", "git diff", "git log", "git show")


def check_dangerous(arguments: dict) -> str | None:
    cmd = arguments.get("command", "")
    for pattern, why in DANGEROUS_PATTERNS:
        if re.search(pattern, cmd, re.IGNORECASE):
            return why
    return None


def validate_bash_input(ctx: ToolContext, arguments: dict) -> None:
    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ToolError("command must be a non-empty shell command string")


def is_read_only_command(arguments: dict) -> bool:
    command = str(arguments.get("command", "")).strip()
    if not command:
        return False
    lowered = command.lower()
    if any(lowered == prefix or lowered.startswith(prefix + " ")
           for prefix in READ_ONLY_GIT_PREFIXES):
        return True
    if re.search(r"(^|[;&|])\s*(>|>>|set-content|out-file|remove-item|rm\b|mv\b|cp\b|"
                 r"python\b|node\b|npm\b|pip\b|git\s+(?!status|diff|log|show))",
                 command, re.IGNORECASE):
        return False
    parts = re.split(r"\s*(?:&&|\|\||;|\|)\s*", command)
    saw_command = False
    for part in parts:
        if not part.strip():
            continue
        try:
            argv = shlex.split(part, posix=sys.platform != "win32")
        except ValueError:
            return False
        if not argv:
            continue
        base = PathLikeCommand(argv[0])
        if base not in READ_ONLY_COMMANDS:
            return False
        saw_command = True
    return saw_command


def PathLikeCommand(value: str) -> str:
    return value.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


def _shell_argv(command: str) -> tuple[list[str] | str, bool]:
    bash_exe = shutil.which("bash")
    if sys.platform == "win32":
        # Windows 自带 C:\Windows\System32\bash.exe 只是 WSL launcher；未安装
        # distro 时会打印乱码并退出。Git Bash 才是这里想要的可用 bash。
        if bash_exe and "windows\\system32" not in bash_exe.lower():
            return [bash_exe, "-lc", command], False
        ps = shutil.which("pwsh") or shutil.which("powershell")
        if ps:
            return [ps, "-NoProfile", "-NonInteractive", "-ExecutionPolicy",
                    "Bypass", "-Command", command], False
        return command, True
    if bash_exe:
        return [bash_exe, "-c", command], False
    return command, True


def _kill_tree(proc: subprocess.Popen) -> None:
    if sys.platform == "win32":
        # Windows 下 proc.kill() 只杀直接子进程，taskkill /T 杀整棵树
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
    else:
        import signal
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


@tool(
    name="bash",
    description=(
        "Run a shell command inside the workspace and return exit code, stdout and "
        "stderr. The working directory is the workspace root; it is the only place "
        "you can write. Use this for anything the other tools can't do efficiently: "
        "processing large files (awk/head/wc), generating data, running python "
        "one-liners (python -c \"...\"). Commands run with a timeout; long-running or "
        "interactive commands (editors, watch, servers) will be killed. Destructive "
        "commands are blocked by policy — if a command is rejected, take a safer "
        "approach instead of rephrasing the same command."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
        },
    },
    dangerous_check=check_dangerous,
    read_only=is_read_only_command,
    concurrency_safe=is_read_only_command,
    validate_input=validate_bash_input,
)
def bash(ctx: ToolContext, command: str) -> str:
    if not command.strip():
        raise ToolError("empty command")
    argv, use_shell = _shell_argv(command)

    popen_kwargs: dict = dict(
        cwd=str(ctx.workdir), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        shell=use_shell,
    )
    if sys.platform != "win32":
        popen_kwargs["start_new_session"] = True  # 自成进程组，超时可整组击杀

    ctx.throw_if_cancelled()
    proc = subprocess.Popen(argv, **popen_kwargs)
    stdout_buf = bytearray()
    stderr_buf = bytearray()

    def drain(pipe, buf: bytearray) -> None:
        try:
            for chunk in iter(lambda: pipe.read(4096), b""):
                if chunk:
                    buf.extend(chunk)
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    stdout_thread = threading.Thread(target=drain, args=(proc.stdout, stdout_buf),
                                     daemon=True)
    stderr_thread = threading.Thread(target=drain, args=(proc.stderr, stderr_buf),
                                     daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    t0 = time.monotonic()
    next_progress = t0 + 1.0
    try:
        while proc.poll() is None:
            ctx.throw_if_cancelled()
            now = time.monotonic()
            if now - t0 > ctx.bash_timeout:
                _kill_tree(proc)
                stdout_thread.join(timeout=1)
                stderr_thread.join(timeout=1)
                stdout_b, stderr_b = bytes(stdout_buf), bytes(stderr_buf)
                raise ToolError(
                    f"command killed after {ctx.bash_timeout}s timeout. "
                    f"Partial stdout: {stdout_b.decode('utf-8', 'replace')[:500]!r}. "
                    "Avoid interactive/long-running commands; for big data prefer "
                    "streaming tools (awk, head) or write intermediate results to files.")
            if now >= next_progress:
                ctx.progress(phase="running", elapsed_s=round(now - t0, 1),
                             stdout_chars=len(stdout_buf),
                             stderr_chars=len(stderr_buf))
                next_progress = now + 1.0
            time.sleep(0.05)
    except Exception:
        if proc.poll() is None:
            _kill_tree(proc)
        raise
    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)
    stdout_b, stderr_b = bytes(stdout_buf), bytes(stderr_buf)

    stdout = stdout_b.decode("utf-8", "replace")
    stderr = stderr_b.decode("utf-8", "replace")
    parts = [f"exit code: {proc.returncode}"]
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    if not stdout and not stderr:
        parts.append("(no output)")
    return "\n".join(parts)
