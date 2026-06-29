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
import shutil
import subprocess
import sys

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


def check_dangerous(arguments: dict) -> str | None:
    cmd = arguments.get("command", "")
    for pattern, why in DANGEROUS_PATTERNS:
        if re.search(pattern, cmd, re.IGNORECASE):
            return why
    return None


def _bash_executable() -> str | None:
    return shutil.which("bash")


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
)
def bash(ctx: ToolContext, command: str) -> str:
    if not command.strip():
        raise ToolError("empty command")
    bash_exe = _bash_executable()
    if bash_exe:
        argv: list[str] | str = [bash_exe, "-c", command]
        use_shell = False
    else:  # 无 bash 的 Windows 环境退化为 cmd；工具行为差异写进返回值提醒模型
        argv, use_shell = command, True

    popen_kwargs: dict = dict(
        cwd=str(ctx.workdir), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        shell=use_shell,
    )
    if sys.platform != "win32":
        popen_kwargs["start_new_session"] = True  # 自成进程组，超时可整组击杀

    proc = subprocess.Popen(argv, **popen_kwargs)
    try:
        stdout_b, stderr_b = proc.communicate(timeout=ctx.bash_timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        stdout_b, stderr_b = proc.communicate()
        raise ToolError(
            f"command killed after {ctx.bash_timeout}s timeout. "
            f"Partial stdout: {stdout_b.decode('utf-8', 'replace')[:500]!r}. "
            "Avoid interactive/long-running commands; for big data prefer "
            "streaming tools (awk, head) or write intermediate results to files.")

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
