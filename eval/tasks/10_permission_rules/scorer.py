import json
import subprocess
import sys
from pathlib import Path


def _events(workdir: Path) -> list[dict]:
    runs = workdir.parent / "runs"
    paths = sorted(runs.glob("*/trajectory.jsonl"), key=lambda p: p.stat().st_mtime)
    if not paths:
        return []
    return [json.loads(line) for line in paths[-1].read_text(encoding="utf-8").splitlines()
            if line.strip()]


def score(workdir: Path) -> tuple[bool, str]:
    report = workdir / "permission_report.txt"
    if not report.exists():
        return False, "permission_report.txt 不存在"
    proc = subprocess.run([sys.executable, "check.py"], cwd=workdir,
                          capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        return False, f"check.py 未通过: {proc.stderr or proc.stdout}"
    events = _events(workdir)
    if not any(e.get("type") == "tool_permission"
               and e.get("decision") == "ask"
               and e.get("rule") == "Bash(python -c *)" for e in events):
        return False, "缺少 Bash(python -c *) ask 权限事件"
    if not any(e.get("type") == "tool_permission_resolved"
               and e.get("decision") == "deny"
               and e.get("reason_type") == "rule" for e in events):
        return False, "ask 未在非交互环境解析为 deny"
    if not any(e.get("type") == "tool_permission"
               and e.get("ok") is True
               and e.get("rule") == "Bash(python *)" for e in events):
        return False, "缺少 Bash(python *) allow 事件"
    if not any(e.get("name") == "edit_file" for e in events):
        return False, "缺少 edit_file 精确修改"
    return True, "ask 优先于 allow，允许规则和 edit_file 流程均通过"
