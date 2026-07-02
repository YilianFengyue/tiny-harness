import json
import subprocess
import sys
from pathlib import Path


def _events(workdir: Path) -> list[dict]:
    runs = workdir.parent / "runs"
    if not runs.exists():
        return []
    paths = sorted(runs.glob("*/trajectory.jsonl"), key=lambda p: p.stat().st_mtime)
    if not paths:
        return []
    events: list[dict] = []
    for line in paths[-1].read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _tool_events(events: list[dict], name: str) -> list[dict]:
    return [e for e in events if e.get("name") == name]


def _tool_used_on(events: list[dict], name: str, path: str) -> bool:
    needle = path.replace("\\", "/")
    for e in _tool_events(events, name):
        args = e.get("arguments") or {}
        got = str(args.get("path", "")).replace("\\", "/")
        if got == needle or got.endswith("/" + needle):
            return True
    return False


def score(workdir: Path) -> tuple[bool, str]:
    app = workdir / "app.py"
    report = workdir / "fix_report.txt"
    if not app.exists():
        return False, "app.py 不存在"
    if not report.exists():
        return False, "fix_report.txt 不存在"

    text = app.read_text(encoding="utf-8")
    if "HEADER_SENTINEL = \"do-not-touch-header-v1\"" not in text:
        return False, "header sentinel 被改动"
    if "FOOTER_SENTINEL = \"do-not-touch-footer-v1\"" not in text:
        return False, "footer sentinel 被改动"
    if "def compute_score(passed: int, total: int) -> float:" not in text:
        return False, "compute_score 函数签名丢失"

    proc = subprocess.run([sys.executable, "check.py"], cwd=workdir,
                          capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        return False, f"check.py 未通过: {proc.stderr or proc.stdout}"

    events = _events(workdir)
    if not _tool_used_on(events, "read_file", "app.py"):
        return False, "过程缺少 read_file(app.py)"
    if not _tool_used_on(events, "edit_file", "app.py"):
        return False, "过程缺少 edit_file(app.py)"
    if _tool_used_on(events, "write_file", "app.py"):
        return False, "不允许用 write_file 覆写 app.py"
    if not _tool_events(events, "bash"):
        return False, "过程缺少 bash/test"
    return True, "除零修复通过，且过程使用 read_file -> edit_file -> bash"
