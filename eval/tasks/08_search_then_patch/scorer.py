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
    return [json.loads(line) for line in paths[-1].read_text(encoding="utf-8").splitlines()
            if line.strip()]


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
    target = workdir / "modules" / "module_17.py"
    report = workdir / "patch_report.txt"
    if not target.exists():
        return False, "目标模块不存在"
    if not report.exists():
        return False, "patch_report.txt 不存在"

    parser = (workdir / "parser.py").read_text(encoding="utf-8")
    if "def legacy_parse(payload: str) -> dict:" not in parser:
        return False, "parser.py 中 legacy_parse 定义不应被删除"

    bad = []
    for i in range(1, 31):
        path = workdir / "modules" / f"module_{i:02d}.py"
        text = path.read_text(encoding="utf-8")
        if f'SENTINEL = "module-{i:02d}-stable"' not in text:
            return False, f"module_{i:02d}.py sentinel 被改动"
        if "legacy_parse(payload)" in text:
            bad.append(path.name)
    if bad:
        return False, f"仍有 legacy_parse 调用: {bad}"

    proc = subprocess.run([sys.executable, "check.py"], cwd=workdir,
                          capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        return False, f"check.py 未通过: {proc.stderr or proc.stdout}"

    events = _events(workdir)
    if not (_tool_events(events, "grep") or _tool_events(events, "glob_files")):
        return False, "过程缺少 grep 或 glob_files"
    if not _tool_used_on(events, "edit_file", "modules/module_17.py"):
        return False, "过程缺少 edit_file(modules/module_17.py)"
    if _tool_used_on(events, "write_file", "modules/module_17.py"):
        return False, "不允许用 write_file 覆写目标模块"
    return True, "搜索定位并精确替换成功"
