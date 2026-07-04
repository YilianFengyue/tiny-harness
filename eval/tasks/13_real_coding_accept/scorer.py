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


def _tool_used(events: list[dict], name: str) -> bool:
    return any(e.get("type") == "tool_call" and e.get("name") == name for e in events)


def score(workdir: Path) -> tuple[bool, str]:
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=workdir,
                          capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        return False, f"pytest 未通过: {proc.stdout or proc.stderr}"
    acceptance = workdir / "ACCEPTANCE.md"
    if not acceptance.exists():
        return False, "ACCEPTANCE.md 不存在"
    events = _events(workdir)
    if not events:
        return False, "缺少 trajectory"
    run_start = next((e for e in events if e.get("type") == "run_start"), {})
    if not run_start.get("settings_sources"):
        return False, "run_start 缺少 settings_sources"
    features = run_start.get("features") or {}
    if features.get("coding_acceptance_trace") is not True:
        return False, "run_start 缺少 coding_acceptance_trace feature"
    for tool in ("glob_files", "grep", "read_file", "edit_file", "bash"):
        if not _tool_used(events, tool):
            return False, f"缺少 coding 流程工具: {tool}"
    overwritten = [
        e for e in events
        if e.get("type") == "tool_call"
        and e.get("name") == "write_file"
        and str((e.get("arguments") or {}).get("path", "")).startswith("src/invoice/")
    ]
    if overwritten:
        return False, "源码不应使用 write_file 整文件覆写"
    return True, "真实 coding 修复、settings/features 审计和过程约束均通过"
