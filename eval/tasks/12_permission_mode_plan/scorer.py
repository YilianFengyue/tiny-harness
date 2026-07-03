import json
from pathlib import Path


def _events(workdir: Path) -> list[dict]:
    runs = workdir.parent / "runs"
    paths = sorted(runs.glob("*/trajectory.jsonl"), key=lambda p: p.stat().st_mtime)
    if not paths:
        return []
    return [json.loads(line) for line in paths[-1].read_text(encoding="utf-8").splitlines()
            if line.strip()]


def score(workdir: Path) -> tuple[bool, str]:
    if (workdir / "plan.txt").exists():
        return False, "plan 模式下不应成功写出 plan.txt"
    events = _events(workdir)
    if not any(e.get("type") == "tool_permission"
               and e.get("name") in {"read_file", "grep"}
               and e.get("ok") is True for e in events):
        return False, "缺少只读工具允许事件"
    if not any(e.get("type") == "tool_permission"
               and e.get("name") == "write_file"
               and e.get("decision") == "deny"
               and e.get("reason_type") == "mode"
               and e.get("mode") == "plan" for e in events):
        return False, "缺少 plan 模式拒绝 write_file 的权限事件"
    return True, "plan 模式允许读取/搜索并拒绝写入"
