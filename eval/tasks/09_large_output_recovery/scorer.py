import json
import re
from collections import Counter
from pathlib import Path


FAIL_IDS = [137, 274, 411, 548, 685, 822, 959, 1096, 1233, 1370, 1507, 1644,
            1781, 1918, 2055]


def _events(workdir: Path) -> list[dict]:
    runs = workdir.parent / "runs"
    if not runs.exists():
        return []
    paths = sorted(runs.glob("*/trajectory.jsonl"), key=lambda p: p.stat().st_mtime)
    if not paths:
        return []
    return [json.loads(line) for line in paths[-1].read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _expected_codes() -> Counter:
    return Counter(f"E{i % 7}" for i in FAIL_IDS)


def score(workdir: Path) -> tuple[bool, str]:
    report = workdir / "report.txt"
    if not report.exists():
        return False, "report.txt 不存在"
    text = report.read_text(encoding="utf-8")

    ids = {int(m) for m in re.findall(r"\b0*(\d{3,4})\b", text)}
    missing = [i for i in FAIL_IDS if i not in ids]
    if missing:
        return False, f"report.txt 缺少失败 case id: {missing[:5]}"

    if str(len(FAIL_IDS)) not in text:
        return False, f"report.txt 未包含失败总数 {len(FAIL_IDS)}"

    expected = _expected_codes()
    for code, count in expected.items():
        if code not in text or str(count) not in text:
            return False, f"report.txt 缺少错误码分布 {code}={count}"

    events = _events(workdir)
    if not any(e.get("type") == "tool_result_persisted" for e in events):
        return False, "过程缺少 tool_result_persisted"
    if not any(e.get("name") == "bash" for e in events):
        return False, "过程缺少 bash 运行日志脚本"
    return True, "长输出落盘后成功恢复并写出报告"
