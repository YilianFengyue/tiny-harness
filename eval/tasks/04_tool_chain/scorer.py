import re
from pathlib import Path


def score(workdir: Path) -> tuple[bool, str]:
    nums = workdir / "nums.txt"
    if not nums.exists():
        return False, "nums.txt 不存在（未按要求用 bash 生成）"
    lines = [l for l in nums.read_text(encoding="utf-8").split() if l.strip()]
    if len(lines) != 100 or lines[0] != "1" or lines[-1] != "100":
        return False, f"nums.txt 内容不对：{len(lines)} 行，首尾 {lines[:1]}..{lines[-1:]}"
    out = workdir / "result.txt"
    if not out.exists():
        return False, "result.txt 不存在"
    m = re.search(r"-?\d+(?:\.\d+)?", out.read_text(encoding="utf-8"))
    if not m:
        return False, "result.txt 中没有数字"
    got = float(m.group())
    return (abs(got - 338350) < 0.5), f"got={got} want=338350"
