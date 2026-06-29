import re
from pathlib import Path


def score(workdir: Path) -> tuple[bool, str]:
    out = workdir / "mean.txt"
    if not out.exists():
        return False, "mean.txt 不存在（未从文件名错误中恢复？）"
    m = re.search(r"-?\d+(?:\.\d+)?", out.read_text(encoding="utf-8"))
    if not m:
        return False, "mean.txt 中没有数字"
    got = float(m.group())
    values = [float(line.split(",")[2]) for line in
              (workdir / "sales_data.csv").read_text(encoding="utf-8").splitlines()[1:]]
    want = sum(values) / len(values)
    ok = abs(got - want) <= max(abs(want) * 1e-4, 0.01)
    return ok, f"got={got} want={want:.4f}"
