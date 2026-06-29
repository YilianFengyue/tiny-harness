import re
from pathlib import Path


def score(workdir: Path) -> tuple[bool, str]:
    out = workdir / "mean.txt"
    if not out.exists():
        return False, "mean.txt 不存在"
    m = re.search(r"-?\d+(?:\.\d+)?", out.read_text(encoding="utf-8"))
    if not m:
        return False, "mean.txt 中没有数字"
    got = float(m.group())

    valid, n_invalid = [], 0
    for line in (workdir / "data.csv").read_text(encoding="utf-8").splitlines()[1:]:
        cell = line.split(",")[2]
        try:
            valid.append(float(cell))
        except ValueError:
            n_invalid += 1
    want = sum(valid) / len(valid)            # 正确答案：剔除无效值
    want_as_zero = sum(valid) / (len(valid) + n_invalid)  # 常见错误：无效按 0 计

    if abs(got - want) <= max(abs(want) * 1e-4, 0.01):
        return True, f"got={got} want={want:.4f} (正确剔除了 {n_invalid} 个无效值)"
    if abs(got - want_as_zero) <= 0.01:
        return False, f"got={got}: 把无效值当 0 计入分母了（want={want:.4f}）"
    return False, f"got={got} want={want:.4f}"
