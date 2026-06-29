import re
from pathlib import Path


def expected(workdir: Path, name: str) -> float:
    values = [float(line.split(",")[1]) for line in
              (workdir / f"{name}.csv").read_text(encoding="utf-8").splitlines()[1:]]
    return sum(values) / len(values)


def score(workdir: Path) -> tuple[bool, str]:
    out = workdir / "means.txt"
    if not out.exists():
        return False, "means.txt 不存在"
    nums = [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", out.read_text(encoding="utf-8"))]
    if len(nums) != 3:
        return False, f"期望 3 个数字，得到 {len(nums)} 个"
    wants = [expected(workdir, n) for n in ("a", "b", "c")]
    for got, want, name in zip(nums, wants, "abc"):
        if abs(got - want) > max(abs(want) * 1e-3, 0.01):
            return False, f"{name}.csv: got={got} want={want:.4f}"
    return True, f"三个均值全部正确: {[f'{w:.3f}' for w in wants]}"
