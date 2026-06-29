import re
from pathlib import Path


def expected_mean(csv_path: Path, col: int = 2) -> float:
    values = []
    for line in csv_path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) > col:
            try:
                values.append(float(parts[col]))
            except ValueError:
                pass
    return sum(values) / len(values)


def extract_number(text: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(m.group()) if m else None


def score(workdir: Path) -> tuple[bool, str]:
    out = workdir / "mean.txt"
    if not out.exists():
        return False, "mean.txt 不存在"
    got = extract_number(out.read_text(encoding="utf-8"))
    if got is None:
        return False, f"mean.txt 中没有数字: {out.read_text(encoding='utf-8')[:80]!r}"
    want = expected_mean(workdir / "data.csv")
    ok = abs(got - want) <= max(abs(want) * 1e-4, 0.01)  # 容忍合理舍入
    return ok, f"got={got} want={want:.4f}"
