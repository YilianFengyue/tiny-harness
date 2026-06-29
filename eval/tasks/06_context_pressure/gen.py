"""上下文压力任务：三个中等 CSV，配合极小 context 预算运行。"""
import random
from pathlib import Path


def generate(workdir: Path) -> None:
    for name, seed in (("a", 1), ("b", 2), ("c", 3)):
        rng = random.Random(seed)
        rows = ["key,value"]
        for i in range(300):
            rows.append(f"{name}{i},{rng.uniform(0, 100):.3f}")
        (workdir / f"{name}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
