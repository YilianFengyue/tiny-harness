"""脏数据任务：amount 列 10% 概率为无效值。"""
import random
from pathlib import Path


def generate(workdir: Path) -> None:
    rng = random.Random(99)
    rows = ["id,name,amount,qty"]
    dirty = ["", "N/A", "n/a", "abc", "-"]
    for i in range(1, 121):
        amount = rng.choice(dirty) if rng.random() < 0.10 else f"{rng.uniform(5, 800):.2f}"
        rows.append(f"{i},item-{i},{amount},{rng.randint(1, 5)}")
    (workdir / "data.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
