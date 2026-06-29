"""题目原题：干净 CSV，第三列(amount)为数值。固定 seed 保证可复现。"""
import random
from pathlib import Path


def generate(workdir: Path) -> None:
    rng = random.Random(42)
    rows = ["id,name,amount,qty"]
    for i in range(1, 61):
        rows.append(f"{i},item-{i},{rng.uniform(10, 500):.2f},{rng.randint(1, 9)}")
    (workdir / "data.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
