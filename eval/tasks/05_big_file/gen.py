"""大文件任务：30000 行，工具截断逼迫模型换流式方案。"""
import random
from pathlib import Path


def generate(workdir: Path) -> None:
    rng = random.Random(2026)
    rows = ["ts,sensor,reading,flag"]
    for i in range(30_000):
        rows.append(f"{1700000000 + i},s{rng.randint(1, 50)},{rng.uniform(-40, 120):.3f},{rng.randint(0, 1)}")
    (workdir / "data.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
