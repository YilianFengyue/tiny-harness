"""错误恢复任务：prompt 说 data.csv，实际文件是 sales_data.csv。"""
import random
from pathlib import Path


def generate(workdir: Path) -> None:
    rng = random.Random(7)
    rows = ["order_id,customer,revenue,region"]
    for i in range(1, 81):
        rows.append(f"{1000 + i},cust-{i},{rng.uniform(50, 2000):.2f},R{rng.randint(1, 4)}")
    (workdir / "sales_data.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
