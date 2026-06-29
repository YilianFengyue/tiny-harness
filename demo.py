"""生成演示数据：在 ./workspace 放一个 data.csv（复用 eval 任务 01 的生成器）。"""
import importlib.util
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

root = Path(__file__).resolve().parent
ws = root / "workspace"
ws.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "gen", root / "eval" / "tasks" / "01_csv_mean" / "gen.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.generate(ws)

print(f"已生成 {ws / 'data.csv'}")
print('下一步: python main.py "读 data.csv，算第三列的均值，写到 mean.txt" --workdir ./workspace')
