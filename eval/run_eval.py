"""eval harness：Dataset → Solver(agent 子进程) → Scorer 的微缩实现。

  python eval/run_eval.py --runs 3                       # 全任务 x 默认模型
  python eval/run_eval.py --tasks 01,02 --skill csv-data-processing
  python eval/run_eval.py --matrix --runs 3              # {主模型,便宜模型} x {裸,skill} 2x2

每个任务一个目录 eval/tasks/<id>/：
  task.json   {"prompt": ..., "max_turns": ..., "extra_args": [...]}
  gen.py      def generate(workdir)  确定性生成输入文件（固定 seed）
  scorer.py   def score(workdir) -> (ok: bool, note: str)  程序化判分

agent 以子进程运行（环境隔离 + 熔断兜底），结果从 summary.json 读取——
eval 消费的就是 harness 自己的可观测性输出，闭环自洽。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Windows 控制台默认 GBK，会被 ▶/✅ 等字符炸死
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

EVAL_DIR = Path(__file__).resolve().parent
PROJECT = EVAL_DIR.parent
sys.path.insert(0, str(PROJECT))

from harness.config import load_dotenv  # noqa: E402


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def discover_tasks(only: list[str] | None) -> list[Path]:
    tasks = sorted(d for d in (EVAL_DIR / "tasks").iterdir() if (d / "task.json").exists())
    if only:
        tasks = [t for t in tasks if any(t.name.startswith(o) for o in only)]
    return tasks


def run_once(task_dir: Path, model: str, skill: str | None,
             out_dir: Path, run_index: int, max_cost: float) -> dict:
    spec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    base = out_dir / task_dir.name / (model + ("+skill" if skill else "")) / f"run{run_index}"
    workdir, runs_dir = base / "ws", base / "runs"
    workdir.mkdir(parents=True, exist_ok=True)

    gen = task_dir / "gen.py"
    if gen.exists():
        load_module(gen, f"gen_{task_dir.name}").generate(workdir)

    cmd = [sys.executable, str(PROJECT / "main.py"), spec["prompt"],
           "--workdir", str(workdir), "--runs-dir", str(runs_dir),
           "--model", model, "--yolo",
           "--max-turns", str(spec.get("max_turns", 15)),
           "--max-cost", str(max_cost)]
    if skill:
        cmd += ["--skill", skill]
    cmd += spec.get("extra_args", [])

    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              timeout=900, cwd=str(PROJECT))
        crashed, crash_note = False, ""
    except subprocess.TimeoutExpired:
        crashed, crash_note = True, "subprocess timeout (900s)"
        proc = None

    summary = {}
    run_dirs = sorted(runs_dir.iterdir()) if runs_dir.exists() else []
    if run_dirs and (run_dirs[-1] / "summary.json").exists():
        summary = json.loads((run_dirs[-1] / "summary.json").read_text(encoding="utf-8"))

    if crashed:
        ok, note = False, crash_note
    else:
        try:
            ok, note = load_module(task_dir / "scorer.py", f"scorer_{task_dir.name}").score(workdir)
        except Exception as e:
            ok, note = False, f"scorer error: {e}"
        if proc and proc.returncode != 0 and ok:
            note += f" (agent exited {proc.returncode}: {summary.get('reason')})"

    u = summary.get("usage_total", {})
    return {
        "task": task_dir.name, "model": model, "skill": skill or "-",
        "run": run_index, "success": ok, "note": note,
        "reason": summary.get("reason", "?"), "turns": summary.get("turns", 0),
        "cost_usd": summary.get("cost_usd", 0.0),
        "tokens_in": u.get("prompt_tokens", 0), "tokens_out": u.get("completion_tokens", 0),
        "reasoning_tokens": u.get("reasoning_tokens", 0),
        "cached_tokens": u.get("cached_tokens", 0),
        "wall_s": round(time.monotonic() - t0, 1),
        "run_id": summary.get("run_id"),
    }


def aggregate(rows: list[dict]) -> str:
    """生成 markdown 报告：按 (model, skill) 汇总 + 任务明细。"""
    lines = ["# Eval 报告", "",
             f"- 时间: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
             f"- 总运行数: {len(rows)}", ""]
    combos = sorted({(r["model"], r["skill"]) for r in rows})
    lines += ["## 汇总（模型 x skill）", "",
              "| model | skill | 成功率 | 平均轮数 | 平均成本$ | 平均reasoning tok | 缓存命中tok |",
              "|---|---|---|---|---|---|---|"]
    for model, skill in combos:
        sub = [r for r in rows if r["model"] == model and r["skill"] == skill]
        n, wins = len(sub), sum(r["success"] for r in sub)
        lines.append(
            f"| {model} | {skill} | {wins}/{n} | "
            f"{sum(r['turns'] for r in sub)/n:.1f} | "
            f"{sum(r['cost_usd'] for r in sub)/n:.4f} | "
            f"{sum(r['reasoning_tokens'] for r in sub)/n:.0f} | "
            f"{sum(r['cached_tokens'] for r in sub)/n:.0f} |")
    lines += ["", "## 明细", "",
              "| task | model | skill | run | 结果 | 终止 | 轮数 | 成本$ | 备注 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        mark = "✅" if r["success"] else "❌"
        lines.append(f"| {r['task']} | {r['model']} | {r['skill']} | {r['run']} | {mark} "
                     f"| {r['reason']} | {r['turns']} | {r['cost_usd']:.4f} "
                     f"| {r['note'][:80]} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", help="逗号分隔的任务前缀，如 01,02（默认全部）")
    p.add_argument("--runs", type=int, default=1, help="每个组合重复次数")
    p.add_argument("--model", default=None)
    p.add_argument("--skill", default=None)
    p.add_argument("--matrix", action="store_true",
                   help="2x2 实验：{TINY_HARNESS_MODEL, TINY_HARNESS_CHEAP_MODEL} x {裸, --skill}")
    p.add_argument("--max-cost", type=float, default=0.5, help="单次运行成本熔断")
    args = p.parse_args()

    load_dotenv()
    default_model = args.model or os.environ.get("TINY_HARNESS_MODEL", "gpt-5.5")
    if args.matrix:
        cheap = os.environ.get("TINY_HARNESS_CHEAP_MODEL", "gpt-5.5-mini")
        combos = [(m, s) for m in (default_model, cheap)
                  for s in (None, args.skill or "csv-data-processing")]
    else:
        combos = [(default_model, args.skill)]

    tasks = discover_tasks(args.tasks.split(",") if args.tasks else None)
    if not tasks:
        print("没有发现任务", file=sys.stderr)
        return 2

    out_dir = EVAL_DIR / "results" / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True)
    total = len(tasks) * len(combos) * args.runs
    print(f"任务 {len(tasks)} 个 x 组合 {len(combos)} x 重复 {args.runs} = {total} 次运行")
    print(f"结果目录: {out_dir}\n")

    rows = []
    for task_dir in tasks:
        for model, skill in combos:
            for i in range(1, args.runs + 1):
                label = f"{task_dir.name} [{model}{'+' + skill if skill else ''}] #{i}"
                print(f"▶ {label} ...", end=" ", flush=True)
                row = run_once(task_dir, model, skill, out_dir, i, args.max_cost)
                rows.append(row)
                print(("✅" if row["success"] else f"❌ {row['note'][:60]}")
                      + f"  ({row['turns']}轮 ${row['cost_usd']:.4f} {row['wall_s']}s)")
                (out_dir / "results.json").write_text(
                    json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    report = aggregate(rows)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"报告: {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
