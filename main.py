"""tiny-harness CLI

  python main.py "读 data.csv，算第三列的均值，写到 mean.txt" --workdir ./workspace
  python main.py --replay <run_id>          # 离线重放历史运行（不打 API、零成本）
  python main.py --resume <run_id> "继续：..."  # 从历史运行的消息现场继续
  python main.py serve                       # 起本地服务，在浏览器里看 trajectory
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows 控制台默认 GBK，中文输出会乱码
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from harness.config import Config, PROJECT_ROOT
from harness.loop import build_resume_messages, run_agent
from harness.providers import OpenAIChatProvider, ReplayProvider
from harness.telemetry import RunLogger, read_trajectory


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="tiny-harness: a from-scratch agent harness")
    p.add_argument("task", nargs="?", help="任务描述（--replay 时可省略）")
    p.add_argument("--model", default=None)
    p.add_argument("--workdir", default=None, help="agent 的沙箱工作目录（默认 ./workspace）")
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--max-cost", type=float, default=1.0, help="美元成本熔断线")
    p.add_argument("--reasoning-effort", default=None,
                   choices=["none", "low", "medium", "high", "xhigh"])
    p.add_argument("--max-completion-tokens", type=int, default=None)
    p.add_argument("--context-budget", type=int, default=240_000,
                   help="input token 预算，超过触发工具结果清理")
    p.add_argument("--keep-recent", type=int, default=3, help="清理时保留最近 N 条工具结果")
    p.add_argument("--skill", action="append", default=[],
                   help="注入领域 skill（名字或路径），可重复")
    p.add_argument("--replay", metavar="RUN_ID", help="离线重放该 run 的模型响应")
    p.add_argument("--resume", metavar="RUN_ID", help="从该 run 的消息现场继续")
    p.add_argument("--runs-dir", default=None)
    p.add_argument("--yolo", action="store_true", help="跳过危险命令确认（eval 自动化用）")
    return p.parse_args(argv)


def cmd_serve(argv: list[str]) -> None:
    import functools
    import http.server

    port = int(argv[0]) if argv else 8765
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(PROJECT_ROOT))
    print(f"viewer:  http://localhost:{port}/viewer/index.html")
    print(f"加载某次运行: http://localhost:{port}/viewer/index.html?file=/runs/<run_id>/trajectory.jsonl")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "serve":
        cmd_serve(argv[1:])
        return 0

    args = parse_args(argv)
    overrides: dict = {
        "max_turns": args.max_turns, "max_cost_usd": args.max_cost,
        "context_budget": args.context_budget, "context_keep_recent": args.keep_recent,
        "reasoning_effort": args.reasoning_effort,
        "max_completion_tokens": args.max_completion_tokens,
        "skills": args.skill, "yolo": args.yolo,
    }
    if args.model:
        overrides["model"] = args.model
    if args.workdir:
        overrides["workdir"] = Path(args.workdir)
    if args.runs_dir:
        overrides["runs_dir"] = Path(args.runs_dir)
    cfg = Config.from_env(**overrides)

    resume_messages = None
    if args.replay:
        events = read_trajectory(cfg.runs_dir, args.replay)
        provider: object = ReplayProvider(events)
        task = args.task or next(
            (e.get("task") for e in events if e["type"] == "run_start"), None)
    else:
        if not cfg.api_key:
            print("缺少 OPENAI_API_KEY：复制 .env.example 为 .env 并填写", file=sys.stderr)
            return 2
        if not args.task:
            print("需要任务描述，或使用 --replay <run_id>", file=sys.stderr)
            return 2
        provider = OpenAIChatProvider(
            model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url,
            max_retries=cfg.max_retries, reasoning_effort=cfg.reasoning_effort,
            max_completion_tokens=cfg.max_completion_tokens)
        task = args.task
        if args.resume:
            events = read_trajectory(cfg.runs_dir, args.resume)
            resume_messages = build_resume_messages(events)
            resume_messages.append({"role": "user", "content": task})

    logger = RunLogger(cfg.runs_dir)
    print(f"run_id: {logger.run_id}\nworkdir: {cfg.workdir}\nmodel: {cfg.model}"
          + (f"  (replay of {args.replay})" if args.replay else ""))

    summary = run_agent(task, cfg, provider, logger, resume_messages=resume_messages)  # type: ignore[arg-type]

    u = summary["usage_total"]
    print("\n" + "=" * 60)
    print(f"结束原因: {summary['reason']}    轮数: {summary['turns']}    "
          f"耗时: {summary['duration_s']}s")
    print(f"tokens: input={u['prompt_tokens']} (cached {u['cached_tokens']})  "
          f"output={u['completion_tokens']} (reasoning {u['reasoning_tokens']})")
    cost_note = "  [!] 未知模型价格，成本未计入" if summary["pricing_unknown"] else ""
    print(f"成本: ${summary['cost_usd']:.4f}{cost_note}")
    if summary.get("final_message"):
        print("-" * 60 + f"\n{summary['final_message']}")
    print("-" * 60)
    print(f"trajectory: runs/{summary['run_id']}/trajectory.jsonl")
    print(f"可视化: python main.py serve  → http://localhost:8765/viewer/index.html"
          f"?file=/runs/{summary['run_id']}/trajectory.jsonl")
    return 0 if summary["reason"] in ("completed",) else 1


if __name__ == "__main__":
    sys.exit(main())
