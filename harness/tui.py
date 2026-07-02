"""TinyAgent TUI v1: a dependency-free interactive session surface."""
from __future__ import annotations

import sys
from pathlib import Path

from .config import Config
from .providers.base import Provider
from .session import AgentSession


def run_tui(cfg: Config, provider: Provider, resume_run_id: str | None = None) -> int:
    session = (AgentSession.from_run(cfg, provider, resume_run_id)
               if resume_run_id else AgentSession.fresh(cfg, provider))
    _banner(cfg, session, resume_run_id)
    idle_interrupts = 0

    while True:
        try:
            text = input("\nyou> ").strip()
        except EOFError:
            print("\nbye")
            return 0
        except KeyboardInterrupt:
            idle_interrupts += 1
            if idle_interrupts >= 2:
                print("\nbye")
                return 130
            print("\n[interrupt] press Ctrl+C again to quit, or enter a new request.")
            continue

        idle_interrupts = 0
        if not text:
            continue
        if text.startswith("/"):
            if _handle_command(text, cfg, session):
                return 0
            continue

        print("agent> running")
        try:
            turn = session.submit(text, on_event=_print_event)
        except KeyboardInterrupt:
            session.cancel_current()
            print("\n[interrupt] current request interrupted")
            continue

        summary = turn.summary
        print()
        print(f"[done] reason={summary['reason']} turns={summary['turns']} "
              f"cost=${summary['cost_usd']:.4f} run_id={summary['run_id']}")
        streamed_text = any(e.get("type") == "assistant_delta" and e.get("content")
                            for e in turn.events)
        if summary.get("final_message") and not streamed_text:
            print(summary["final_message"])


def _banner(cfg: Config, session: AgentSession, resume_run_id: str | None) -> None:
    print("=" * 64)
    print("TinyAgent ProMax TUI v1")
    print(f"session_id: {session.session_id}")
    print(f"model:      {cfg.model}")
    print(f"workdir:    {cfg.workdir}")
    print(f"runs_dir:   {cfg.runs_dir}")
    if resume_run_id:
        print(f"resumed:    {resume_run_id}")
    print("-" * 64)
    print("Commands: /help  /cost  /trace  /runs  /exit")
    print("=" * 64)


def _handle_command(command: str, cfg: Config, session: AgentSession) -> bool:
    name, _, rest = command.partition(" ")
    if name in ("/exit", "/quit", "/q"):
        print("bye")
        return True
    if name == "/help":
        print("Commands:")
        print("  /cost   Show cumulative session token/cost totals")
        print("  /trace  Print latest trajectory path and viewer URL")
        print("  /runs   List recent run ids in this runs directory")
        print("  /exit   Quit the TUI session")
        return False
    if name == "/cost":
        s = session.cumulative_summary()
        u = s["usage_total"]
        note = " [pricing unknown]" if s["pricing_unknown"] else ""
        print(f"session turns={s['turns_submitted']} cost=${s['cost_usd']:.4f}{note}")
        print(f"tokens input={u['prompt_tokens']} cached={u['cached_tokens']} "
              f"output={u['completion_tokens']} reasoning={u['reasoning_tokens']}")
        print(f"last_run_id={s['last_run_id']}")
        return False
    if name == "/trace":
        path = session.trajectory_path()
        if not path:
            print("No run yet.")
            return False
        rel = _rel(path)
        print(f"trajectory: {rel}")
        print("viewer: python main.py serve")
        print(f"       http://localhost:8765/viewer/index.html?file=/{rel.as_posix()}")
        return False
    if name == "/runs":
        runs = sorted((p for p in cfg.runs_dir.iterdir() if p.is_dir()),
                      key=lambda p: p.stat().st_mtime, reverse=True)[:10]
        if not runs:
            print("No runs yet.")
            return False
        for p in runs:
            mark = " *" if p.name == session.last_run_id else "  "
            print(f"{mark} {p.name}")
        return False

    print(f"Unknown command: {name}. Try /help.")
    if rest:
        print(f"(ignored trailing text: {rest})")
    return False


def _print_event(event: dict) -> None:
    t = event.get("type")
    if t == "assistant_delta":
        content = event.get("content")
        if content:
            print(content, end="", flush=True)
    elif t == "stream_request_start":
        print(f"[stream] request turn={event['turn']} model={event['model']}")
    elif t == "turn_start":
        transition = event.get("transition") or "fresh"
        print(f"[turn {event['turn']}] start ({transition})")
    elif t == "tool_call":
        print(f"[tool] {event['name']} call_id={event['tool_call_id']}")
    elif t == "tool_start":
        print(f"[tool] start {event['name']} call_id={event['tool_call_id']}")
    elif t == "tool_progress":
        phase = event.get("phase", "progress")
        extras = " ".join(f"{k}={v}" for k, v in event.items()
                          if k not in {"type", "turn", "tool_call_id", "name", "phase"})
        suffix = f" {extras}" if extras else ""
        print(f"[tool] progress {event['name']} {phase}{suffix}")
    elif t == "tool_result":
        status = "ok" if event.get("ok") else "error"
        trunc = " truncated" if event.get("truncated") else ""
        print(f"[tool] {status}{trunc} {event['name']} "
              f"{event.get('duration_ms', 0)}ms")
    elif t == "tool_end":
        status = "ok" if event.get("ok") else "error"
        print(f"[tool] end {status} {event['name']} {event.get('duration_ms', 0)}ms")
    elif t == "transition":
        print(f"[transition] {event['reason']}")
    elif t == "context_edit":
        print(f"[context] cleared={event.get('cleared_messages')} "
              f"freed~={event.get('est_tokens_freed')} tokens")
    elif t == "retry":
        print(f"[retry] attempt={event['attempt']} status={event.get('status')} "
              f"sleep={event.get('sleep_s')}s")
    elif t == "error":
        print(f"[error] {event.get('error')}", file=sys.stderr)


def _rel(path: Path) -> Path:
    try:
        return path.relative_to(Path.cwd())
    except ValueError:
        return path
