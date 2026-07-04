"""TinyAgent TUI v1: a dependency-free interactive session surface."""
from __future__ import annotations

import sys
from pathlib import Path

from .config import Config
from .permissions import (
    PERMISSION_MODES,
    PermissionUpdate,
    apply_permission_update,
    format_permission_context,
    load_permission_context,
    permission_rule_value_from_string,
    persist_permission_updates,
    summarize_permission_update,
)
from .providers.base import Provider
from .session import AgentSession
from .session_store import list_workspace_sessions
from .lifecycle_hooks import format_hooks_summary, hooks_status_line
from .memory_view import (
    add_memory_from_text,
    extract_memory_for_session,
    forget_memory_from_text,
    format_memory_runtime_status,
    format_memory_summary,
    format_memory_tail,
    memory_status_line,
    rebuild_memory_index_for_cfg,
)
from .settings_view import format_features, format_settings_summary, settings_status_line
from .context_view import (
    context_status_line,
    format_compact_result,
    format_context_summary,
)


def run_tui(cfg: Config, provider: Provider, resume_run_id: str | None = None) -> int:
    restored_run_id: str | None = None
    if resume_run_id:
        session = AgentSession.from_run(cfg, provider, resume_run_id)
    else:
        session = AgentSession.from_workspace_latest(cfg, provider)
        if session is not None:
            restored_run_id = session.last_run_id
        else:
            session = AgentSession.fresh(cfg, provider)
    _banner(cfg, session, resume_run_id, restored_run_id)
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


def _banner(cfg: Config, session: AgentSession, resume_run_id: str | None,
            restored_run_id: str | None = None) -> None:
    print("=" * 64)
    print("TinyAgent ProMax TUI v1")
    print(f"session_id: {session.session_id}")
    print(f"model:      {cfg.model}")
    print(f"workdir:    {cfg.workdir}")
    print(f"runs_dir:   {cfg.runs_dir}")
    print(f"settings:   {settings_status_line(cfg)}")
    print(f"memory:     {memory_status_line(cfg)}")
    print(f"hooks:      {hooks_status_line(cfg)}")
    print(f"context:    {context_status_line(session.context_status())}")
    if resume_run_id:
        print(f"resumed:    {resume_run_id}")
    elif restored_run_id:
        print(f"restored:   {restored_run_id}")
    print("-" * 64)
    print("Commands: /help  /context  /compact  /memory  /hooks  /settings  /features  /permissions  /cost  /trace  /runs  /sessions  /exit")
    print("=" * 64)


def _handle_command(command: str, cfg: Config, session: AgentSession) -> bool:
    name, _, rest = command.partition(" ")
    if name in ("/exit", "/quit", "/q"):
        print("bye")
        return True
    if name == "/help":
        print("Commands:")
        print("  /cost   Show cumulative session token/cost totals")
        print("  /context  Show current context budget and warning state")
        print("  /compact [note]  Manually compact old tool results")
        print("  /trace  Print latest trajectory path and viewer URL")
        print("  /runs   List recent run ids in this runs directory")
        print("  /sessions   List persisted chat sessions for this workspace")
        print("  /memory [status|sources|prompt|tail|extract|on|off|list [type]|read <id>|add ...|forget <id>|rebuild]")
        print("          Add format: /memory add <type> <title> | <description> | <content>")
        print("  /hooks  Show lifecycle hook status")
        print("  /settings [sources|effective|trust]  Show config snapshot")
        print("  /features  Show active feature flags")
        print("  /permissions  Show active permission mode/rules")
        print("  /allow <rule> [local|project]  Add allow rule")
        print("  /deny <rule> [local|project]   Add deny rule")
        print("  /ask <rule> [local|project]    Add ask rule")
        print("  /mode <mode> [local|project]   Set permission mode")
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
    if name == "/context":
        print(format_context_summary(session.context_status()))
        return False
    if name == "/compact":
        edit = session.compact_context(rest.strip())
        print(format_compact_result(edit, session.context_status()))
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
    if name == "/sessions":
        print(_format_sessions(cfg, session))
        return False
    if name == "/memory":
        _handle_memory_command(rest, cfg, session)
        return False
    if name == "/hooks":
        print(format_hooks_summary(cfg))
        return False
    if name == "/settings":
        print(format_settings_summary(cfg, rest))
        return False
    if name == "/features":
        print(format_features(cfg))
        return False
    if name == "/permissions":
        context = load_permission_context(
            cfg.workdir, _mode_override(cfg),
            settings_snapshot=cfg.settings_snapshot)
        print(format_permission_context(context))
        return False
    if name in ("/allow", "/deny", "/ask"):
        _handle_rule_command(name[1:], rest, cfg, session)
        return False
    if name == "/mode":
        _handle_mode_command(rest, cfg, session)
        return False

    print(f"Unknown command: {name}. Try /help.")
    if rest:
        print(f"(ignored trailing text: {rest})")
    return False


def _handle_memory_command(text: str, cfg: Config, session: AgentSession) -> None:
    rest = text.strip()
    if not rest:
        print(format_memory_summary(cfg))
        return
    name, _, tail = rest.partition(" ")
    if name == "status":
        print(format_memory_runtime_status(cfg, session.memory_controller))
        return
    if name == "tail":
        print(format_memory_tail(cfg, tail.strip()))
        return
    if name == "extract":
        print("◈ Extracting..")
        print(extract_memory_for_session(session))
        return
    if name == "on":
        session.set_memory_auto_extract(True)
        print("[◈ Memory Auto Extract On]")
        return
    if name == "off":
        session.set_memory_auto_extract(False)
        print("[◈ Memory Auto Extract Off]")
        return
    if name == "add":
        print(add_memory_from_text(cfg, tail))
        return
    if name == "forget":
        print(forget_memory_from_text(cfg, tail))
        return
    if name == "rebuild":
        print(rebuild_memory_index_for_cfg(cfg))
        return
    if name == "list":
        print(format_memory_summary(cfg, f"list {tail}".strip()))
        return
    if name == "read":
        print(format_memory_summary(cfg, f"read {tail}".strip()))
        return
    if name in {"sources", "prompt"}:
        print(format_memory_summary(cfg, name))
        return
    print("Usage: /memory [status|sources|prompt|tail|extract|on|off|list [type]|read <id>|add ...|forget <id>|rebuild]")


def _format_sessions(cfg: Config, current: AgentSession) -> str:
    sessions = list_workspace_sessions(cfg.workdir)
    if not sessions:
        return "No persisted sessions for this workspace yet."
    lines = ["Workspace sessions:"]
    for item in sessions[:10]:
        mark = "*" if item.session_id == current.session_id else " "
        run = item.last_run_id or "-"
        lines.append(
            f"{mark} {item.title}  session={item.session_id}  run={run}  turns={item.turns}")
    return "\n".join(lines)


def _handle_rule_command(behavior: str, text: str, cfg: Config,
                         session: AgentSession) -> None:
    raw, destination = _split_destination(text)
    if not raw:
        print(f"Usage: /{behavior} <rule> [local|project]")
        return
    try:
        update = PermissionUpdate(
            "addRules", destination, behavior,
            (permission_rule_value_from_string(raw),))
    except ValueError as e:
        print(f"Invalid rule: {e}")
        return
    base_context = session.permission_context or load_permission_context(
        cfg.workdir, _mode_override(cfg),
        settings_snapshot=cfg.settings_snapshot)
    context = apply_permission_update(base_context, update)
    session.permission_context = context
    if destination in {"local", "project"}:
        persist_permission_updates(cfg.workdir, (update,))
    print(summarize_permission_update(update))
    print(format_permission_context(context))


def _handle_mode_command(text: str, cfg: Config, session: AgentSession) -> None:
    raw, destination = _split_destination(text)
    if raw not in PERMISSION_MODES:
        print("Usage: /mode <default|plan|acceptEdits|bypass|dontAsk> [local|project]")
        return
    update = PermissionUpdate("setMode", destination, mode=raw)
    base_context = session.permission_context or load_permission_context(
        cfg.workdir, _mode_override(cfg),
        settings_snapshot=cfg.settings_snapshot)
    context = apply_permission_update(base_context, update)
    session.permission_context = context
    cfg.permission_mode = raw
    if destination in {"local", "project"}:
        persist_permission_updates(cfg.workdir, (update,))
    print(summarize_permission_update(update))
    print(format_permission_context(context))


def _split_destination(text: str) -> tuple[str, str]:
    text = text.strip()
    if not text:
        return "", "local"
    parts = text.rsplit(maxsplit=1)
    if len(parts) == 2 and parts[1] in {"local", "project", "session"}:
        return parts[0].strip(), parts[1]
    return text, "local"


def _mode_override(cfg: Config) -> str | None:
    return None if cfg.permission_mode == "default" else cfg.permission_mode


def _print_event(event: dict) -> None:
    t = event.get("type")
    if t == "assistant_delta":
        content = event.get("content")
        if content:
            print(content, end="", flush=True)
    elif t == "stream_request_start":
        print(f"[stream] request turn={event['turn']} model={event['model']}")
    elif t == "memory_extract_start":
        print("◈ Extracting..")
    elif t == "memory_extract_saved":
        paths = ", ".join(event.get("paths") or [])
        print(f"[◈ Memory Saved] {event.get('count', 0)} {paths}")
    elif t == "memory_extract_skipped":
        print(f"◈ Memory skipped: {event.get('reason')}")
    elif t == "memory_extract_error":
        print(f"◈ Memory error: {event.get('error')}", file=sys.stderr)
    elif t == "memory_extract_trailing":
        print(f"◈ Extracting.. {event.get('status')}")
    elif t == "memory_load":
        print(f"[◈ Memory Loaded] {event.get('count', 0)}")
    elif t == "turn_start":
        transition = event.get("transition") or "fresh"
        print(f"[turn {event['turn']}] start ({transition})")
    elif t == "tool_call":
        print(f"[tool] {event['name']} call_id={event['tool_call_id']}")
    elif t == "tool_queued":
        print(f"[tool] queued {event['name']} call_id={event['tool_call_id']}")
    elif t == "tool_validate":
        if event.get("ok"):
            flags = []
            if event.get("read_only"):
                flags.append("read")
            if event.get("concurrency_safe"):
                flags.append("parallel")
            if event.get("destructive"):
                flags.append("write")
            suffix = f" ({','.join(flags)})" if flags else ""
            print(f"[tool] validate ok {event['name']}{suffix}")
        else:
            print(f"[tool] validate error {event['name']}: {event.get('error')}")
    elif t == "tool_permission":
        status = event.get("decision") or ("allow" if event.get("ok") else "deny")
        details = _permission_details(event)
        print(f"[tool] permission {status} {event['name']}{details}")
    elif t == "tool_permission_wait":
        details = _permission_details(event)
        print(f"[tool] permission waiting {event['name']}{details}")
    elif t == "tool_permission_resolved":
        status = event.get("decision") or ("allow" if event.get("ok") else "deny")
        resolver = event.get("resolver") or "unknown"
        details = _permission_details(event)
        print(f"[tool] permission resolved {status} by={resolver} {event['name']}{details}")
    elif t == "tool_permission_update":
        persisted = "persisted" if event.get("persisted") else "memory"
        print(f"[tool] permission update {persisted} {event.get('summary')}")
    elif t == "tool_input_updated":
        print(f"[tool] input updated by hook {event['name']} call_id={event['tool_call_id']}")
    elif t == "hook_start":
        print(f"[hook] start {event.get('hook_event')} source={event.get('source')}")
    elif t == "hook_end":
        status = "blocked" if event.get("blocked") else ("ok" if event.get("ok") else "error")
        reason = f" reason={event.get('reason')}" if event.get("reason") else ""
        print(f"[hook] end {event.get('hook_event')} {status}{reason}")
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
        persisted = f" saved={event.get('persisted_path')}" if event.get("persisted_path") else ""
        err = f" kind={event.get('error_kind')}" if event.get("error_kind") else ""
        print(f"[tool] {status}{trunc} {event['name']} "
              f"{event.get('duration_ms', 0)}ms{persisted}{err}")
    elif t == "tool_result_persisted":
        print(f"[tool] persisted {event['name']} -> {event.get('path')}")
    elif t == "tool_context_modified":
        details = " ".join(f"{k}={v}" for k, v in event.items()
                           if k not in {"type", "turn", "tool_call_id", "name"})
        print(f"[tool] context {event['name']} {details}")
    elif t == "tool_end":
        status = "ok" if event.get("ok") else "error"
        print(f"[tool] end {status} {event['name']} {event.get('duration_ms', 0)}ms")
    elif t == "transition":
        print(f"[transition] {event['reason']}")
    elif t == "context_edit":
        print(f"[context] cleared={event.get('cleared_messages')} "
              f"freed~={event.get('est_tokens_freed')} tokens")
    elif t == "context_status":
        print(f"[context] {context_status_line(event)}")
    elif t == "auto_compact_start":
        print(f"[context] compact start trigger={event.get('trigger')}")
    elif t == "auto_compact_saved":
        print(f"[context] summarized={event.get('messages_summarized')} "
              f"kept={event.get('messages_kept')} "
              f"tokens={event.get('pre_tokens')}->{event.get('post_tokens')}")
    elif t == "auto_compact_error":
        print(f"[context] compact error failures={event.get('failures')} "
              f"{event.get('error')}")
    elif t == "auto_compact_circuit_open":
        print(f"[context] compact circuit open failures={event.get('failures')}")
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


def _permission_details(event: dict) -> str:
    parts = []
    for key in ("reason_type", "mode", "rule", "source"):
        if event.get(key):
            parts.append(f"{key}={event[key]}")
    if event.get("safety_check"):
        parts.append("safety_check=true")
    if event.get("reason"):
        parts.append(f"reason={event['reason']}")
    return (" " + " ".join(parts)) if parts else ""
