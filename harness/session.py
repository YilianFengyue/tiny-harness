"""Session layer for interactive Agent use.

`loop.run_agent` is still the one-shot harness used by eval and scripts.
`AgentSession` keeps the conversation alive across many user submits, consumes
the same loop events, and accumulates usage/cost for a TUI/REPL surface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .cancel import CancellationToken
from .compact import compact_conversation
from .config import Config, load_pricing
from .context import ContextManager
from .loop import (
    TerminalState,
    _run_agent_events,
    build_initial_messages,
    build_resume_messages,
)
from .memory_extract import MemoryExtractionController
from .providers.base import Provider
from .session_store import latest_workspace_session, upsert_workspace_session
from .state import AppState, Store, build_app_state, create_store
from .telemetry import CostLedger, RunLogger, Usage, new_run_id, read_trajectory
from .tools import ToolContext, openai_tool_schemas

EventCallback = Callable[[dict], None]


@dataclass
class SessionTurn:
    run_id: str
    summary: dict
    events: list[dict] = field(default_factory=list)


@dataclass
class AgentSession:
    cfg: Config
    provider: Provider
    session_id: str = field(default_factory=new_run_id)
    messages: list[dict] = field(default_factory=list)
    last_run_id: str | None = None
    turns_submitted: int = 0
    usage_total: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    pricing_unknown: bool = False
    cancel_token: CancellationToken = field(default_factory=CancellationToken)
    permission_resolver: Callable | None = None
    permission_context: object | None = None
    app_state: Store[AppState] | None = None
    memory_controller: MemoryExtractionController | None = None
    context_manager: ContextManager | None = None

    def __post_init__(self) -> None:
        if self.memory_controller is None:
            self.memory_controller = MemoryExtractionController(self.cfg)
        if self.context_manager is None:
            self.context_manager = _make_context_manager(self.cfg)
        if self.app_state is None:
            self.app_state = create_store(build_app_state(
                self.cfg, permission_context=self.permission_context,
                memory_runtime=self.memory_controller.status(),
                context_runtime=self.context_status()))

    @classmethod
    def fresh(cls, cfg: Config, provider: Provider) -> "AgentSession":
        return cls(cfg=cfg, provider=provider,
                   messages=build_initial_messages("", cfg)[:1])

    @classmethod
    def from_run(cls, cfg: Config, provider: Provider, run_id: str) -> "AgentSession":
        events = read_trajectory(cfg.runs_dir, run_id)
        session_id = _session_id_from_events(events)
        session = cls(cfg=cfg, provider=provider,
                      session_id=session_id or new_run_id(),
                      messages=build_resume_messages(events))
        if session.memory_controller:
            session.memory_controller.state.last_processed_index = len(session.messages)
        session.last_run_id = run_id
        for e in events:
            if e["type"] == "run_end":
                u = e.get("usage_total", {})
                session.usage_total.add(_usage_from_dict(u))
                session.cost_usd += float(e.get("cost_usd", 0.0))
                session.pricing_unknown = session.pricing_unknown or bool(
                    e.get("pricing_unknown"))
        return session

    @classmethod
    def from_workspace_latest(cls, cfg: Config, provider: Provider) -> "AgentSession | None":
        stored = latest_workspace_session(cfg.workdir)
        if not stored or not stored.last_run_id:
            return None
        try:
            session = cls.from_run(cfg, provider, stored.last_run_id)
        except Exception:
            return None
        session.session_id = stored.session_id or session.session_id
        session.turns_submitted = stored.turns
        session.cost_usd = max(session.cost_usd, stored.cost_usd)
        return session

    def submit(self, user_text: str, on_event: EventCallback | None = None) -> SessionTurn:
        """Run one user message inside this persistent session."""
        self.turns_submitted += 1
        self.messages.append({"role": "user", "content": user_text})
        self._set_status("running")

        ledger = CostLedger(load_pricing())
        cm = self.context_manager or _make_context_manager(self.cfg)
        self.context_manager = cm
        tool_ctx = ToolContext(self.cfg.workdir, self.cfg.bash_timeout,
                               self.cfg.tool_output_limit)
        tool_ctx.runtime.config = self.cfg
        tool_ctx.runtime.permission_context = self.permission_context
        tool_ctx.runtime.permission_resolver = self.permission_resolver
        schemas = openai_tool_schemas()
        logger = RunLogger(self.cfg.runs_dir)
        self.last_run_id = logger.run_id

        events: list[dict] = []
        self.cancel_token.reset()
        gen = _run_agent_events(user_text, self.cfg, self.provider, schemas,
                                ledger, cm, tool_ctx, self.messages,
                                session_id=self.session_id,
                                cancel_token=self.cancel_token)
        terminal: TerminalState | None = None
        while True:
            try:
                event = next(gen)
            except StopIteration as done:
                terminal = done.value
                break
            event = dict(event)
            events.append(dict(event))
            if on_event:
                on_event(dict(event))
            type_ = event.pop("type")
            logger.emit(type_, **event)

        terminal = terminal or TerminalState("error", 0, "loop ended without terminal")
        if terminal.reason == "completed" and self.memory_controller is not None:
            self.memory_controller.extract(
                self.messages,
                emit=lambda event: _emit_memory_event(event, events, on_event, logger),
            )
        summary = logger.finish(terminal.reason, terminal.turns, ledger,
                                terminal.final_message)
        self.usage_total.add(_usage_from_dict(summary["usage_total"]))
        self.cost_usd += float(summary["cost_usd"])
        self.pricing_unknown = self.pricing_unknown or bool(summary["pricing_unknown"])
        self.permission_context = tool_ctx.runtime.permission_context
        self.persist_index()
        self._refresh_app_state(status=summary["reason"])
        return SessionTurn(run_id=logger.run_id, summary=summary, events=events)

    def extract_memory(self, *, force: bool = True,
                       on_event: EventCallback | None = None) -> list[dict]:
        if self.memory_controller is None:
            self.memory_controller = MemoryExtractionController(self.cfg)
        return self.memory_controller.extract(self.messages, emit=on_event, force=force)

    def set_memory_auto_extract(self, enabled: bool) -> None:
        if self.memory_controller is None:
            self.memory_controller = MemoryExtractionController(self.cfg)
        self.memory_controller.set_enabled(enabled)
        self._refresh_app_state()

    def context_status(self) -> dict[str, object]:
        cm = self.context_manager or _make_context_manager(self.cfg)
        self.context_manager = cm
        return cm.status(self.messages).as_dict()

    def compact_context(self, note: str = "", summarize: bool = True) -> dict | None:
        cm = self.context_manager or _make_context_manager(self.cfg)
        self.context_manager = cm
        if summarize:
            try:
                result = compact_conversation(
                    self.messages, self.provider, self.cfg,
                    trigger="manual", custom_instructions=note)
                if result.usage:
                    ledger = CostLedger(load_pricing())
                    cost = ledger.record(self.cfg.model, result.usage)  # type: ignore[arg-type]
                    self.usage_total.add(result.usage)  # type: ignore[arg-type]
                    self.cost_usd += cost
                    self.pricing_unknown = self.pricing_unknown or ledger.pricing_unknown
                cm.record_auto_compact_success()
                cm.last_prompt_tokens = result.post_tokens
                cm.last_compact_kind = "manual_summary_compact"
                self._refresh_app_state()
                return {
                    "kind": "manual_summary_compact",
                    **result.as_event(),
                    "summary": result.summary,
                    "status": self.context_status(),
                }
            except Exception as exc:
                edit = cm.manual_compact(self.messages, note=note)
                if edit is not None:
                    edit["summary_error"] = str(exc)
                self._refresh_app_state()
                return edit
        edit = cm.manual_compact(self.messages, note=note)
        self._refresh_app_state()
        return edit

    def cancel_current(self) -> None:
        self.cancel_token.cancel()
        self._set_status("cancelled")

    def trajectory_path(self) -> Path | None:
        if not self.last_run_id:
            return None
        return self.cfg.runs_dir / self.last_run_id / "trajectory.jsonl"

    def cumulative_summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "turns_submitted": self.turns_submitted,
            "last_run_id": self.last_run_id,
            "usage_total": self.usage_total.as_dict(),
            "cost_usd": round(self.cost_usd, 6),
            "pricing_unknown": self.pricing_unknown,
        }

    def persist_index(self, title: str | None = None) -> None:
        if not self.last_run_id:
            return
        upsert_workspace_session(
            self.cfg.workdir,
            session_id=self.session_id,
            title=title or _title_from_messages(self.messages),
            last_run_id=self.last_run_id,
            turns=self.turns_submitted,
            cost_usd=self.cost_usd,
        )

    def _set_status(self, status: str) -> None:
        self._refresh_app_state(status=status)

    def _refresh_app_state(self, status: str = "ready") -> None:
        if self.app_state is None:
            self.app_state = create_store(build_app_state(
                self.cfg, permission_context=self.permission_context, status=status,
                memory_runtime=(
                    self.memory_controller.status() if self.memory_controller else None),
                context_runtime=self.context_status()))
            return
        self.app_state.set_state(lambda _prev: build_app_state(
            self.cfg, permission_context=self.permission_context, status=status,
            memory_runtime=(
                self.memory_controller.status() if self.memory_controller else None),
            context_runtime=self.context_status()))


def _usage_from_dict(raw: dict) -> Usage:
    return Usage(
        prompt_tokens=int(raw.get("prompt_tokens", 0)),
        cached_tokens=int(raw.get("cached_tokens", 0)),
        completion_tokens=int(raw.get("completion_tokens", 0)),
        reasoning_tokens=int(raw.get("reasoning_tokens", 0)),
    )


def _session_id_from_events(events: list[dict]) -> str | None:
    for event in events:
        if event.get("type") == "run_start" and event.get("session_id"):
            return str(event["session_id"])
    return None


def _title_from_messages(messages: list[dict]) -> str:
    for message in messages:
        if message.get("role") == "user":
            content = str(message.get("content") or "").strip().replace("\n", " ")
            if content:
                return content[:80]
    return "Session"


def _emit_memory_event(event: dict, events: list[dict],
                       on_event: EventCallback | None, logger: RunLogger) -> None:
    payload = dict(event)
    events.append(dict(payload))
    if on_event:
        on_event(dict(payload))
    type_ = payload.pop("type")
    logger.emit(type_, **payload)


def _make_context_manager(cfg: Config) -> ContextManager:
    reserved = cfg.max_completion_tokens or 20_000
    return ContextManager(
        cfg.context_budget,
        cfg.context_keep_recent,
        cfg.context_hard_limit,
        cfg.tool_result_budget_chars,
        reserved,
    )
