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
from .config import Config, load_pricing
from .context import ContextManager
from .loop import (
    TerminalState,
    _run_agent_events,
    build_initial_messages,
    build_resume_messages,
)
from .providers.base import Provider
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

    def __post_init__(self) -> None:
        if self.app_state is None:
            self.app_state = create_store(build_app_state(
                self.cfg, permission_context=self.permission_context))

    @classmethod
    def fresh(cls, cfg: Config, provider: Provider) -> "AgentSession":
        return cls(cfg=cfg, provider=provider,
                   messages=build_initial_messages("", cfg)[:1])

    @classmethod
    def from_run(cls, cfg: Config, provider: Provider, run_id: str) -> "AgentSession":
        events = read_trajectory(cfg.runs_dir, run_id)
        session = cls(cfg=cfg, provider=provider,
                      messages=build_resume_messages(events))
        session.last_run_id = run_id
        for e in events:
            if e["type"] == "run_end":
                u = e.get("usage_total", {})
                session.usage_total.add(_usage_from_dict(u))
                session.cost_usd += float(e.get("cost_usd", 0.0))
                session.pricing_unknown = session.pricing_unknown or bool(
                    e.get("pricing_unknown"))
        return session

    def submit(self, user_text: str, on_event: EventCallback | None = None) -> SessionTurn:
        """Run one user message inside this persistent session."""
        self.turns_submitted += 1
        self.messages.append({"role": "user", "content": user_text})
        self._set_status("running")

        ledger = CostLedger(load_pricing())
        cm = ContextManager(self.cfg.context_budget, self.cfg.context_keep_recent,
                            self.cfg.context_hard_limit,
                            self.cfg.tool_result_budget_chars)
        tool_ctx = ToolContext(self.cfg.workdir, self.cfg.bash_timeout,
                               self.cfg.tool_output_limit)
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
        summary = logger.finish(terminal.reason, terminal.turns, ledger,
                                terminal.final_message)
        self.usage_total.add(_usage_from_dict(summary["usage_total"]))
        self.cost_usd += float(summary["cost_usd"])
        self.pricing_unknown = self.pricing_unknown or bool(summary["pricing_unknown"])
        self.permission_context = tool_ctx.runtime.permission_context
        self._refresh_app_state(status=summary["reason"])
        return SessionTurn(run_id=logger.run_id, summary=summary, events=events)

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

    def _set_status(self, status: str) -> None:
        self._refresh_app_state(status=status)

    def _refresh_app_state(self, status: str = "ready") -> None:
        if self.app_state is None:
            self.app_state = create_store(build_app_state(
                self.cfg, permission_context=self.permission_context, status=status))
            return
        self.app_state.set_state(lambda _prev: build_app_state(
            self.cfg, permission_context=self.permission_context, status=status))


def _usage_from_dict(raw: dict) -> Usage:
    return Usage(
        prompt_tokens=int(raw.get("prompt_tokens", 0)),
        cached_tokens=int(raw.get("cached_tokens", 0)),
        completion_tokens=int(raw.get("completion_tokens", 0)),
        reasoning_tokens=int(raw.get("reasoning_tokens", 0)),
    )
