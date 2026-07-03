"""Claude-Code-style Textual/Rich frontend for tiny-harness.

The UI is transcript-first: the conversation owns the screen, tool calls fold
into compact activity lines, and full lifecycle detail lives in /verbose.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import Config, PROJECT_ROOT
from .permissions import (
    PERMISSION_MODES,
    PermissionUpdate,
    apply_permission_update,
    format_permission_context,
    load_permission_context,
    permission_rule_value_from_string,
    permission_rule_value_to_string,
    persist_permission_updates,
    suggest_permission_rule_value,
    summarize_permission_update,
)
from .providers.base import Provider
from .session import AgentSession

try:  # pragma: no cover - UI smoke-tested with Textual's headless runner.
    from rich.console import Group
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from textual import events
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Vertical, VerticalScroll
    from textual.message import Message
    from textual.screen import ModalScreen
    from textual.widgets import Footer, OptionList, Static, TextArea
    from textual.widgets.option_list import Option

    _TUI_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    _TUI_IMPORT_ERROR = exc


@dataclass
class ToolActivity:
    """One visible tool row, updated as lifecycle events arrive."""

    call_id: str
    name: str = "tool"
    arguments: dict[str, Any] = field(default_factory=dict)
    phase: str = "queued"
    ok: bool | None = None
    read_only: bool = False
    concurrency_safe: bool = False
    destructive: bool = False
    permission: str | None = None
    permission_reason: str | None = None
    permission_mode: str | None = None
    permission_rule: str | None = None
    permission_update: str | None = None
    result_preview: str | None = None
    persisted_path: str | None = None
    context_note: str | None = None
    duration_ms: int | None = None
    error_kind: str | None = None
    audit: list[dict] = field(default_factory=list)


@dataclass
class UiRecord:
    role: str
    content: str = ""
    kind: str = "message"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class UiSession:
    id: int
    title: str
    agent: AgentSession
    records: list[UiRecord] = field(default_factory=list)
    audit_events: list[dict] = field(default_factory=list)
    activities: dict[str, ToolActivity] = field(default_factory=dict)
    running: bool = False
    status: str = "ready"
    assistant_buffer: str = ""
    verbose: bool = False


@dataclass
class PermissionPromptState:
    tool: str
    arguments: dict
    decision: Any
    result: "queue.Queue[str]"


COMMANDS: tuple[tuple[str, str], ...] = (
    ("/help", "show shortcuts and slash commands"),
    ("/verbose", "show exact lifecycle events for this session"),
    ("/sessions", "switch between chat sessions"),
    ("/new", "create a new session"),
    ("/rename <title>", "rename current session"),
    ("/drop", "close current session"),
    ("/clear", "clear visible transcript"),
    ("/cost", "show cumulative token and cost totals"),
    ("/trace", "show latest trajectory path and viewer URL"),
    ("/runs", "list recent run ids"),
    ("/permissions", "show active permission mode and rules"),
    ("/allow <rule> [session|local|project]", "add an allow rule"),
    ("/deny <rule> [session|local|project]", "add a deny rule"),
    ("/ask <rule> [session|local|project]", "add an ask rule"),
    ("/mode <mode> [session|local|project]", "set permission mode"),
    ("/theme", "choose a theme"),
    ("/exit", "quit"),
)

TOOL_EVENT_TYPES = {
    "tool_call",
    "tool_queued",
    "tool_validate",
    "tool_permission",
    "tool_permission_wait",
    "tool_permission_resolved",
    "tool_permission_update",
    "tool_start",
    "tool_progress",
    "tool_result_persisted",
    "tool_result",
    "tool_end",
    "tool_context_modified",
}


if _TUI_IMPORT_ERROR is None:

    class TranscriptBody(Static):
        """Stable transcript surface.

        Textual's mouse selection path can crash if individual message widgets
        are removed while a mouse event is being forwarded. Keeping one stable
        body widget and updating its renderable avoids that race.
        """

        ALLOW_SELECT = False

    class PromptInput(TextArea):
        """Bottom prompt: Enter sends, Ctrl+J keeps the drafting flow multiline."""

        class Submitted(Message):
            def __init__(self, text: str) -> None:
                super().__init__()
                self.text = text

        BINDINGS = [
            Binding("enter", "submit", "Send", show=False),
            Binding("ctrl+enter", "submit", "Send", show=False),
            Binding("ctrl+j", "newline", "Newline", show=False),
            Binding("shift+enter", "newline", "Newline", show=False),
            Binding("ctrl+u", "clear", "Clear", show=False),
        ]

        def action_submit(self) -> None:
            self.post_message(self.Submitted(self.text))
            self.text = ""

        def action_newline(self) -> None:
            self.insert("\n")

        def action_clear(self) -> None:
            self.text = ""

        async def _on_key(self, event: events.Key) -> None:
            if event.key == "enter":
                event.prevent_default()
                event.stop()
                self.action_submit()
            elif event.key in {"ctrl+j", "shift+enter"}:
                event.prevent_default()
                event.stop()
                self.action_newline()
            else:
                await super()._on_key(event)


    class PermissionPrompt(ModalScreen[str]):
        BINDINGS = [
            Binding("escape", "deny", "Deny", show=True),
            Binding("y", "allow_once", "Once", show=False),
            Binding("s", "allow_session", "Session", show=False),
            Binding("l", "allow_local", "Local", show=False),
            Binding("p", "allow_project", "Project", show=False),
            Binding("n", "deny", "Deny", show=False),
        ]

        def __init__(self, prompt: PermissionPromptState) -> None:
            super().__init__()
            self.prompt = prompt

        def compose(self) -> ComposeResult:
            rule = permission_rule_value_to_string(
                suggest_permission_rule_value(self.prompt.tool, self.prompt.arguments))
            body = Table.grid(padding=(0, 1))
            body.add_column(style="bold yellow", no_wrap=True)
            body.add_column()
            body.add_row("Tool", self.prompt.tool)
            body.add_row("Rule", rule)
            body.add_row("Mode", str(getattr(self.prompt.decision, "mode", "default")))
            body.add_row("Reason", str(getattr(self.prompt.decision, "message", "")))
            body.add_row("Input", _compact(_pretty_args(self.prompt.arguments), 900))
            yield Vertical(
                Static(Panel(body, title="Permission required", border_style="yellow")),
                OptionList(
                    Option("Allow once", id="y"),
                    Option("Allow this session", id="s"),
                    Option("Allow locally", id="l"),
                    Option("Allow in project", id="p"),
                    Option("Deny", id="n"),
                    id="permission-options",
                ),
                Static("Esc cancel", id="modal-hint"),
                id="permission-modal",
            )

        def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
            self.dismiss(str(event.option.id or "n"))

        def action_allow_once(self) -> None:
            self.dismiss("y")

        def action_allow_session(self) -> None:
            self.dismiss("s")

        def action_allow_local(self) -> None:
            self.dismiss("l")

        def action_allow_project(self) -> None:
            self.dismiss("p")

        def action_deny(self) -> None:
            self.dismiss("n")


    class HelpScreen(ModalScreen[None]):
        BINDINGS = [Binding("escape", "app.pop_screen", "Close", show=True)]

        def compose(self) -> ComposeResult:
            table = Table.grid(padding=(0, 2))
            table.add_column(style="bold cyan", no_wrap=True)
            table.add_column()
            for command, help_text in COMMANDS:
                table.add_row(command, help_text)
            table.add_row("", "")
            table.add_row("Enter", "send prompt")
            table.add_row("Ctrl+J", "insert newline")
            table.add_row("Ctrl+O", "toggle verbose lifecycle view")
            table.add_row("Ctrl+N", "new session")
            table.add_row("Ctrl+Up/Down", "switch session")
            table.add_row("Ctrl+C", "cancel running turn; quit when idle")
            yield Vertical(
                Static(Panel(table, title="TinyAgent commands", border_style="cyan")),
                id="help-modal",
            )


    class VerboseScreen(ModalScreen[None]):
        BINDINGS = [Binding("escape", "app.pop_screen", "Close", show=True)]

        def __init__(self, events: list[dict]) -> None:
            super().__init__()
            self.events = events

        def compose(self) -> ComposeResult:
            text = Text()
            if not self.events:
                text.append("No lifecycle events yet.", style="dim")
            for event in self.events[-160:]:
                text.append(_format_audit_event(event), style=_audit_style(event))
                text.append("\n")
            yield Vertical(
                Static(Panel(text, title="/verbose lifecycle", border_style="blue")),
                id="verbose-modal",
            )


    class SessionPicker(ModalScreen[int | None]):
        BINDINGS = [Binding("escape", "app.pop_screen", "Close", show=True)]

        def __init__(self, sessions: dict[int, UiSession], current_id: int) -> None:
            super().__init__()
            self.sessions = sessions
            self.current_id = current_id

        def compose(self) -> ComposeResult:
            options = []
            for sid, ui in sorted(self.sessions.items()):
                marker = "* " if sid == self.current_id else "  "
                running = " running" if ui.running else ""
                label = f"{marker}{sid}. {ui.title}  {ui.status}{running}"
                options.append(Option(label, id=str(sid)))
            yield Vertical(
                Static("Sessions", id="modal-title"),
                OptionList(*options, id="session-options"),
                id="session-modal",
            )

        def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
            raw = str(event.option.id or "")
            self.dismiss(int(raw) if raw.isdigit() else None)


    class CommandPalette(ModalScreen[str | None]):
        BINDINGS = [Binding("escape", "app.pop_screen", "Close", show=True)]

        def compose(self) -> ComposeResult:
            yield Vertical(
                Static("Commands", id="modal-title"),
                OptionList(*(Option(f"{cmd}  -  {desc}", id=cmd.split()[0])
                             for cmd, desc in COMMANDS),
                           id="command-options"),
                id="command-modal",
            )

        def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
            self.dismiss(str(event.option.id or ""))


    class ThemePicker(ModalScreen[str]):
        THEMES = ("textual-dark", "textual-light", "nord", "gruvbox", "dracula", "tokyo-night")
        BINDINGS = [Binding("escape", "app.pop_screen", "Close", show=True)]

        def compose(self) -> ComposeResult:
            yield Vertical(
                Static("Themes", id="modal-title"),
                OptionList(*(Option(name, id=name) for name in self.THEMES),
                           id="theme-options"),
                id="theme-modal",
            )

        def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
            self.dismiss(str(event.option.id or "textual-dark"))


    class TinyHarnessTextualApp(App[None]):
        """Terminal-first coding-agent REPL."""

        CSS = """
        Screen {
            background: #0d1117;
            color: #d8dee9;
        }

        #topbar {
            dock: top;
            height: 1;
            padding: 0 1;
            background: #0d1117;
            color: #c9d1d9;
        }

        #transcript {
            height: 1fr;
            padding: 1 3 0 3;
        }

        .message {
            margin: 0 0 1 0;
        }

        #statusline {
            height: 1;
            padding: 0 2;
            color: #8b949e;
            background: #161b22;
        }

        #prompt {
            height: 4;
            min-height: 3;
            border: tall #30363d;
            background: #0d1117;
        }

        #footerline {
            height: 1;
            padding: 0 2;
            color: #8b949e;
            background: #0d1117;
        }

        #permission-modal, #help-modal, #verbose-modal,
        #session-modal, #command-modal, #theme-modal {
            width: 84%;
            max-width: 112;
            height: auto;
            max-height: 90%;
            margin: 2 6;
            padding: 1 2;
            background: #161b22;
            border: round #30363d;
        }

        #permission-options, #session-options, #command-options, #theme-options {
            height: auto;
            margin-top: 1;
        }

        #modal-title {
            height: 1;
            color: #58a6ff;
        }

        #modal-hint {
            height: 1;
            color: #8b949e;
            margin-top: 1;
        }
        """

        BINDINGS = [
            Binding("ctrl+n", "new_session", "New", show=True),
            Binding("ctrl+up", "prev_session", "Prev", show=False),
            Binding("ctrl+down", "next_session", "Next", show=False),
            Binding("ctrl+d", "drop_session", "Drop", show=False),
            Binding("ctrl+o", "verbose", "Verbose", show=True),
            Binding("pageup", "transcript_page_up", "PgUp", show=False),
            Binding("pagedown", "transcript_page_down", "PgDn", show=False),
            Binding("ctrl+home", "transcript_home", "Top", show=False),
            Binding("ctrl+end", "transcript_end", "Bottom", show=False),
            Binding("ctrl+/", "help", "Help", show=True),
            Binding("ctrl+c", "cancel_or_quit", "Cancel", show=True),
        ]

        def __init__(self, cfg: Config, provider: Provider,
                     resume_run_id: str | None = None) -> None:
            super().__init__()
            self.cfg = cfg
            self.provider = provider
            self.resume_run_id = resume_run_id
            self.sessions: dict[int, UiSession] = {}
            self.current_id = 1
            self.next_id = 1

        def compose(self) -> ComposeResult:
            yield Static(id="topbar")
            with VerticalScroll(id="transcript"):
                yield TranscriptBody(id="transcript-body")
            yield Static(id="statusline")
            yield PromptInput(id="prompt")
            yield Static(id="footerline")
            yield Footer()

        def on_mount(self) -> None:
            self._new_session(resume_run_id=self.resume_run_id)
            self.set_interval(1.0, self._refresh_chrome)
            self._refresh_all()
            self.query_one("#prompt", PromptInput).focus()

        def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
            text = event.text.strip()
            if not text:
                return
            if text == "/":
                self.push_screen(CommandPalette(), self._run_palette_command)
                return
            if text.startswith("/"):
                self._handle_command(text)
                return
            self._submit_user_text(text)

        def action_new_session(self) -> None:
            self._new_session()
            self._refresh_all()

        def action_prev_session(self) -> None:
            self._switch_session(-1)

        def action_next_session(self) -> None:
            self._switch_session(1)

        def action_drop_session(self) -> None:
            self._drop_current_session()

        def action_verbose(self) -> None:
            self.push_screen(VerboseScreen(self._current().audit_events))

        def action_transcript_page_up(self) -> None:
            self.query_one("#transcript", VerticalScroll).scroll_page_up(animate=False)

        def action_transcript_page_down(self) -> None:
            self.query_one("#transcript", VerticalScroll).scroll_page_down(animate=False)

        def action_transcript_home(self) -> None:
            self.query_one("#transcript", VerticalScroll).scroll_home(animate=False)

        def action_transcript_end(self) -> None:
            self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)

        def action_help(self) -> None:
            self.push_screen(HelpScreen())

        def action_cancel_or_quit(self) -> None:
            ui = self._current()
            if ui.running:
                ui.agent.cancel_current()
                ui.status = "cancelling"
                self._add_system("Cancellation requested.")
            else:
                self.exit()

        def _new_session(self, resume_run_id: str | None = None) -> UiSession:
            sid = self.next_id
            self.next_id += 1
            agent = (AgentSession.from_run(self.cfg, self.provider, resume_run_id)
                     if resume_run_id else AgentSession.fresh(self.cfg, self.provider))
            ui = UiSession(sid, f"Session {sid}", agent)
            agent.permission_resolver = self._make_permission_resolver(sid)
            ui.records.append(UiRecord(
                "system",
                f"ready  model={self.cfg.model}  cwd={self.cfg.workdir}",
                "system",
            ))
            if resume_run_id:
                ui.records.append(UiRecord("system", f"resumed run {resume_run_id}", "system"))
            self.sessions[sid] = ui
            self.current_id = sid
            return ui

        def _current(self) -> UiSession:
            return self.sessions[self.current_id]

        def _switch_session(self, delta: int) -> None:
            ids = sorted(self.sessions)
            if not ids:
                self._new_session()
                return
            self.current_id = ids[(ids.index(self.current_id) + delta) % len(ids)]
            self._refresh_all()

        def _drop_current_session(self) -> None:
            ui = self._current()
            if ui.running:
                ui.agent.cancel_current()
            del self.sessions[self.current_id]
            if not self.sessions:
                self._new_session()
            else:
                self.current_id = sorted(self.sessions)[0]
            self._refresh_all()

        def _submit_user_text(self, text: str) -> None:
            ui = self._current()
            if ui.running:
                self._add_system("A turn is already running in this session.")
                return
            ui.running = True
            ui.status = "thinking"
            ui.assistant_buffer = ""
            ui.records.append(UiRecord("user", text))
            if ui.title.startswith("Session "):
                ui.title = _compact(text.replace("\n", " "), 30)
            self._refresh_all()
            thread = threading.Thread(target=self._run_turn, args=(ui.id, text), daemon=True)
            thread.start()

        def _run_turn(self, sid: int, text: str) -> None:
            ui = self.sessions[sid]
            try:
                turn = ui.agent.submit(
                    text, on_event=lambda event: self.call_from_thread(self._on_agent_event, sid, event))
                self.call_from_thread(self._on_turn_finished, sid, turn.summary)
            except Exception as exc:
                self.call_from_thread(self._on_turn_failed, sid, exc)

        def _on_turn_finished(self, sid: int, summary: dict) -> None:
            ui = self.sessions.get(sid)
            if not ui:
                return
            if ui.assistant_buffer.strip():
                ui.records.append(UiRecord("assistant", ui.assistant_buffer.strip()))
                ui.assistant_buffer = ""
            elif summary.get("final_message"):
                ui.records.append(UiRecord("assistant", str(summary["final_message"])))
            ui.running = False
            ui.status = f"done: {summary.get('reason')} turns={summary.get('turns')}"
            self._refresh_all()

        def _on_turn_failed(self, sid: int, exc: Exception) -> None:
            ui = self.sessions.get(sid)
            if not ui:
                return
            ui.running = False
            ui.status = "error"
            ui.records.append(UiRecord("system", f"ERROR: {exc}", "error"))
            self._refresh_all()

        def _on_agent_event(self, sid: int, event: dict) -> None:
            ui = self.sessions.get(sid)
            if not ui:
                return
            event = dict(event)
            ui.audit_events.append(event)
            kind = event.get("type", "")
            if kind == "assistant_delta":
                content = event.get("content") or ""
                if content:
                    ui.assistant_buffer += content
                    ui.status = "responding"
            elif kind == "llm_response":
                ui.status = f"model: {event.get('finish_reason')}"
            elif kind == "turn_start":
                ui.status = "thinking"
            elif kind in TOOL_EVENT_TYPES:
                activity = _fold_tool_event(ui.activities, event)
                if activity and not any(
                    r.kind == "tool_activity" and r.content == activity.call_id
                    for r in ui.records
                ):
                    ui.records.append(UiRecord(
                        "tool", activity.call_id, "tool_activity",
                        {"activity": activity},
                    ))
                if activity:
                    ui.status = _activity_status(activity)
            elif kind == "context_edit":
                ui.status = _compact(_format_context_edit(event), 90)
            elif kind == "stream_request_start":
                ui.status = f"request turn={event.get('turn')} model={event.get('model')}"
            if sid == self.current_id:
                self._refresh_all()

        def _make_permission_resolver(self, sid: int) -> Callable:
            def resolver(name: str, arguments: dict, decision: Any, cfg: Config, tool_ctx: Any) -> str:
                result: queue.Queue[str] = queue.Queue(maxsize=1)
                prompt = PermissionPromptState(name, arguments or {}, decision, result)
                self.call_from_thread(self._open_permission_prompt, sid, prompt)
                return result.get()
            return resolver

        def _open_permission_prompt(self, sid: int, prompt: PermissionPromptState) -> None:
            ui = self.sessions.get(sid)
            if ui:
                call_id = f"permission:{prompt.tool}:{len(ui.audit_events)}"
                activity = ToolActivity(call_id, prompt.tool, prompt.arguments, phase="permission")
                activity.permission = "waiting"
                activity.permission_reason = str(getattr(prompt.decision, "message", ""))
                activity.permission_mode = str(getattr(prompt.decision, "mode", "default"))
                ui.activities[call_id] = activity
                ui.records.append(UiRecord("tool", call_id, "tool_activity", {"activity": activity}))
                ui.status = f"waiting permission: {prompt.tool}"
                self._refresh_all()

            def done(choice: str | None) -> None:
                prompt.result.put(choice or "n")

            self.push_screen(PermissionPrompt(prompt), done)

        def _handle_command(self, command: str) -> None:
            name, _, rest = command.partition(" ")
            if name in {"/exit", "/quit", "/q"}:
                self.exit()
                return
            if name == "/help":
                self.push_screen(HelpScreen())
                return
            if name == "/verbose":
                self.action_verbose()
                return
            if name == "/theme":
                self.push_screen(ThemePicker(), self._apply_theme)
                return
            if name == "/new":
                self.action_new_session()
                return
            if name == "/sessions":
                self.push_screen(SessionPicker(self.sessions, self.current_id), self._switch_to_session)
                return
            if name == "/rename":
                self._current().title = rest.strip() or self._current().title
                self._refresh_all()
                return
            if name == "/drop":
                self._drop_current_session()
                return
            if name == "/clear":
                self._current().records.clear()
                self._refresh_all()
                return
            if name == "/cost":
                self._add_system(_format_cost(self._current().agent.cumulative_summary()))
                return
            if name == "/trace":
                self._add_system(_format_trace(self._current().agent.trajectory_path()))
                return
            if name == "/runs":
                self._add_system(_format_runs(self.cfg.runs_dir, self._current().agent.last_run_id))
                return
            if name == "/permissions":
                context = load_permission_context(self.cfg.workdir, _mode_override(self.cfg))
                self._add_system(format_permission_context(context))
                return
            if name in {"/allow", "/deny", "/ask"}:
                self._handle_rule_command(name[1:], rest)
                return
            if name == "/mode":
                self._handle_mode_command(rest)
                return
            self._add_system(f"Unknown command: {name}. Try /help.")

        def _run_palette_command(self, command: str | None) -> None:
            if command:
                self._handle_command(command)
            self.query_one("#prompt", PromptInput).focus()

        def _switch_to_session(self, sid: int | None) -> None:
            if sid in self.sessions:
                self.current_id = int(sid)
                self._refresh_all()
            self.query_one("#prompt", PromptInput).focus()

        def _handle_rule_command(self, behavior: str, text: str) -> None:
            raw, destination = _split_destination(text)
            if not raw:
                self._add_system(f"Usage: /{behavior} <rule> [session|local|project]")
                return
            try:
                update = PermissionUpdate(
                    "addRules", destination, behavior,
                    (permission_rule_value_from_string(raw),))
            except ValueError as exc:
                self._add_system(f"Invalid rule: {exc}")
                return
            current = self._current()
            base_context = current.agent.permission_context or load_permission_context(
                self.cfg.workdir, _mode_override(self.cfg))
            context = apply_permission_update(base_context, update)
            current.agent.permission_context = context
            current.agent.permission_resolver = self._make_permission_resolver(current.id)
            if destination in {"local", "project"}:
                persist_permission_updates(self.cfg.workdir, (update,))
            self._add_system(summarize_permission_update(update) + "\n\n" + format_permission_context(context))

        def _handle_mode_command(self, text: str) -> None:
            raw, destination = _split_destination(text)
            if raw not in PERMISSION_MODES:
                self._add_system("Usage: /mode <default|plan|acceptEdits|bypass|dontAsk> [session|local|project]")
                return
            update = PermissionUpdate("setMode", destination, mode=raw)
            current = self._current()
            base_context = current.agent.permission_context or load_permission_context(
                self.cfg.workdir, _mode_override(self.cfg))
            context = apply_permission_update(base_context, update)
            current.agent.permission_context = context
            self.cfg.permission_mode = raw
            if destination in {"local", "project"}:
                persist_permission_updates(self.cfg.workdir, (update,))
            self._add_system(summarize_permission_update(update) + "\n\n" + format_permission_context(context))

        def _apply_theme(self, theme: str | None) -> None:
            if theme:
                self.theme = theme

        def _add_system(self, text: str) -> None:
            self._current().records.append(UiRecord("system", text, "system"))
            self._refresh_all()

        def _refresh_all(self) -> None:
            self._refresh_chrome()
            self._refresh_messages()

        def _refresh_chrome(self) -> None:
            ui = self._current()
            cwd = _compact(str(self.cfg.workdir), 46).replace("\n", " ")
            top = (
                f"[bold cyan]TinyAgent[/]  [bold]ProMax[/]  "
                f"[green]{self.cfg.model}[/]  [yellow]{self.cfg.permission_mode}[/]  "
                f"[dim]{cwd}[/]  [bold]${ui.agent.cost_usd:.4f}[/]"
            )
            self.query_one("#topbar", Static).update(top)
            self.query_one("#statusline", Static).update(
                f"{_status_dot(ui)} {ui.status}  |  run={ui.agent.last_run_id or '-'}")
            self.query_one("#footerline", Static).update(
                "Enter send | Ctrl+J newline | Ctrl+O verbose | / commands | /permissions | /trace")

        def _refresh_messages(self) -> None:
            box = self.query_one("#transcript", VerticalScroll)
            was_at_end = box.is_vertical_scroll_end or box.max_scroll_y == 0
            ui = self._current()
            renderables = []
            for record in ui.records[-120:]:
                renderables.append(_render_record(record))
                renderables.append(Text(""))
            if ui.assistant_buffer.strip():
                renderables.append(_render_record(UiRecord("assistant", ui.assistant_buffer.strip())))
            body = self.query_one("#transcript-body", TranscriptBody)
            body.update(Group(*renderables) if renderables else Text(""))
            if was_at_end:
                box.scroll_end(animate=False)


def run_textual_tui(cfg: Config, provider: Provider,
                    resume_run_id: str | None = None) -> int:
    if _TUI_IMPORT_ERROR is not None:
        print("Textual TUI dependencies are missing.")
        print(f"Import error: {_TUI_IMPORT_ERROR}")
        print("Install with: pip install -r requirements.txt")
        return 2
    app = TinyHarnessTextualApp(cfg, provider, resume_run_id=resume_run_id)
    app.run()
    return 0


def _fold_tool_event(activities: dict[str, ToolActivity], event: dict) -> ToolActivity | None:
    call_id = str(event.get("tool_call_id") or event.get("id") or "")
    if not call_id:
        return None
    activity = activities.get(call_id)
    if activity is None:
        activity = ToolActivity(call_id=call_id, name=str(event.get("name") or "tool"))
        activities[call_id] = activity
    activity.audit.append(dict(event))
    t = event.get("type")
    if event.get("name"):
        activity.name = str(event["name"])
    if t == "tool_call":
        activity.phase = "called"
        activity.arguments = event.get("arguments") or {}
    elif t == "tool_queued":
        activity.phase = "queued"
        activity.arguments = event.get("arguments") or activity.arguments
    elif t == "tool_validate":
        activity.phase = "validated" if event.get("ok") else "validation error"
        activity.ok = bool(event.get("ok"))
        activity.read_only = bool(event.get("read_only"))
        activity.concurrency_safe = bool(event.get("concurrency_safe"))
        activity.destructive = bool(event.get("destructive"))
        if event.get("error"):
            activity.result_preview = str(event.get("error"))
    elif t == "tool_permission":
        activity.permission = str(event.get("decision") or ("allow" if event.get("ok") else "deny"))
        activity.permission_reason = event.get("reason")
        activity.permission_mode = event.get("mode")
        activity.permission_rule = event.get("rule")
    elif t == "tool_permission_wait":
        activity.phase = "waiting"
        activity.permission = "waiting"
        activity.permission_reason = event.get("reason")
        activity.permission_mode = event.get("mode")
        activity.permission_rule = event.get("rule")
    elif t == "tool_permission_resolved":
        activity.permission = str(event.get("decision") or ("allow" if event.get("ok") else "deny"))
        activity.permission_reason = event.get("reason")
        activity.permission_mode = event.get("mode")
        activity.permission_rule = event.get("rule")
        if not event.get("ok", True):
            activity.phase = "denied"
            activity.ok = False
    elif t == "tool_permission_update":
        activity.permission_update = str(event.get("summary") or "")
    elif t == "tool_start":
        activity.phase = "running"
    elif t == "tool_progress":
        activity.phase = str(event.get("phase") or "running")
    elif t == "tool_result_persisted":
        activity.persisted_path = str(event.get("path") or "")
    elif t == "tool_result":
        activity.phase = "done" if event.get("ok") else "error"
        activity.ok = bool(event.get("ok"))
        activity.result_preview = _compact(str(event.get("result") or ""), 1200)
        activity.duration_ms = int(event.get("duration_ms") or 0)
        activity.error_kind = event.get("error_kind")
        if event.get("persisted_path"):
            activity.persisted_path = str(event.get("persisted_path"))
    elif t == "tool_end":
        if activity.phase not in {"error", "denied"}:
            activity.phase = "done" if event.get("ok", True) else "error"
        activity.ok = bool(event.get("ok"))
        activity.duration_ms = int(event.get("duration_ms") or activity.duration_ms or 0)
    elif t == "tool_context_modified":
        activity.context_note = " ".join(
            f"{k}={v}" for k, v in event.items()
            if k not in {"type", "turn", "tool_call_id", "name"}
        )
    return activity


def _render_record(record: UiRecord):
    if record.kind == "tool_activity":
        activity = record.meta.get("activity")
        if isinstance(activity, ToolActivity):
            return _render_tool_activity(activity)
    if record.role == "user":
        text = Text()
        text.append("> ", style="bold cyan")
        text.append(record.content)
        return text
    if record.role == "assistant":
        return Markdown(record.content)
    if record.role == "system":
        text = Text()
        style = "red" if record.kind == "error" else "dim"
        text.append("! ", style=style)
        text.append(record.content, style=style)
        return text
    return Text(record.content)


def _render_tool_activity(activity: ToolActivity) -> Text:
    text = Text()
    status_style = _activity_style(activity)
    text.append("- ", style=status_style)
    text.append(_tool_title(activity), style=f"bold {status_style}")
    details = _tool_inline_details(activity)
    if details:
        text.append(f"  {details}", style="dim")
    if activity.permission in {"waiting", "ask", "deny"} and activity.permission_reason:
        text.append(f"\n  permission: {activity.permission_reason}", style="yellow")
    if activity.permission_update:
        text.append(f"\n  permission update: {activity.permission_update}", style="dim")
    if activity.persisted_path:
        text.append(f"\n  saved: {activity.persisted_path}", style="cyan")
    if activity.context_note:
        text.append(f"\n  contextModifier: {activity.context_note}", style="magenta")
    preview = _result_preview_for_display(activity)
    if preview:
        text.append(f"\n  -> {preview}", style="dim" if activity.ok else "red")
    return text


def _tool_title(activity: ToolActivity) -> str:
    args = activity.arguments or {}
    if activity.name == "bash":
        return f"Bash {_compact(str(args.get('command') or ''), 120).replace(chr(10), ' ')}".strip()
    if activity.name in {"read_file", "write_file", "edit_file", "glob_files", "grep"}:
        path = args.get("path") or args.get("pattern") or args.get("query") or ""
        verb = {
            "read_file": "Read",
            "write_file": "Write",
            "edit_file": "Edit",
            "glob_files": "Glob",
            "grep": "Grep",
        }[activity.name]
        return f"{verb} {path}".strip()
    return f"{activity.name} {_compact(_pretty_args(args), 100)}".strip()


def _tool_inline_details(activity: ToolActivity) -> str:
    bits = [activity.phase]
    flags = []
    if activity.read_only:
        flags.append("read")
    if activity.concurrency_safe:
        flags.append("parallel")
    if activity.destructive:
        flags.append("write")
    if flags:
        bits.append(",".join(flags))
    if activity.permission and activity.permission not in {"allow"}:
        bits.append(f"permission={activity.permission}")
    if activity.permission_rule:
        bits.append(f"rule={activity.permission_rule}")
    if activity.duration_ms is not None:
        bits.append(f"{activity.duration_ms}ms")
    if activity.error_kind:
        bits.append(f"kind={activity.error_kind}")
    return " | ".join(bit for bit in bits if bit)


def _result_preview_for_display(activity: ToolActivity) -> str:
    preview = (activity.result_preview or "").strip()
    if not preview:
        if activity.ok is True and activity.phase == "done":
            return "ok"
        return ""
    lines = [line.rstrip() for line in preview.splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) == 1:
        return _compact(lines[0], 180)
    return _compact(lines[0], 160) + f"  ... ({len(lines)} lines)"


def _activity_style(activity: ToolActivity) -> str:
    if activity.phase in {"error", "denied", "validation error"} or activity.ok is False:
        return "red"
    if activity.permission == "waiting" or activity.phase == "waiting":
        return "yellow"
    if activity.phase == "running":
        return "cyan"
    if activity.phase == "done" or activity.ok is True:
        return "green"
    return "magenta"


def _activity_status(activity: ToolActivity) -> str:
    if activity.permission == "waiting":
        return f"waiting permission: {activity.name}"
    return f"{activity.phase}: {activity.name}"


def _status_dot(ui: UiSession) -> str:
    if ui.running:
        return "*"
    if ui.status.startswith("error"):
        return "!"
    return "o"


def _format_audit_event(event: dict) -> str:
    t = event.get("type", "")
    name = event.get("name")
    call_id = event.get("tool_call_id")
    if t in TOOL_EVENT_TYPES:
        bits = [str(t)]
        if name:
            bits.append(str(name))
        if call_id:
            bits.append(str(call_id))
        for key in ("decision", "reason_type", "mode", "resolver", "ok", "persisted"):
            if key in event:
                bits.append(f"{key}={event[key]}")
        if event.get("reason"):
            bits.append(f"reason={event['reason']}")
        if event.get("summary"):
            bits.append(str(event["summary"]))
        if event.get("path"):
            bits.append(f"path={event['path']}")
        return " | ".join(bits)
    return f"{t} | {_compact(_pretty_args(event), 260)}"


def _audit_style(event: dict) -> str:
    if event.get("ok") is False or event.get("decision") == "deny":
        return "red"
    if event.get("type") == "tool_permission_wait":
        return "yellow"
    if event.get("type") == "tool_context_modified":
        return "magenta"
    if event.get("type") == "tool_result_persisted":
        return "cyan"
    return "dim"


def _format_context_edit(event: dict) -> str:
    details = " ".join(f"{k}={v}" for k, v in event.items() if k != "type")
    return f"context lifecycle: {details}"


def _format_cost(summary: dict) -> str:
    usage = summary["usage_total"]
    note = " (pricing unknown)" if summary["pricing_unknown"] else ""
    return (
        f"turns={summary['turns_submitted']} cost=${summary['cost_usd']:.4f}{note}\n"
        f"tokens input={usage['prompt_tokens']} cached={usage['cached_tokens']} "
        f"output={usage['completion_tokens']} reasoning={usage['reasoning_tokens']}\n"
        f"last_run_id={summary['last_run_id']}"
    )


def _format_trace(path: Path | None) -> str:
    if not path:
        return "No run yet."
    rel = _rel(path)
    return (
        f"trajectory: {rel}\n"
        "viewer: python main.py serve\n"
        f"http://localhost:8765/viewer/index.html?file=/{rel.as_posix()}"
    )


def _format_runs(runs_dir: Path, last_run_id: str | None) -> str:
    if not runs_dir.exists():
        return "No runs directory yet."
    runs = sorted((p for p in runs_dir.iterdir() if p.is_dir()),
                  key=lambda p: p.stat().st_mtime, reverse=True)[:12]
    if not runs:
        return "No runs yet."
    rows = []
    for path in runs:
        marker = "*" if path.name == last_run_id else "-"
        rows.append(f"{marker} {path.name}")
    return "\n".join(rows)


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


def _rel(path: Path) -> Path:
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def _pretty_args(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _compact(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = max(1, int(limit * 0.65))
    tail = max(1, limit - head - 18)
    return text[:head] + "\n...[snipped]...\n" + text[-tail:]
