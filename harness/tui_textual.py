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
from datetime import datetime
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
from .session_store import list_workspace_sessions
from .telemetry import read_trajectory
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
from .settings_view import (
    format_features,
    format_settings_summary,
    managed_permission_rules_only,
    settings_status_line,
)
from .context_view import (
    context_pill,
    context_status_line,
    format_compact_result,
    format_context_summary,
)
from .coordinator import coordinator_status

try:  # pragma: no cover - UI smoke-tested with Textual's headless runner.
    from rich.console import Group
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from textual import events
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
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
    agent_id: str | None = None
    agent_type: str | None = None
    agent_status: str | None = None
    agent_run_id: str | None = None
    agent_trajectory_path: str | None = None
    agent_background: bool = False
    agent_fork: bool = False
    audit: list[dict] = field(default_factory=list)


@dataclass
class BuildActivity:
    """One OpenCode-style running block for a user turn."""

    id: str
    model: str
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    phase: str = "thinking"
    status: str = "thinking"
    tool_ids: list[str] = field(default_factory=list)
    current_tool: str | None = None
    thinking: str = ""
    audit: list[dict] = field(default_factory=list)
    request_turn: int | None = None
    run_id: str | None = None

    @property
    def running(self) -> bool:
        return self.finished_at is None

    @property
    def elapsed_s(self) -> float:
        end = self.finished_at or time.monotonic()
        return max(0.0, end - self.started_at)


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
    current_build: BuildActivity | None = None
    builds: list[BuildActivity] = field(default_factory=list)
    build_seq: int = 0
    thinking_buffer: str = ""
    pending_permission: "PermissionPromptState | None" = None


@dataclass
class PermissionPromptState:
    tool: str
    arguments: dict
    decision: Any
    result: "queue.Queue[str]"
    cfg: Config | None = None
    selected: int = 0
    choice: str | None = None


COMMANDS: tuple[tuple[str, str], ...] = (
    ("/help", "show shortcuts and slash commands"),
    ("/agents", "show background sub-agent status"),
    ("/verbose", "show exact lifecycle events for this session"),
    ("/sessions", "switch between chat sessions"),
    ("/new", "create a new session"),
    ("/rename <title>", "rename current session"),
    ("/drop", "close current session"),
    ("/clear", "clear visible transcript"),
    ("/cost", "show cumulative token and cost totals"),
    ("/context", "show context budget and warning state"),
    ("/compact [note]", "manually compact old tool results"),
    ("/trace", "show latest trajectory path and viewer URL"),
    ("/build <n>", "open details for a build block"),
    ("/runs", "list recent run ids"),
    ("/memory [status|sources|prompt|tail|extract|on|off|list|read|add|forget|rebuild]", "manage persistent memory"),
    ("/hooks", "show lifecycle hook status"),
    ("/settings [sources|effective|trust]", "show config snapshot"),
    ("/features", "show active feature flags"),
    ("/permissions", "show active permission mode and rules"),
    ("/allow <rule> [session|local|project]", "add an allow rule"),
    ("/deny <rule> [session|local|project]", "add a deny rule"),
    ("/ask <rule> [session|local|project]", "add an ask rule"),
    ("/mode <mode> [session|local|project]", "set permission mode"),
    ("/auto_mode [on|off|full|status]", "toggle automatic approval mode"),
    ("/coordinator [on|off|status]", "toggle CH10 coordinator mode"),
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
    "tool_input_updated",
    "tool_start",
    "tool_progress",
    "tool_result_persisted",
    "tool_result",
    "tool_end",
    "tool_context_modified",
    "agent_start",
    "agent_progress",
    "agent_done",
    "agent_error",
    "agent_background_start",
    "agent_background_done",
}

MEMORY_EVENT_TYPES = {
    "memory_extract_start",
    "memory_extract_saved",
    "memory_extract_skipped",
    "memory_extract_error",
    "memory_extract_trailing",
}


if _TUI_IMPORT_ERROR is None:

    class TranscriptBody(Static):
        """Stable transcript surface.

        Textual's mouse selection path can crash if individual message widgets
        are removed while a mouse event is being forwarded. Keeping one stable
        body widget and updating its renderable avoids that race.
        """

        ALLOW_SELECT = False

        def on_click(self, event: events.Click) -> None:
            app = self.app
            if hasattr(app, "open_build_at_line"):
                app.open_build_at_line(int(event.y))


    class CommandMenuBody(Static):
        ALLOW_SELECT = False

    class PromptInput(TextArea):
        """Bottom prompt: Enter sends, Ctrl+J keeps the drafting flow multiline."""

        class Submitted(Message):
            def __init__(self, text: str) -> None:
                super().__init__()
                self.text = text

        class CommandNavigate(Message):
            def __init__(self, direction: int) -> None:
                super().__init__()
                self.direction = direction

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
            app = self.app
            if hasattr(app, "has_pending_permission") and app.has_pending_permission():
                if event.key in {"up", "shift+tab"}:
                    event.prevent_default()
                    event.stop()
                    app.action_permission_prev()
                    return
                if event.key in {"down", "tab"}:
                    event.prevent_default()
                    event.stop()
                    app.action_permission_next()
                    return
                if event.key == "enter":
                    event.prevent_default()
                    event.stop()
                    app.action_permission_submit()
                    return
                if event.key == "escape":
                    event.prevent_default()
                    event.stop()
                    app.action_permission_deny()
                    return
            if event.key == "enter":
                event.prevent_default()
                event.stop()
                self.action_submit()
            elif event.key in {"up", "down"} and self.text.lstrip().startswith("/"):
                event.prevent_default()
                event.stop()
                self.post_message(self.CommandNavigate(-1 if event.key == "up" else 1))
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
            body = _render_permission_prompt_body(self.prompt)
            yield Vertical(
                Static(Panel(body, title="Permission required", border_style="yellow")),
                OptionList(
                    Option(_permission_option("1. Allow once", "Approve this tool call only"), id="y"),
                    Option(_permission_option("2. Allow this session", "Remember the matching rule for this session"), id="s"),
                    Option(_permission_option("3. Allow locally", "Persist the matching rule to local settings"), id="l"),
                    Option(_permission_option("4. Allow in project", "Persist the matching rule to project settings"), id="p"),
                    Option(_permission_option("5. Deny", "Reject this request"), id="n"),
                    id="permission-options",
                ),
                Static("up/down select   enter submit   esc dismiss", id="modal-hint"),
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


    class BuildDetailScreen(ModalScreen[None]):
        BINDINGS = [
            Binding("escape", "app.pop_screen", "Parent", show=True),
            Binding("left", "prev_build", "Prev Build", show=True),
            Binding("right", "next_build", "Next Build", show=True),
            Binding("pageup", "page_up", "PgUp", show=False),
            Binding("pagedown", "page_down", "PgDn", show=False),
        ]

        def __init__(self, builds: list[BuildActivity], build_index: int,
                     activities: dict[str, ToolActivity]) -> None:
            super().__init__()
            self.builds = builds
            self.build_index = max(0, min(build_index, len(builds) - 1)) if builds else -1
            self.activities = activities
            self.tool_index = 0 if self.build and self.build.tool_ids else -1

        @property
        def build(self) -> BuildActivity | None:
            if self.build_index < 0 or self.build_index >= len(self.builds):
                return None
            return self.builds[self.build_index]

        def compose(self) -> ComposeResult:
            yield Vertical(
                Static(id="build-detail-title"),
                Horizontal(
                    OptionList(*_build_tool_options(self.build, self.activities),
                               id="build-tool-list"),
                    VerticalScroll(TranscriptBody(id="build-detail-body"),
                                   id="build-detail-scroll"),
                    id="build-detail-main",
                ),
                Static(id="build-detail-footer"),
                id="build-detail-modal",
            )

        def on_mount(self) -> None:
            self._refresh()

        def action_prev_build(self) -> None:
            self._switch_build(-1)

        def action_next_build(self) -> None:
            self._switch_build(1)

        def _switch_build(self, direction: int) -> None:
            if not self.builds:
                return
            self.build_index = (self.build_index + direction) % len(self.builds)
            build = self.build
            self.tool_index = 0 if build and build.tool_ids else -1
            self._refresh()

        def action_page_up(self) -> None:
            self.query_one("#build-detail-scroll", VerticalScroll).scroll_page_up(animate=False)

        def action_page_down(self) -> None:
            self.query_one("#build-detail-scroll", VerticalScroll).scroll_page_down(animate=False)

        def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
            if event.option_list.id != "build-tool-list":
                return
            raw = str(event.option.id or "")
            if raw.isdigit():
                self.tool_index = int(raw)
                self._refresh()

        def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
            if event.option_list.id != "build-tool-list":
                return
            raw = str(event.option.id or "")
            if raw.isdigit():
                self.tool_index = int(raw)
                self._refresh()

        def _refresh(self) -> None:
            build = self.build
            if build is None:
                empty = Text("No build selected.", style="dim")
                self.query_one("#build-detail-title", Static).update(empty)
                self.query_one("#build-detail-body", TranscriptBody).update(empty)
                self.query_one("#build-detail-footer", Static).update(empty)
                return
            options = _build_tool_options(build, self.activities)
            option_list = self.query_one("#build-tool-list", OptionList)
            option_list.clear_options()
            option_list.add_options(options)
            self.query_one("#build-detail-title", Static).update(
                _render_build_detail_title(build, self.build_index, len(self.builds)))
            self.query_one("#build-detail-body", TranscriptBody).update(
                _render_build_detail(build, self.activities, self.tool_index))
            self.query_one("#build-detail-footer", Static).update(
                _render_build_detail_footer(build, self.tool_index,
                                            self.build_index, len(self.builds)))


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

        #context-pill {
            height: 1;
            padding: 0 2;
            text-align: right;
            color: #c9d1d9;
            background: #0d1117;
        }

        #prompt {
            height: 4;
            min-height: 3;
            border-left: tall #58a6ff;
            border-top: none;
            border-right: none;
            border-bottom: none;
            background: #161616;
            padding: 0 1;
        }

        #command-menu {
            height: auto;
            max-height: 12;
            padding: 0 3;
            background: #1f1f1f;
            border-left: tall #58a6ff;
        }

        #command-menu.hidden {
            display: none;
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

        #build-detail-modal {
            width: 94%;
            height: 92%;
            margin: 1 3;
            background: #050505;
            border: tall #30363d;
        }

        #build-detail-title {
            height: 3;
            padding: 1 2 0 2;
            background: #050505;
        }

        #build-detail-main {
            height: 1fr;
            background: #050505;
        }

        #build-tool-list {
            width: 38;
            min-width: 28;
            height: 1fr;
            background: #0f0f0f;
            border-left: tall #58a6ff;
            border-right: solid #30363d;
        }

        #build-detail-scroll {
            height: 1fr;
            padding: 0 2;
            background: #050505;
        }

        #build-detail-footer {
            height: 3;
            padding: 1 2;
            background: #151515;
            color: #8b949e;
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
            self.command_selected = 0
            self.build_click_zones: list[tuple[int, int, str]] = []

        def compose(self) -> ComposeResult:
            yield Static(id="topbar")
            with VerticalScroll(id="transcript"):
                yield TranscriptBody(id="transcript-body")
            yield Static(id="context-pill")
            yield Static(id="statusline")
            yield CommandMenuBody(id="command-menu", classes="hidden")
            yield PromptInput(id="prompt")
            yield Static(id="footerline")
            yield Footer()

        def on_mount(self) -> None:
            self._new_session(
                resume_run_id=self.resume_run_id,
                restore_workspace_latest=self.resume_run_id is None,
            )
            self.set_interval(0.25, self._tick)
            self._refresh_all()
            self.query_one("#prompt", PromptInput).focus()

        def _tick(self) -> None:
            self._refresh_chrome()
            ui = self._current()
            if ui.running or (ui.current_build and ui.current_build.running):
                self._refresh_messages()

        def on_text_area_changed(self, event: PromptInput.Changed) -> None:
            if event.text_area.id == "prompt":
                self._refresh_command_menu(event.text_area.text)

        def on_prompt_input_command_navigate(self, event: PromptInput.CommandNavigate) -> None:
            matches = _command_matches(self.query_one("#prompt", PromptInput).text)
            if not matches:
                return
            self.command_selected = (self.command_selected + event.direction) % len(matches)
            self._refresh_command_menu(self.query_one("#prompt", PromptInput).text)

        def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
            if self.has_pending_permission():
                self.action_permission_submit()
                return
            text = event.text.strip()
            if not text:
                return
            if text.startswith("/"):
                selected = self._selected_slash_command(text)
                if selected and " " not in text and text != selected:
                    text = selected
                self._refresh_command_menu("")
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
            ui = self._current()
            build = _latest_build(ui)
            if build:
                self.push_screen(BuildDetailScreen(ui.builds, _build_index(ui.builds, build),
                                                   ui.activities))
            else:
                self.push_screen(VerboseScreen(ui.audit_events))

        def action_open_build(self, build_id: str = "") -> None:
            self._open_build_by_id(build_id)

        def open_build_at_line(self, y: int) -> None:
            try:
                scroll_y = int(self.query_one("#transcript", VerticalScroll).scroll_y)
            except Exception:
                scroll_y = 0
            line = y + scroll_y
            for start, end, build_id in self.build_click_zones:
                if start <= line <= end:
                    self._open_build_by_id(build_id)
                    return
            if len(self.build_click_zones) == 1:
                self._open_build_by_id(self.build_click_zones[0][2])
                return
            nearest = None
            nearest_distance = 999_999
            for start, end, build_id in self.build_click_zones:
                distance = min(abs(line - start), abs(line - end))
                if distance < nearest_distance:
                    nearest = build_id
                    nearest_distance = distance
            if nearest is not None and nearest_distance <= 8:
                self._open_build_by_id(nearest)

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

        def has_pending_permission(self) -> bool:
            return self._current().pending_permission is not None

        def action_permission_prev(self) -> None:
            self._move_permission_selection(-1)

        def action_permission_next(self) -> None:
            self._move_permission_selection(1)

        def action_permission_submit(self) -> None:
            ui = self._current()
            prompt = ui.pending_permission
            if not prompt:
                return
            choice = _permission_choices()[prompt.selected][0]
            self._resolve_pending_permission(ui, choice)

        def action_permission_deny(self) -> None:
            ui = self._current()
            if ui.pending_permission:
                self._resolve_pending_permission(ui, "n")

        def _move_permission_selection(self, delta: int) -> None:
            ui = self._current()
            prompt = ui.pending_permission
            if not prompt:
                return
            choices = _permission_choices()
            prompt.selected = (prompt.selected + delta) % len(choices)
            self._refresh_all()

        def _resolve_pending_permission(self, ui: UiSession, choice: str) -> None:
            prompt = ui.pending_permission
            if not prompt or prompt.choice is not None:
                return
            prompt.choice = choice
            ui.pending_permission = None
            ui.status = f"permission selected: {_permission_choice_label(choice)}"
            prompt.result.put(choice)
            self._refresh_all()

        def _new_session(self, resume_run_id: str | None = None,
                         restore_workspace_latest: bool = False) -> UiSession:
            sid = self.next_id
            self.next_id += 1
            restored_from_workspace = False
            if resume_run_id:
                agent = AgentSession.from_run(self.cfg, self.provider, resume_run_id)
            elif restore_workspace_latest:
                agent = AgentSession.from_workspace_latest(self.cfg, self.provider)
                restored_from_workspace = agent is not None
                if agent is None:
                    agent = AgentSession.fresh(self.cfg, self.provider)
            else:
                agent = AgentSession.fresh(self.cfg, self.provider)
            ui = UiSession(sid, _session_title_from_messages(agent.messages, sid), agent)
            agent.permission_resolver = self._make_permission_resolver(sid)
            ui.records.append(UiRecord("system", "", "logo"))
            ui.records.append(UiRecord(
                "system",
                f"ready  model={self.cfg.model}  cwd={self.cfg.workdir}",
                "system",
            ))
            if resume_run_id:
                ui.records.append(UiRecord("system", f"resumed run {resume_run_id}", "system"))
                ui.records.extend(_records_from_messages(agent.messages))
                _restore_builds_from_run(ui, self.cfg)
            elif restored_from_workspace:
                ui.records.append(UiRecord(
                    "system",
                    f"restored workspace session {agent.session_id} from run {agent.last_run_id}",
                    "system",
                ))
                ui.records.extend(_records_from_messages(agent.messages))
                _restore_builds_from_run(ui, self.cfg)
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
            ui.thinking_buffer = ""
            ui.records.append(UiRecord("user", text))
            if ui.title.startswith("Session "):
                ui.title = _compact(text.replace("\n", " "), 30)
            self._refresh_all()
            thread = threading.Thread(target=self._run_turn, args=(ui.id, text), daemon=True)
            thread.start()

        def _start_build(self, ui: UiSession) -> BuildActivity:
            ui.build_seq += 1
            build = BuildActivity(f"build-{ui.id}-{ui.build_seq}", model=self.cfg.model)
            ui.current_build = build
            ui.builds.append(build)
            return build

        def _append_build_record(self, ui: UiSession, build: BuildActivity) -> None:
            if not any(r.kind == "build_activity" and r.content == build.id for r in ui.records):
                ui.records.append(UiRecord("assistant", build.id, "build_activity", {"build": build}))

        def _ensure_tool_build(self, ui: UiSession) -> BuildActivity:
            build = ui.current_build
            if build is None or not build.running:
                build = self._start_build(ui)
            self._append_build_record(ui, build)
            return build

        def _finish_current_build(self, ui: UiSession, status: str = "done") -> None:
            build = ui.current_build
            if not build or not build.running:
                return
            if not build.tool_ids and not build.audit:
                ui.current_build = None
                return
            build.finished_at = time.monotonic()
            build.phase = "done"
            build.status = status
            ui.current_build = None

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
            self._flush_thinking_buffer(ui)
            if not self._flush_assistant_buffer(ui) and summary.get("final_message"):
                ui.records.append(UiRecord("assistant", str(summary["final_message"])))
            final_status = str(summary.get("reason") or "done")
            if ui.current_build:
                ui.current_build.finished_at = time.monotonic()
                ui.current_build.phase = "done" if summary.get("reason") == "completed" else "stopped"
                ui.current_build.status = final_status
                ui.current_build.run_id = str(summary.get("run_id") or "")
                ui.current_build = None
            for build in ui.builds:
                if not build.run_id:
                    build.run_id = str(summary.get("run_id") or "")
            ui.running = False
            ui.status = f"done: {summary.get('reason')} turns={summary.get('turns')}"
            self._refresh_all()

        def _on_turn_failed(self, sid: int, exc: Exception) -> None:
            ui = self.sessions.get(sid)
            if not ui:
                return
            ui.running = False
            ui.status = "error"
            if ui.current_build:
                ui.current_build.finished_at = time.monotonic()
                ui.current_build.phase = "error"
                ui.current_build.status = str(exc)
                ui.current_build = None
            ui.records.append(UiRecord("system", f"ERROR: {exc}", "error"))
            self._refresh_all()

        def _on_agent_event(self, sid: int, event: dict) -> None:
            ui = self.sessions.get(sid)
            if not ui:
                return
            event = dict(event)
            ui.audit_events.append(event)
            kind = event.get("type", "")
            build = ui.current_build
            if kind == "assistant_delta":
                reasoning = event.get("reasoning_content") or ""
                if reasoning:
                    ui.thinking_buffer += reasoning
                    if build and build.running:
                        build.thinking += reasoning
                    ui.status = "thinking"
                content = event.get("content") or ""
                if content:
                    self._finish_current_build(ui, "completed")
                    self._flush_thinking_buffer(ui)
                    ui.assistant_buffer += content
                    ui.status = "responding"
            elif kind == "llm_response":
                if build:
                    build.status = str(event.get("finish_reason") or "model response")
                    if event.get("tool_calls"):
                        build.phase = "tools"
                        build.status = "running tools"
                ui.status = f"model: {event.get('finish_reason')}"
            elif kind == "turn_start":
                if build:
                    build.phase = "thinking"
                    build.status = "thinking"
                ui.status = "thinking"
            elif kind in TOOL_EVENT_TYPES:
                self._flush_assistant_buffer(ui)
                self._flush_thinking_buffer(ui)
                build = self._ensure_tool_build(ui)
                build.audit.append(event)
                activity = _fold_tool_event(ui.activities, event)
                if activity:
                    if activity.call_id not in build.tool_ids:
                        build.tool_ids.append(activity.call_id)
                    build.current_tool = _tool_title(activity)
                    build.phase = "permission" if activity.permission == "waiting" else "tools"
                    build.status = _activity_status(activity)
                    ui.status = _activity_status(activity)
            elif kind == "context_edit":
                ui.status = _compact(_format_context_edit(event), 90)
            elif kind == "context_status":
                ui.status = context_status_line(event)
            elif kind == "auto_compact_start":
                ui.status = f"compacting context: {event.get('trigger')}"
            elif kind == "auto_compact_saved":
                ui.status = (
                    f"context summarized {event.get('messages_summarized', 0)} msgs"
                )
            elif kind == "auto_compact_error":
                ui.status = f"context compact error: {_compact(str(event.get('error') or ''), 60)}"
            elif kind == "auto_compact_circuit_open":
                ui.status = "context compact circuit open"
            elif kind == "hook_start":
                ui.status = f"hook {event.get('hook_event')} running"
            elif kind == "hook_end":
                status = "blocked" if event.get("blocked") else ("ok" if event.get("ok") else "error")
                ui.status = f"hook {event.get('hook_event')} {status}"
            elif kind == "stream_request_start":
                ui.status = f"request turn={event.get('turn')} model={event.get('model')}"
            elif kind == "memory_load":
                count = int(event.get("count") or 0)
                if count:
                    ui.status = f"◈ Memory Loaded: {count}"
            elif kind in MEMORY_EVENT_TYPES:
                ui.status = _format_memory_event_status(event)
            if sid == self.current_id:
                self._refresh_all()

        def _make_permission_resolver(self, sid: int) -> Callable:
            def resolver(name: str, arguments: dict, decision: Any, cfg: Config, tool_ctx: Any) -> str:
                result: queue.Queue[str] = queue.Queue(maxsize=1)
                prompt = PermissionPromptState(name, arguments or {}, decision, result, cfg)
                self.call_from_thread(self._open_permission_prompt, sid, prompt)
                return result.get()
            return resolver

        def _open_permission_prompt(self, sid: int, prompt: PermissionPromptState) -> None:
            ui = self.sessions.get(sid)
            if ui:
                self._flush_assistant_buffer(ui)
                self._flush_thinking_buffer(ui)
                build = self._ensure_tool_build(ui)
                build.current_tool = prompt.tool
                build.phase = "permission"
                build.status = f"waiting permission: {prompt.tool}"
                ui.pending_permission = prompt
                ui.records.append(UiRecord(
                    "system",
                    "",
                    "permission_request",
                    {"prompt": prompt},
                ))
                ui.status = f"waiting permission: {prompt.tool}"
                self._refresh_all()

        def _flush_assistant_buffer(self, ui: UiSession) -> bool:
            if not ui.assistant_buffer.strip():
                return False
            ui.records.append(UiRecord("assistant", ui.assistant_buffer.strip()))
            ui.assistant_buffer = ""
            return True

        def _flush_thinking_buffer(self, ui: UiSession) -> bool:
            if not ui.thinking_buffer.strip():
                return False
            ui.records.append(UiRecord("assistant", ui.thinking_buffer.strip(), "thinking"))
            ui.thinking_buffer = ""
            return True

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
            if name == "/agents":
                self._add_system(self._current().agent.agents_summary())
                return
            if name == "/theme":
                self.push_screen(ThemePicker(), self._apply_theme)
                return
            if name == "/new":
                self.action_new_session()
                return
            if name == "/sessions":
                history = _format_workspace_sessions(self.cfg, self._current().agent)
                if history:
                    self._add_system(history)
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
            if name == "/context":
                self._add_system(format_context_summary(self._current().agent.context_status()))
                return
            if name == "/compact":
                current = self._current().agent
                edit = current.compact_context(rest.strip())
                self._add_system(format_compact_result(edit, current.context_status()))
                return
            if name == "/trace":
                self._add_system(_format_trace(self._current().agent.trajectory_path()))
                return
            if name == "/build":
                self._open_build_command(rest)
                return
            if name == "/runs":
                self._add_system(_format_runs(self.cfg.runs_dir, self._current().agent.last_run_id))
                return
            if name == "/memory":
                self._add_system(_handle_memory_command(rest, self.cfg, self._current().agent))
                return
            if name == "/hooks":
                self._add_system(format_hooks_summary(self.cfg))
                return
            if name == "/settings":
                self._add_system(format_settings_summary(self.cfg, rest))
                return
            if name == "/features":
                self._add_system(format_features(self.cfg))
                return
            if name == "/permissions":
                context = load_permission_context(
                    self.cfg.workdir, _mode_override(self.cfg),
                    settings_snapshot=self.cfg.settings_snapshot)
                self._add_system(format_permission_context(context))
                return
            if name in {"/allow", "/deny", "/ask"}:
                self._handle_rule_command(name[1:], rest)
                return
            if name == "/mode":
                self._handle_mode_command(rest)
                return
            if name == "/auto_mode":
                self._handle_auto_mode_command(rest)
                return
            if name == "/coordinator":
                self._handle_coordinator_command(rest)
                return
            self._add_system(f"Unknown command: {name}. Try /help.")

        def _open_build_command(self, text: str) -> None:
            self._open_build_by_id(text.strip())

        def _open_build_by_id(self, raw: str = "") -> None:
            ui = self._current()
            build: BuildActivity | None = None
            if raw:
                for candidate in ui.builds:
                    if raw in {candidate.id, _build_number(candidate)}:
                        build = candidate
                        break
            else:
                build = _latest_build(ui)
            if not build:
                self._add_system("No matching build block. Try /build 1.")
                return
            self.push_screen(BuildDetailScreen(ui.builds, _build_index(ui.builds, build),
                                               ui.activities))

        def _run_palette_command(self, command: str | None) -> None:
            if command:
                self._handle_command(command)
            self.query_one("#prompt", PromptInput).focus()

        def _selected_slash_command(self, text: str) -> str | None:
            matches = _command_matches(text)
            if not matches:
                return None
            self.command_selected %= len(matches)
            return matches[self.command_selected][0].split()[0]

        def _refresh_command_menu(self, text: str) -> None:
            menu = self.query_one("#command-menu", CommandMenuBody)
            matches = _command_matches(text)
            if not text.lstrip().startswith("/") or not matches:
                menu.set_class(True, "hidden")
                self.command_selected = 0
                menu.update("")
                return
            self.command_selected %= len(matches)
            body = Table.grid(expand=True)
            body.add_column(no_wrap=True, width=22)
            body.add_column()
            for index, (command, desc) in enumerate(matches[:10]):
                cmd = command.split()[0]
                style = "black on #ffb482" if index == self.command_selected else "white"
                desc_style = "black on #ffb482" if index == self.command_selected else "dim"
                body.add_row(Text(cmd, style=style), Text(desc, style=desc_style))
            menu.update(body)
            menu.set_class(False, "hidden")

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
                self.cfg.workdir, _mode_override(self.cfg),
                settings_snapshot=self.cfg.settings_snapshot)
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
                self.cfg.workdir, _mode_override(self.cfg),
                settings_snapshot=self.cfg.settings_snapshot)
            context = apply_permission_update(base_context, update)
            current.agent.permission_context = context
            self.cfg.permission_mode = raw
            if destination in {"local", "project"}:
                persist_permission_updates(self.cfg.workdir, (update,))
            self._add_system(summarize_permission_update(update) + "\n\n" + format_permission_context(context))

        def _handle_auto_mode_command(self, text: str) -> None:
            raw = (text.strip() or "status").lower()
            if raw in {"status", "?"}:
                self._add_system(_format_auto_mode_status(self.cfg))
                return
            if raw in {"on", "true", "1"}:
                mode, yolo = "bypass", False
            elif raw == "full":
                mode, yolo = "bypass", True
            elif raw in {"off", "false", "0"}:
                mode, yolo = "default", False
            else:
                self._add_system("Usage: /auto_mode [on|off|full|status]")
                return
            update = PermissionUpdate("setMode", "session", mode=mode)
            current = self._current()
            base_context = current.agent.permission_context or load_permission_context(
                self.cfg.workdir, _mode_override(self.cfg),
                settings_snapshot=self.cfg.settings_snapshot)
            context = apply_permission_update(base_context, update)
            current.agent.permission_context = context
            self.cfg.permission_mode = mode
            self.cfg.yolo = yolo
            self._add_system(_format_auto_mode_status(self.cfg) + "\n\n" + format_permission_context(context))

        def _handle_coordinator_command(self, text: str) -> None:
            raw = (text.strip() or "status").lower()
            current = self._current()
            if raw in {"status", "?"}:
                self._add_system(coordinator_status(self.cfg, current.agent.session_id))
                return
            if raw in {"on", "true", "1"}:
                self.cfg.coordinator_mode = True
            elif raw in {"off", "false", "0"}:
                self.cfg.coordinator_mode = False
            else:
                self._add_system("Usage: /coordinator [on|off|status]")
                return
            current.agent.refresh_system_prompt()
            state = "enabled" if self.cfg.coordinator_mode else "disabled"
            self._add_system(
                f"Coordinator mode {state}.\n\n"
                + coordinator_status(self.cfg, current.agent.session_id)
            )

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
            self.query_one("#context-pill", Static).update(
                context_pill(ui.agent.context_status()))
            self.query_one("#footerline", Static).update(
                _footer_line(ui, self.cfg))

        def _refresh_messages(self) -> None:
            box = self.query_one("#transcript", VerticalScroll)
            was_at_end = box.is_vertical_scroll_end or box.max_scroll_y == 0
            ui = self._current()
            renderables = []
            self.build_click_zones = []
            line = 0
            for record in ui.records[-120:]:
                renderable = _render_record(record)
                if record.kind == "build_activity":
                    build = record.meta.get("build")
                    if isinstance(build, BuildActivity):
                        line_count = _renderable_line_count(renderable)
                        self.build_click_zones.append((max(0, line - 1), line + line_count + 2, build.id))
                renderables.append(renderable)
                renderables.append(Text(""))
                line += _renderable_line_count(renderable) + 1
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


def _records_from_messages(messages: list[dict]) -> list[UiRecord]:
    records: list[UiRecord] = []
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        records.append(UiRecord(str(role), content))
    return records[-80:]


def _restore_builds_from_run(ui: UiSession, cfg: Config) -> None:
    run_id = ui.agent.last_run_id
    if not run_id:
        return
    try:
        events = read_trajectory(cfg.runs_dir, run_id)
    except (OSError, json.JSONDecodeError):
        return
    _restore_builds_from_events(ui, events, cfg.model)


def _restore_builds_from_events(ui: UiSession, events: list[dict],
                                default_model: str) -> None:
    if not events:
        return

    run_id = _trajectory_run_id(events)
    model = _trajectory_model(events) or default_model
    run_status = _trajectory_run_status(events) or "restored"
    final_ts = _trajectory_final_ts(events)
    restored: dict[int, BuildActivity] = {}

    def build_for(event: dict) -> BuildActivity:
        turn = _event_turn(event)
        build = restored.get(turn)
        if build is None:
            ui.build_seq += 1
            build = BuildActivity(f"build-{ui.id}-{ui.build_seq}", model=model)
            build.request_turn = turn
            build.run_id = run_id
            build.started_at = _event_ts(event) or final_ts or build.started_at
            restored[turn] = build
        return build

    for event in events:
        event = dict(event)
        kind = event.get("type")
        if kind == "assistant_delta":
            reasoning = str(event.get("reasoning_content") or "")
            if reasoning:
                build_for(event).thinking += reasoning
        elif kind == "llm_response":
            if event.get("tool_calls"):
                build = build_for(event)
                build.phase = "tools"
                build.status = "running tools"
            elif event.get("finish_reason") and _event_turn(event) in restored:
                restored[_event_turn(event)].status = str(event.get("finish_reason"))
        elif kind in TOOL_EVENT_TYPES:
            build = build_for(event)
            build.audit.append(event)
            activity = _fold_tool_event(ui.activities, event)
            if activity:
                if activity.call_id not in build.tool_ids:
                    build.tool_ids.append(activity.call_id)
                build.current_tool = _tool_title(activity)
                build.phase = "permission" if activity.permission == "waiting" else "tools"
                build.status = _activity_status(activity)

    for build in restored.values():
        if not _build_has_details(build):
            continue
        build.finished_at = final_ts or build.started_at
        build.phase = "done" if run_status == "completed" else "stopped"
        build.status = run_status
        ui.builds.append(build)
        ui.records.append(UiRecord("assistant", build.id, "build_activity", {"build": build}))
    ui.current_build = None
    ui.audit_events.extend(dict(event) for event in events)


def _trajectory_run_id(events: list[dict]) -> str | None:
    for event in events:
        if event.get("type") == "run_start" and event.get("run_id"):
            return str(event["run_id"])
    for event in events:
        if event.get("run_id"):
            return str(event["run_id"])
    return None


def _trajectory_model(events: list[dict]) -> str | None:
    for event in events:
        if event.get("type") == "run_start" and event.get("model"):
            return str(event["model"])
    for event in events:
        if event.get("model"):
            return str(event["model"])
    return None


def _trajectory_run_status(events: list[dict]) -> str | None:
    for event in reversed(events):
        if event.get("type") == "run_end" and event.get("reason"):
            return str(event["reason"])
    return None


def _trajectory_final_ts(events: list[dict]) -> float | None:
    for event in reversed(events):
        ts = _event_ts(event)
        if ts is not None:
            return ts
    return None


def _event_turn(event: dict) -> int:
    try:
        return int(event.get("turn") or 1)
    except (TypeError, ValueError):
        return 1


def _event_ts(event: dict) -> float | None:
    raw = event.get("ts")
    if not raw:
        return None
    try:
        text = str(raw)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _session_title_from_messages(messages: list[dict], sid: int) -> str:
    for message in messages:
        if message.get("role") == "user":
            content = str(message.get("content") or "").strip().replace("\n", " ")
            if content:
                return _compact(content, 30)
    return f"Session {sid}"


def _format_workspace_sessions(cfg: Config, current: AgentSession) -> str:
    sessions = list_workspace_sessions(cfg.workdir)
    if not sessions:
        return ""
    lines = ["Workspace history:"]
    for item in sessions[:8]:
        mark = "*" if item.session_id == current.session_id else " "
        lines.append(
            f"{mark} {item.title}  session={item.session_id}  run={item.last_run_id or '-'}")
    return "\n".join(lines)


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
    elif t == "tool_input_updated":
        activity.phase = "updated"
        activity.arguments = event.get("arguments") or activity.arguments
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
    elif t == "agent_start":
        activity.phase = "agent running"
        activity.name = "agent"
        activity.agent_id = str(event.get("agent_id") or "")
        activity.agent_type = str(event.get("agent_type") or "")
        activity.agent_run_id = str(event.get("run_id") or "")
        activity.agent_fork = bool(event.get("fork"))
        activity.arguments = {
            "description": event.get("description"),
            "prompt": event.get("prompt"),
            "subagent_type": event.get("agent_type"),
            "fork": event.get("fork"),
            "resume_count": event.get("resume_count"),
        }
    elif t == "agent_progress":
        activity.phase = str(event.get("phase") or "agent running")
        activity.name = "agent"
        activity.agent_id = str(event.get("agent_id") or activity.agent_id or "")
        activity.agent_type = str(event.get("agent_type") or activity.agent_type or "")
    elif t in {"agent_done", "agent_error"}:
        activity.phase = "done" if t == "agent_done" else "error"
        activity.ok = t == "agent_done"
        activity.name = "agent"
        activity.agent_id = str(event.get("agent_id") or activity.agent_id or "")
        activity.agent_type = str(event.get("agent_type") or activity.agent_type or "")
        activity.agent_status = str(event.get("status") or "")
        activity.agent_run_id = str(event.get("run_id") or activity.agent_run_id or "")
        activity.agent_trajectory_path = str(event.get("trajectory_path") or "")
        activity.result_preview = _compact(str(event.get("final_message") or ""), 1200)
    elif t == "agent_background_start":
        activity.phase = "background"
        activity.name = "agent"
        activity.agent_id = str(event.get("agent_id") or "")
        activity.agent_type = str(event.get("agent_type") or "")
        activity.agent_background = True
        activity.agent_fork = bool(event.get("fork"))
        activity.arguments = {
            "description": event.get("description"),
            "subagent_type": event.get("agent_type"),
            "fork": event.get("fork"),
            "run_in_background": True,
            "resumed": event.get("resumed"),
            "resume_count": event.get("resume_count"),
        }
    elif t == "agent_background_done":
        activity.phase = "done" if event.get("status") == "completed" else "error"
        activity.ok = event.get("status") == "completed"
        activity.name = "agent"
        activity.agent_id = str(event.get("agent_id") or activity.agent_id or "")
        activity.agent_type = str(event.get("agent_type") or activity.agent_type or "")
        activity.agent_status = str(event.get("status") or "")
        activity.agent_run_id = str(event.get("run_id") or activity.agent_run_id or "")
        activity.agent_trajectory_path = str(event.get("trajectory_path") or "")
        activity.agent_background = True
        activity.agent_fork = bool(event.get("fork", activity.agent_fork))
        if activity.arguments is None:
            activity.arguments = {}
        activity.arguments["resumable"] = event.get("resumable")
        activity.arguments["resume_count"] = event.get("resume_count")
        activity.result_preview = _compact(str(event.get("final_message") or event.get("error") or ""), 1200)
    return activity


def _render_record(record: UiRecord):
    if record.kind == "logo":
        return _tiny_agent_logo()
    if record.kind == "build_activity":
        build = record.meta.get("build")
        if isinstance(build, BuildActivity):
            return _render_build_activity(build)
    if record.kind == "tool_activity":
        activity = record.meta.get("activity")
        if isinstance(activity, ToolActivity):
            return _render_tool_activity(activity)
    if record.kind == "permission_request":
        prompt = record.meta.get("prompt")
        if isinstance(prompt, PermissionPromptState):
            return _render_permission_request(prompt)
    if record.kind == "thinking":
        return _render_thinking(record.content)
    if record.role == "user":
        text = Text()
        text.append("> ", style="bold cyan")
        text.append(record.content)
        return text
    if record.role == "assistant":
        return _render_assistant_markdown(record.content)
    if record.role == "system":
        text = Text()
        style = "red" if record.kind == "error" else "dim"
        text.append("! ", style=style)
        text.append(record.content, style=style)
        return text
    return Text(record.content)


def _tiny_agent_logo() -> Text:
    text = Text()
    # 赛博朋克风格 ASCII Art
    lines = [
        "████████╗██╗███╗   ██╗██╗   ██╗    █████╗  ██████╗ ███████╗███╗   ██╗████████╗",
        "╚══██╔══╝██║████╗  ██║╚██╗ ██╔╝   ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝",
        "   ██║   ██║██╔██╗ ██║ ╚████╔╝    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ",
        "   ██║   ██║██║╚██╗██║  ╚██╔╝     ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ",
        "   ██║   ██║██║ ╚████║   ██║      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ",
        "   ╚═╝   ╚═╝╚═╝  ╚═══╝   ╚═╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ",
    ]
    for index, line in enumerate(lines):
        # 前3行用科技蓝，后3行用活力橙，打造上下双色渐变效果
        style = "bold #0082ff" if index < 3 else "bold #105286"
        text.append(line, style=style)
        text.append("\n")
    # 底部标语使用暗色 (dim)，不喧宾夺主
    text.append("ProMax coding agent harness", style="dim")
    return text


def _render_assistant_markdown(content: str):
    return Markdown(content)


def _render_thinking(content: str) -> Text:
    text = Text()
    text.append(". Thinking: ", style="italic #6e7681")
    text.append(_compact(_one_line(content), 900), style="italic #6e7681")
    return text


def _render_permission_prompt_body(prompt: PermissionPromptState) -> Text:
    rule = permission_rule_value_to_string(
        suggest_permission_rule_value(prompt.tool, prompt.arguments))
    reason = str(getattr(prompt.decision, "message", "") or "permission required")
    mode = str(getattr(prompt.decision, "mode", "default") or "default")
    text = Text()
    text.append("Tool  ", style="bold yellow")
    text.append(prompt.tool, style="bold white")
    text.append("\nRule  ", style="bold yellow")
    text.append(_compact(_one_line(rule), 160), style="white")
    text.append("\nMode  ", style="bold yellow")
    text.append(mode, style="dim")
    text.append("\nReason  ", style="bold yellow")
    text.append(_compact(_one_line(reason), 180), style="dim")
    try:
        if prompt.cfg is not None and managed_permission_rules_only(prompt.cfg):
            text.append("\nPolicy  ", style="bold red")
            text.append("managed permission rules only; local/session choices may not persist for future calls",
                        style="red")
    except Exception:
        pass
    return text


def _permission_option(title: str, description: str) -> Text:
    text = Text()
    text.append(title, style="bold #58a6ff")
    text.append("\n   ")
    text.append(description, style="dim")
    return text


def _permission_choices() -> list[tuple[str, str, str]]:
    return [
        ("y", "Allow once", "Approve this tool call only"),
        ("s", "Allow this session", "Remember the matching rule for this session"),
        ("l", "Allow locally", "Persist the matching rule to local settings"),
        ("p", "Allow in project", "Persist the matching rule to project settings"),
        ("n", "Deny", "Reject this request"),
    ]


def _permission_choice_label(choice: str | None) -> str:
    for item_choice, label, _description in _permission_choices():
        if item_choice == choice:
            return label
    return "Deny"


def _render_permission_request(prompt: PermissionPromptState) -> Text:
    rule = permission_rule_value_to_string(
        suggest_permission_rule_value(prompt.tool, prompt.arguments))
    reason = str(getattr(prompt.decision, "message", "") or "permission required")
    text = Text()
    answered = prompt.choice is not None
    base_style = "dim" if answered else "white"
    accent = "#6e7681" if answered else "#b083f0"
    text.append("| ", style=accent)
    text.append("# Questions", style=f"bold {accent}")
    text.append("\n| ", style=accent)
    text.append(f"Permission required for {prompt.tool}", style=base_style)
    text.append("\n| ", style=accent)
    text.append(_compact(_one_line(reason), 180), style="dim")
    text.append("\n| ", style=accent)
    text.append("rule: ", style="dim")
    text.append(_compact(_one_line(rule), 160), style="dim")
    try:
        if prompt.cfg is not None and managed_permission_rules_only(prompt.cfg):
            text.append("\n| ", style=accent)
            text.append("policy: managed permission rules only", style="red")
    except Exception:
        pass
    if answered:
        text.append("\n| \n| ", style=accent)
        text.append("selected: ", style="dim")
        text.append(_permission_choice_label(prompt.choice), style="bold white" if prompt.choice != "n" else "bold red")
        return text

    text.append("\n| \n", style=accent)
    for index, (_choice, label, description) in enumerate(_permission_choices(), start=1):
        selected = index - 1 == prompt.selected
        marker = ">" if selected else "|"
        style = "bold #58a6ff" if selected else "white"
        text.append(f"{marker} {index}. {label}", style=style)
        text.append(f"\n|    {description}\n", style="dim")
    text.append("| \n| tab/up/down select   enter confirm   esc dismiss", style="dim")
    return text


def _render_build_activity(build: BuildActivity) -> Text:
    text = Text()
    spinner = _spinner_frame(build) if build.running else "done"
    text.append(f"{spinner} ", style="blue" if build.running else "green")
    text.append("Build", style="bold white")
    text.append(f" | {build.model}", style="dim")
    text.append(f" | {_format_elapsed(build.elapsed_s)}", style="dim")
    text.append(f" | {build.status}", style="dim")
    if build.thinking.strip():
        text.append("\n  . Thinking: ", style="italic #6e7681")
        text.append(_compact(_one_line(build.thinking), 220), style="italic #6e7681")
    if build.tool_ids:
        noun = "toolcall" if len(build.tool_ids) == 1 else "toolcalls"
        text.append(f"\n  {len(build.tool_ids)} {noun}", style="dim")
        if build.current_tool:
            text.append(f" | current: {_compact(build.current_tool, 140)}", style="dim")
    if not build.running:
        text.append(f"\n  done in {_format_elapsed(build.elapsed_s)} | details: /build {_build_number(build)}", style="dim")
    return text


def _render_build_detail_title(build: BuildActivity,
                               build_index: int | None = None,
                               build_count: int | None = None) -> Text:
    text = Text()
    text.append("▣ ", style="blue")
    text.append("Build", style="bold white")
    if build_index is not None and build_count:
        text.append(f" {build_index + 1}/{build_count}", style="dim")
    text.append(f" · {build.model}", style="dim")
    text.append(f" · {_format_elapsed(build.elapsed_s)}", style="dim")
    text.append(f" · {build.status}", style="dim")
    if build.run_id:
        text.append(f" · {build.run_id}", style="dim")
    return text


def _build_tool_options(build: BuildActivity | None,
                        activities: dict[str, ToolActivity]) -> list[Option]:
    if build is None or not build.tool_ids:
        return [Option("No tool calls", id="-1", disabled=True)]
    options: list[Option] = []
    for index, call_id in enumerate(build.tool_ids):
        activity = activities.get(call_id)
        if not activity:
            label = f"{index + 1:02d}. unknown"
        else:
            status = "ok" if activity.ok else activity.phase
            label = f"{index + 1:02d}. {_compact(_tool_title(activity), 28)}  {status}"
        options.append(Option(label, id=str(index)))
    return options


def _render_build_detail(build: BuildActivity, activities: dict[str, ToolActivity],
                         tool_index: int = 0) -> Text:
    text = Text()
    text.append("Build details", style="bold cyan")
    text.append(f"  {len(build.tool_ids)} toolcalls", style="dim")
    text.append(f"  {_format_elapsed(build.elapsed_s)}", style="dim")
    text.append("\n\n")
    if build.thinking.strip():
        text.append("Thinking\n", style="italic #b7a46a")
        text.append(_compact(build.thinking.strip(), 3000), style="dim")
        text.append("\n\n")
    if not build.tool_ids:
        text.append("No tool calls for this build.", style="dim")
        return text

    tool_index = max(0, min(tool_index, len(build.tool_ids) - 1))
    call_id = build.tool_ids[tool_index]
    selected = activities.get(call_id)

    text.append("Selected tool\n", style="bold cyan")
    if selected:
        text.append(_tool_title(selected), style=f"bold {_activity_style(selected)}")
        details = _tool_inline_details(selected)
        if details:
            text.append(f"  {details}", style="dim")
        if selected.permission_reason:
            text.append("\npermission: ", style="bold #ffad42")
            text.append(selected.permission_reason, style="#ffad42")
        if selected.permission_rule:
            text.append("\nrule: ", style="bold #ffad42")
            text.append(selected.permission_rule, style="#ffad42")
        if selected.persisted_path:
            text.append("\nsaved: ", style="bold cyan")
            text.append(selected.persisted_path, style="cyan")
        if selected.agent_type:
            text.append("\nsubagent: ", style="bold cyan")
            text.append(selected.agent_type, style=_agent_style(selected.agent_type))
        if selected.agent_background or selected.agent_fork:
            text.append("\nsubagent mode: ", style="bold cyan")
            modes = []
            if selected.agent_background:
                modes.append("background")
            if selected.agent_fork:
                modes.append("fork")
            text.append(", ".join(modes), style="cyan")
        if selected.agent_run_id:
            text.append("\nsubagent run: ", style="bold cyan")
            text.append(selected.agent_run_id, style="cyan")
        if selected.agent_trajectory_path:
            text.append("\nsubagent trajectory: ", style="bold cyan")
            text.append(selected.agent_trajectory_path, style="cyan")
        if selected.context_note:
            text.append("\ncontextModifier: ", style="bold magenta")
            text.append(selected.context_note, style="magenta")
        text.append("\n")
        if selected.arguments:
            text.append("\ninput\n", style="bold #58a6ff")
            text.append(_pretty_args(selected.arguments), style="dim")
            text.append("\n")
        if selected.result_preview:
            text.append("\noutput\n", style="bold #58a6ff")
            text.append(_compact(selected.result_preview, 5000),
                        style="white" if selected.ok else "red")
            text.append("\n")

    if build.audit:
        text.append("\nLifecycle\n", style="bold blue")
        for event in build.audit[-40:]:
            text.append(_format_audit_event(event), style=_audit_style(event))
            text.append("\n")
    return text


def _render_build_detail_footer(build: BuildActivity, tool_index: int,
                                build_index: int | None = None,
                                build_count: int | None = None) -> Text:
    text = Text()
    count = len(build.tool_ids)
    position = f"{tool_index + 1} of {count}" if count else "0 of 0"
    text.append("Build ", style="bold white")
    if build_index is not None and build_count:
        text.append(f"{build_index + 1}/{build_count} ", style="dim")
    text.append(f"({position} tools)", style="dim")
    text.append(f"  {build.model}", style="dim")
    text.append(f"  {_format_elapsed(build.elapsed_s)}", style="dim")
    spacer = " " * 8
    text.append(spacer)
    text.append("Parent", style="bold white")
    text.append(" esc   ", style="dim")
    text.append("Prev Build", style="bold white")
    text.append(" left   ", style="dim")
    text.append("Next Build", style="bold white")
    text.append(" right", style="dim")
    return text


def _render_build_detail_legacy(build: BuildActivity, activities: dict[str, ToolActivity]) -> Text:
    text = Text()
    text.append("Build", style="bold white")
    text.append(f" | {build.model} | {_format_elapsed(build.elapsed_s)} | {build.status}\n", style="dim")
    if build.thinking.strip():
        text.append("\nThinking\n", style="bold #b7a46a")
        text.append(_compact(build.thinking.strip(), 3000), style="dim")
        text.append("\n")
    if build.tool_ids:
        text.append("\nTools\n", style="bold cyan")
    for call_id in build.tool_ids:
        activity = activities.get(call_id)
        if not activity:
            continue
        text.append("\n")
        text.append(_tool_title(activity), style=f"bold {_activity_style(activity)}")
        details = _tool_inline_details(activity)
        if details:
            text.append(f"  {details}", style="dim")
        if activity.permission_reason:
            text.append(f"\n  permission: {activity.permission_reason}", style="yellow")
        if activity.persisted_path:
            text.append(f"\n  saved: {activity.persisted_path}", style="cyan")
        if activity.context_note:
            text.append(f"\n  contextModifier: {activity.context_note}", style="magenta")
        if activity.result_preview:
            text.append("\n  output: ", style="dim")
            text.append(_compact(activity.result_preview, 1200), style="dim" if activity.ok else "red")
        text.append("\n")
    if build.audit:
        text.append("\nLifecycle\n", style="bold blue")
        for event in build.audit[-80:]:
            text.append(_format_audit_event(event), style=_audit_style(event))
            text.append("\n")
    return text


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
    if activity.name == "agent":
        target = activity.agent_type or args.get("subagent_type") or "general"
        desc = args.get("description") or ""
        flags = []
        if activity.agent_background:
            flags.append("bg")
        if activity.agent_fork:
            flags.append("fork")
        marker = f" [{' '.join(flags)}]" if flags else ""
        return f"Agent {target}{marker} {_compact(str(desc), 80)}".strip()
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
    if activity.agent_type:
        bits.append(f"agent={activity.agent_type}")
    if activity.agent_status:
        bits.append(f"status={activity.agent_status}")
    if activity.agent_background:
        bits.append("background")
    if activity.agent_fork:
        bits.append("fork")
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
    if activity.name == "agent":
        return _agent_style(activity.agent_type)
    if activity.phase == "running":
        return "cyan"
    if activity.phase == "done" or activity.ok is True:
        return "green"
    return "magenta"


def _agent_style(agent_type: str | None) -> str:
    return {
        "explore": "cyan",
        "plan": "green",
        "general": "yellow",
        "verify": "red",
        "worker": "magenta",
    }.get(agent_type or "", "magenta")


def _activity_status(activity: ToolActivity) -> str:
    if activity.permission == "waiting":
        return f"waiting permission: {activity.name}"
    if activity.name == "agent":
        return f"{activity.phase}: {activity.agent_type or 'agent'}"
    return f"{activity.phase}: {activity.name}"


def _latest_build(ui: UiSession) -> BuildActivity | None:
    if ui.current_build and _build_has_details(ui.current_build):
        return ui.current_build
    for record in reversed(ui.records):
        if record.kind == "build_activity":
            build = record.meta.get("build")
            if isinstance(build, BuildActivity) and _build_has_details(build):
                return build
    if ui.current_build:
        return ui.current_build
    return None


def _build_has_details(build: BuildActivity) -> bool:
    return bool(build.tool_ids or build.audit or build.thinking.strip())


def _build_index(builds: list[BuildActivity], build: BuildActivity) -> int:
    try:
        return builds.index(build)
    except ValueError:
        return max(len(builds) - 1, 0)


def _spinner_frame(build: BuildActivity) -> str:
    frames = ("|", "/", "-", "\\")
    index = int(build.elapsed_s * 8) % len(frames)
    return frames[index]


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rest = int(seconds % 60)
    return f"{minutes}m {rest}s"


def _build_number(build: BuildActivity) -> str:
    return build.id.rsplit("-", 1)[-1]


def _renderable_line_count(renderable: Any) -> int:
    plain = getattr(renderable, "plain", None)
    if isinstance(plain, str):
        return max(1, plain.count("\n") + 1)
    return max(1, str(renderable).count("\n") + 1)


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _command_matches(text: str) -> list[tuple[str, str]]:
    query = text.strip()
    if not query.startswith("/"):
        return []
    name = query.split(maxsplit=1)[0].lower()
    if name == "/":
        return list(COMMANDS)
    starts = [item for item in COMMANDS if item[0].split()[0].lower().startswith(name)]
    contains = [
        item for item in COMMANDS
        if item not in starts and name.strip("/") in item[0].lower()
    ]
    return starts + contains


def _footer_line(ui: UiSession, cfg: Config) -> str:
    build = _latest_build(ui)
    mode = "coordinator" if cfg.coordinator_mode else "normal"
    left = f"Build | {cfg.model} | {cfg.permission_mode} | {mode}"
    if build and build.running:
        left += f" | {build.status}"
    tokens = ui.agent.usage_total.as_dict()
    total_tokens = sum(int(v) for v in tokens.values())
    return (
        f"{left} | {settings_status_line(cfg)} | {memory_status_line(cfg)} | "
        f"{hooks_status_line(cfg)} | "
        f"{context_status_line(ui.agent.context_status())}    "
        f"{total_tokens:,} tokens | ${ui.agent.cost_usd:.4f}    "
        "ctrl+p commands | enter send | ctrl+j newline"
    )


def _handle_memory_command(text: str, cfg: Config, session: AgentSession) -> str:
    rest = text.strip()
    if not rest:
        return format_memory_summary(cfg)
    name, _, tail = rest.partition(" ")
    if name == "status":
        return format_memory_runtime_status(cfg, session.memory_controller)
    if name == "tail":
        return format_memory_tail(cfg, tail.strip())
    if name == "extract":
        return "◈ Extracting..\n" + extract_memory_for_session(session)
    if name == "on":
        session.set_memory_auto_extract(True)
        return "[◈ Memory Auto Extract On]"
    if name == "off":
        session.set_memory_auto_extract(False)
        return "[◈ Memory Auto Extract Off]"
    if name == "add":
        return add_memory_from_text(cfg, tail)
    if name == "forget":
        return forget_memory_from_text(cfg, tail)
    if name == "rebuild":
        return rebuild_memory_index_for_cfg(cfg)
    if name == "list":
        return format_memory_summary(cfg, f"list {tail}".strip())
    if name == "read":
        return format_memory_summary(cfg, f"read {tail}".strip())
    if name in {"sources", "prompt"}:
        return format_memory_summary(cfg, name)
    return "Usage: /memory [status|sources|prompt|tail|extract|on|off|list [type]|read <id>|add ...|forget <id>|rebuild]"


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
        for key in ("decision", "reason_type", "mode", "resolver", "ok", "persisted",
                    "agent_type", "status", "phase"):
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


def _format_memory_event_status(event: dict) -> str:
    kind = event.get("type")
    if kind == "memory_extract_start":
        return "◈ Extracting.."
    if kind == "memory_extract_saved":
        return f"[◈ Memory Saved] {event.get('count', 0)}"
    if kind == "memory_extract_error":
        return f"◈ Memory Error: {_compact(str(event.get('error') or ''), 60)}"
    if kind == "memory_extract_trailing":
        return f"◈ Extracting.. {event.get('status')}"
    return f"◈ Memory skipped: {event.get('reason')}"


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


def _format_auto_mode_status(cfg: Config) -> str:
    if cfg.permission_mode == "bypass" and cfg.yolo:
        state = "full"
        detail = "bypass mode plus yolo for legacy dangerous checks"
    elif cfg.permission_mode == "bypass":
        state = "on"
        detail = "bypass mode; sensitive file checks still apply"
    else:
        state = "off"
        detail = f"permission_mode={cfg.permission_mode}"
    return f"auto_mode: {state}\n{detail}\nyolo={cfg.yolo}"


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
