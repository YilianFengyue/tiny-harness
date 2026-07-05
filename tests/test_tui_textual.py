from harness.tui_textual import (
    BuildActivity,
    PermissionPromptState,
    ToolActivity,
    UiRecord,
    _activity_style,
    _command_matches,
    _fold_tool_event,
    _render_build_activity,
    _render_build_detail_footer,
    _render_permission_request,
    _restore_builds_from_events,
    _latest_build,
    _tool_title,
)
from harness.context_view import context_pill


def test_tui_folds_tool_lifecycle_into_single_activity():
    activities: dict[str, ToolActivity] = {}
    events = [
        {"type": "tool_call", "tool_call_id": "c1", "name": "bash",
         "arguments": {"command": "python -m pytest -q"}},
        {"type": "tool_queued", "tool_call_id": "c1", "name": "bash",
         "arguments": {"command": "python -m pytest -q"}},
        {"type": "tool_validate", "tool_call_id": "c1", "name": "bash",
         "ok": True, "read_only": False, "concurrency_safe": False,
         "destructive": True},
        {"type": "tool_permission", "tool_call_id": "c1", "name": "bash",
         "ok": True, "decision": "allow", "reason": "matched rule",
         "mode": "acceptEdits", "rule": "Bash(python *)"},
        {"type": "tool_start", "tool_call_id": "c1", "name": "bash"},
        {"type": "tool_result", "tool_call_id": "c1", "name": "bash",
         "ok": True, "result": "12 passed in 0.31s", "duration_ms": 310},
        {"type": "tool_end", "tool_call_id": "c1", "name": "bash",
         "ok": True, "duration_ms": 310},
    ]

    for event in events:
        _fold_tool_event(activities, event)

    assert list(activities) == ["c1"]
    activity = activities["c1"]
    assert activity.name == "bash"
    assert activity.phase == "done"
    assert activity.ok is True
    assert activity.destructive is True
    assert activity.permission == "allow"
    assert activity.permission_rule == "Bash(python *)"
    assert activity.result_preview == "12 passed in 0.31s"
    assert len(activity.audit) == len(events)


def test_tui_activity_keeps_verbose_permission_and_persistence_details():
    activities: dict[str, ToolActivity] = {}
    _fold_tool_event(activities, {"type": "tool_call", "tool_call_id": "c2",
                                  "name": "bash", "arguments": {"command": "python big.py"}})
    _fold_tool_event(activities, {"type": "tool_permission_wait", "tool_call_id": "c2",
                                  "name": "bash", "reason": "permission required",
                                  "reason_type": "passthrough_to_ask", "mode": "default"})
    _fold_tool_event(activities, {"type": "tool_permission_update", "tool_call_id": "c2",
                                  "name": "bash", "persisted": False,
                                  "summary": "add session allow Bash(python *)"})
    _fold_tool_event(activities, {"type": "tool_result_persisted", "tool_call_id": "c2",
                                  "name": "bash", "path": "runs/x/large.txt"})
    _fold_tool_event(activities, {"type": "tool_context_modified", "tool_call_id": "c2",
                                  "name": "bash", "kind": "runtime", "changed": 1})

    activity = activities["c2"]
    assert activity.permission == "waiting"
    assert activity.permission_reason == "permission required"
    assert activity.permission_mode == "default"
    assert activity.permission_update == "add session allow Bash(python *)"
    assert activity.persisted_path == "runs/x/large.txt"
    assert activity.context_note == "kind=runtime changed=1"
    assert [event["type"] for event in activity.audit] == [
        "tool_call",
        "tool_permission_wait",
        "tool_permission_update",
        "tool_result_persisted",
        "tool_context_modified",
    ]


def test_tui_folds_agent_lifecycle_into_tool_activity():
    activities: dict[str, ToolActivity] = {}
    _fold_tool_event(activities, {"type": "tool_call", "tool_call_id": "a1",
                                  "name": "agent", "arguments": {
                                      "description": "inspect code",
                                      "subagent_type": "explore",
                                  }})
    _fold_tool_event(activities, {"type": "agent_start", "tool_call_id": "a1",
                                  "name": "agent", "agent_id": "run-child",
                                  "agent_type": "explore", "run_id": "run-child",
                                  "description": "inspect code",
                                  "prompt": "find issue"})
    _fold_tool_event(activities, {"type": "agent_done", "tool_call_id": "a1",
                                  "name": "agent", "agent_id": "run-child",
                                  "agent_type": "explore", "run_id": "run-child",
                                  "status": "completed",
                                  "trajectory_path": "runs/run-child/trajectory.jsonl",
                                  "final_message": "found issue"})

    activity = activities["a1"]
    rendered = _render_build_activity(BuildActivity("b1", model="gpt-test")).plain

    assert activity.name == "agent"
    assert activity.agent_type == "explore"
    assert activity.agent_run_id == "run-child"
    assert activity.agent_trajectory_path.endswith("trajectory.jsonl")
    assert activity.result_preview == "found issue"
    assert "Build" in rendered


def test_tui_folds_background_agent_lifecycle_into_tool_activity():
    activities: dict[str, ToolActivity] = {}
    _fold_tool_event(activities, {"type": "tool_call", "tool_call_id": "a2",
                                  "name": "agent", "arguments": {
                                      "description": "background inspect",
                                      "subagent_type": "explore",
                                      "run_in_background": True,
                                  }})
    _fold_tool_event(activities, {"type": "agent_background_start", "tool_call_id": "a2",
                                  "name": "agent", "agent_id": "bg-1",
                                  "agent_type": "explore", "fork": True,
                                  "description": "background inspect"})
    _fold_tool_event(activities, {"type": "agent_background_done", "tool_call_id": "a2",
                                  "name": "agent", "agent_id": "bg-1",
                                  "agent_type": "explore", "run_id": "run-child",
                                  "status": "completed",
                                  "trajectory_path": "runs/run-child/trajectory.jsonl",
                                  "final_message": "background result"})

    activity = activities["a2"]

    assert activity.phase == "done"
    assert activity.ok is True
    assert activity.agent_type == "explore"
    assert activity.agent_status == "completed"
    assert activity.agent_background is True
    assert activity.agent_fork is True
    assert activity.result_preview == "background result"


def test_tui_agent_activity_uses_type_color_and_mode_labels():
    activity = ToolActivity(
        "a3",
        name="agent",
        agent_type="audit-reviewer",
        agent_background=True,
        agent_fork=True,
    )

    assert _activity_style(activity) == "magenta"
    assert "[bg fork]" in _tool_title(activity)


def test_tui_slash_menu_filters_commands():
    matches = _command_matches("/per")

    assert matches
    assert matches[0][0].startswith("/permissions")


def test_tui_slash_menu_includes_settings_and_features():
    settings = _command_matches("/set")
    agents = _command_matches("/age")
    features = _command_matches("/fea")
    memory = _command_matches("/mem")
    hooks = _command_matches("/hoo")
    context = _command_matches("/con")
    compact = _command_matches("/com")
    auto_mode = _command_matches("/auto")
    coordinator = _command_matches("/coord")

    assert settings and settings[0][0].startswith("/settings")
    assert agents and agents[0][0].startswith("/agents")
    assert features and features[0][0].startswith("/features")
    assert memory and memory[0][0].startswith("/memory")
    assert hooks and hooks[0][0].startswith("/hooks")
    assert context and context[0][0].startswith("/context")
    assert compact and compact[0][0].startswith("/compact")
    assert auto_mode and auto_mode[0][0].startswith("/auto_mode")
    assert coordinator and coordinator[0][0].startswith("/coordinator")


def test_context_pill_renders_circle_percentage():
    rendered = context_pill({"level": "warning", "percent_used": 87})

    assert rendered.startswith("◐ context")
    assert "87%" in rendered


def test_build_activity_collects_thinking_and_tool_refs():
    build = BuildActivity("build-1-1", model="gpt-test")
    build.thinking += "I should inspect the workspace."
    build.tool_ids.append("c1")
    build.current_tool = "Grep legacy_parse"
    build.status = "running tools"

    assert build.running
    assert build.tool_ids == ["c1"]
    assert "inspect" in build.thinking
    assert build.current_tool == "Grep legacy_parse"


def test_build_activity_tracks_folded_tool_activity():
    build = BuildActivity("b1", model="gpt-test")
    activities: dict[str, ToolActivity] = {}
    activity = _fold_tool_event(activities, {
        "type": "tool_call",
        "tool_call_id": "c1",
        "name": "grep",
        "arguments": {"pattern": "legacy_parse"},
    })

    assert activity is not None
    build.tool_ids.append(activity.call_id)
    build.current_tool = activity.name
    build.status = "running tools"

    assert build.tool_ids == ["c1"]
    assert build.current_tool == "grep"
    assert activities["c1"].arguments == {"pattern": "legacy_parse"}


def test_permission_request_render_is_concise_and_not_tool_activity():
    class Decision:
        mode = "default"
        message = "permission required before running this tool"

    prompt = PermissionPromptState(
        "bash",
        {"command": "python - <<'PY'\n" + ("print('hello')\n" * 80) + "PY"},
        Decision(),
        result=None,  # type: ignore[arg-type]
    )
    rendered = _render_permission_request(prompt).plain

    assert "Permission required" in rendered
    assert "Allow once" in rendered
    assert "Allow this session" in rendered
    assert "print('hello')" not in rendered
    assert UiRecord("system", "", "permission_request", {"prompt": prompt}).kind == "permission_request"

    prompt.choice = "s"
    answered = _render_permission_request(prompt).plain
    assert "selected: Allow this session" in answered
    assert "Allow locally" not in answered


def test_build_done_renders_elapsed_and_ghost_thinking():
    build = BuildActivity("build-1-1", model="gpt-test")
    build.started_at = 10.0
    build.finished_at = 13.4
    build.status = "completed"
    build.thinking = "I should inspect the workspace."

    rendered = _render_build_activity(build).plain

    assert "done in 3.4s" in rendered
    assert "details: /build 1" in rendered
    assert "Ctrl+O" not in rendered
    assert ". Thinking:" in rendered
    assert "I should inspect" in rendered


def test_latest_build_skips_empty_current_build():
    class Ui:
        pass

    ui = Ui()
    empty = BuildActivity("build-1-2", model="gpt-test")
    useful = BuildActivity("build-1-1", model="gpt-test")
    useful.tool_ids.append("c1")
    ui.current_build = empty
    ui.records = [
        UiRecord("assistant", useful.id, "build_activity", {"build": useful}),
        UiRecord("assistant", empty.id, "build_activity", {"build": empty}),
    ]

    assert _latest_build(ui) is useful


def test_build_detail_footer_advertises_build_navigation():
    build = BuildActivity("build-1-1", model="gpt-test")
    build.tool_ids.append("c1")

    rendered = _render_build_detail_footer(build, 0, 1, 3).plain

    assert "2/3" in rendered
    assert "Prev Build" in rendered
    assert "Next Build" in rendered


def test_restore_builds_from_trajectory_events_rehydrates_build_details():
    class Ui:
        id = 1
        build_seq = 0
        builds: list[BuildActivity] = []
        records: list[UiRecord] = []
        activities: dict[str, ToolActivity] = {}
        audit_events: list[dict] = []
        current_build = None

    ui = Ui()
    events = [
        {"type": "run_start", "run_id": "run-1", "ts": "2026-07-05T00:00:00+00:00",
         "model": "gpt-test"},
        {"type": "assistant_delta", "turn": 1, "ts": "2026-07-05T00:00:01+00:00",
         "reasoning_content": "Need to run a check."},
        {"type": "tool_call", "turn": 1, "ts": "2026-07-05T00:00:02+00:00",
         "tool_call_id": "c1", "name": "bash",
         "arguments": {"command": "python -m pytest -q"}},
        {"type": "tool_result", "turn": 1, "ts": "2026-07-05T00:00:03+00:00",
         "tool_call_id": "c1", "name": "bash", "ok": True,
         "result": "5 passed", "duration_ms": 100},
        {"type": "run_end", "run_id": "run-1", "ts": "2026-07-05T00:00:04+00:00",
         "reason": "completed"},
    ]

    _restore_builds_from_events(ui, events, "fallback-model")

    build = _latest_build(ui)
    assert build is not None
    assert build.run_id == "run-1"
    assert build.status == "completed"
    assert build.finished_at is not None
    assert build.tool_ids == ["c1"]
    assert "Need to run" in build.thinking
    assert ui.activities["c1"].result_preview == "5 passed"
    assert any(record.kind == "build_activity" for record in ui.records)
