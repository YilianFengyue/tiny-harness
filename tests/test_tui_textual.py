from harness.tui_textual import (
    BuildActivity,
    PermissionPromptState,
    ToolActivity,
    UiRecord,
    _command_matches,
    _fold_tool_event,
    _render_build_activity,
    _render_permission_request,
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


def test_tui_slash_menu_filters_commands():
    matches = _command_matches("/per")

    assert matches
    assert matches[0][0].startswith("/permissions")


def test_tui_slash_menu_includes_settings_and_features():
    settings = _command_matches("/set")
    features = _command_matches("/fea")
    memory = _command_matches("/mem")
    context = _command_matches("/con")
    compact = _command_matches("/com")

    assert settings and settings[0][0].startswith("/settings")
    assert features and features[0][0].startswith("/features")
    assert memory and memory[0][0].startswith("/memory")
    assert context and context[0][0].startswith("/context")
    assert compact and compact[0][0].startswith("/compact")


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
