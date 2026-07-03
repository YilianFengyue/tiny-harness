from harness.tui_textual import ToolActivity, _fold_tool_event


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
