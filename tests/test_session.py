from conftest import MockProvider, turn

from harness.session import AgentSession
from harness.session_store import latest_workspace_session
from harness.telemetry import read_trajectory
import json


def test_session_preserves_history_across_submits(make_cfg):
    cfg = make_cfg()
    provider = MockProvider([
        turn(calls=[("c1", "list_files", '{"path": null}')]),
        turn(content="I listed the workspace."),
        turn(content="I still have the previous workspace listing context."),
    ])
    session = AgentSession.fresh(cfg, provider)

    first = session.submit("List the workspace files.")
    second = session.submit("Continue from the previous result.")

    assert first.summary["reason"] == "completed"
    assert second.summary["reason"] == "completed"
    assert session.last_run_id == second.run_id
    assert session.turns_submitted == 2
    assert session.cost_usd > 0
    assert session.usage_total.prompt_tokens == 300
    assert session.usage_total.completion_tokens == 150

    # Third provider request is the second user submit. It must include the
    # first submit's assistant/tool/final answer history, not start fresh.
    second_submit_request = provider.requests[2]
    roles = [m["role"] for m in second_submit_request]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert second_submit_request[-1]["content"] == "Continue from the previous result."
    assert second_submit_request[2]["tool_calls"][0]["id"] == "c1"
    assert "I listed the workspace." in second_submit_request[4]["content"]


def test_workspace_latest_session_restores_history_after_restart(make_cfg):
    cfg = make_cfg()
    first_provider = MockProvider([turn(content="I remember this workspace.")])
    first = AgentSession.fresh(cfg, first_provider)

    first_turn = first.submit("Remember the project plan.")

    stored = latest_workspace_session(cfg.workdir)
    assert stored is not None
    assert stored.session_id == first.session_id
    assert stored.last_run_id == first_turn.run_id

    second_provider = MockProvider([turn(content="Continuing with the plan.")])
    restored = AgentSession.from_workspace_latest(cfg, second_provider)
    assert restored is not None
    assert restored.session_id == first.session_id
    assert restored.last_run_id == first_turn.run_id

    restored.submit("Continue from the previous message.")

    request = second_provider.requests[0]
    assert [m["role"] for m in request] == ["system", "user", "assistant", "user"]
    assert request[1]["content"] == "Remember the project plan."
    assert "I remember this workspace." in request[2]["content"]
    assert request[-1]["content"] == "Continue from the previous message."


def test_workspace_session_index_is_scoped_to_workdir(make_cfg, tmp_path):
    cfg_a = make_cfg(workdir=tmp_path / "a")
    cfg_b = make_cfg(workdir=tmp_path / "b", runs_dir=cfg_a.runs_dir)
    provider = MockProvider([turn(content="A only")])
    session = AgentSession.fresh(cfg_a, provider)

    session.submit("Keep this in workspace A.")

    assert AgentSession.from_workspace_latest(cfg_b, MockProvider([])) is None


def test_session_captures_live_loop_events_and_trajectory(make_cfg):
    cfg = make_cfg()
    provider = MockProvider([
        turn(calls=[("c1", "calculator", '{"expression": "20+22"}')]),
        turn(content="42"),
    ])
    session = AgentSession.fresh(cfg, provider)
    seen = []

    result = session.submit("Calculate 20+22.", on_event=seen.append)

    assert result.summary["reason"] == "completed"
    assert [e["type"] for e in seen if e["type"] == "transition"] == ["transition"]
    assert [e["reason"] for e in seen if e["type"] == "transition"] == ["next_turn"]
    assert session.trajectory_path().name == "trajectory.jsonl"

    events = read_trajectory(cfg.runs_dir, result.run_id)
    assert events[0]["session_id"] == session.session_id
    assert any(e["type"] == "turn_start" for e in events)
    assert any(e["type"] == "context_status" for e in events)
    assert any(e["type"] == "transition" and e["reason"] == "next_turn"
               for e in events)


def test_session_keeps_context_manager_and_manual_compacts(make_cfg):
    cfg = make_cfg(context_keep_recent=1)
    provider = MockProvider([turn(content="ok")])
    session = AgentSession.fresh(cfg, provider)
    manager_id = id(session.context_manager)
    session.messages.extend([
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "A" * 1000},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c2", "type": "function",
             "function": {"name": "read_file", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c2", "content": "B" * 1000},
    ])

    edit = session.compact_context("keep latest file result", summarize=False)

    assert id(session.context_manager) == manager_id
    assert edit and edit["kind"] == "manual_compact"
    assert edit["cleared_messages"] == 1
    assert "manually compacted" in session.messages[-3]["content"]
    assert session.messages[-1]["content"] == "B" * 1000
    assert session.app_state.get_state().context["last_compact_kind"] == "manual_compact"


def test_session_manual_summary_compact_updates_state(make_cfg):
    cfg = make_cfg()
    provider = MockProvider([
        turn(content="<analysis>draft</analysis><summary>Keep Decimal decision.</summary>")
    ])
    session = AgentSession.fresh(cfg, provider)
    for i in range(10):
        session.messages.append({"role": "user", "content": f"user {i}"})
        session.messages.append({"role": "assistant", "content": f"assistant {i}"})

    edit = session.compact_context("preserve Decimal rule")

    assert edit and edit["kind"] == "manual_summary_compact"
    assert "Keep Decimal decision" in session.messages[2]["content"]
    assert session.context_status()["last_compact_kind"] == "manual_summary_compact"
    assert session.usage_total.completion_tokens == 50


def test_session_app_state_and_run_start_include_settings_features(make_cfg, tmp_path):
    workdir = tmp_path / "ws"
    settings = workdir / ".tiny-harness" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({
        "features": {"coding_acceptance_trace": True}
    }), encoding="utf-8")
    cfg = make_cfg(workdir=workdir)
    cfg = cfg.from_env(workdir=workdir, runs_dir=cfg.runs_dir)
    provider = MockProvider([turn(content="ok")])
    session = AgentSession.fresh(cfg, provider)

    result = session.submit("Say ok.")

    state = session.app_state.get_state()
    assert state.status == "completed"
    assert state.features["coding_acceptance_trace"] is True
    assert state.memory["enabled"] is True
    events = read_trajectory(cfg.runs_dir, result.run_id)
    start = events[0]
    assert start["settings_sources"][0]["source"] == "projectSettings"
    assert start["features"]["coding_acceptance_trace"] is True
    assert start["memory"]["enabled"] is True


def test_session_permission_context_survives_between_submits(make_cfg):
    cfg = make_cfg()
    command = '{"command": "python -c \\"print(1)\\""}'
    provider = MockProvider([
        turn(calls=[("c1", "bash", command)]),
        turn(content="first"),
        turn(calls=[("c2", "bash", command)]),
        turn(content="second"),
    ])
    session = AgentSession.fresh(cfg, provider)
    session.permission_resolver = lambda *_args: "s"

    first = session.submit("Run the probe once.")
    second = session.submit("Run the same probe again.")

    assert any(e["type"] == "tool_permission_resolved"
               and e["resolver"] == "tui" for e in first.events)
    assert any(e["type"] == "tool_permission_update"
               and not e["persisted"] for e in first.events)
    assert any(e["type"] == "tool_permission"
               and e["reason_type"] == "rule" for e in second.events)
    assert not any(e["type"] == "tool_permission_wait" for e in second.events)
