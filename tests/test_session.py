from conftest import MockProvider, turn

from harness.session import AgentSession
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
    assert any(e["type"] == "transition" and e["reason"] == "next_turn"
               for e in events)


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
