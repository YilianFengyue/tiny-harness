from conftest import MockProvider, turn

from harness.session import AgentSession
from harness.telemetry import read_trajectory


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
