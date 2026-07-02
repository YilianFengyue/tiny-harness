"""协议正确性测试：用 MockProvider 验证 loop 对 OpenAI 协议的每条铁律。

这些用例对应真实 API 会以 400 惩罚的违规行为，离线即可全部验证。
"""
import json

from conftest import MockProvider, turn

from harness.loop import build_resume_messages, run_agent
from harness.telemetry import read_trajectory


def run(cfg, logger, provider, task="test task"):
    return run_agent(task, cfg, provider, logger)


def tool_msgs_after_assistant(request_messages):
    """取最后一条 assistant 消息之后的所有 role=tool 消息。"""
    last_a = max(i for i, m in enumerate(request_messages) if m["role"] == "assistant")
    return request_messages[last_a], [m for m in request_messages[last_a + 1:]
                                      if m["role"] == "tool"]


def test_parallel_tool_calls_all_answered_in_one_round(make_cfg, make_logger):
    """一轮 3 个并行 tool_call → 下次请求里必须有 3 条按序对应的 tool 消息。"""
    provider = MockProvider([
        turn(calls=[("c1", "calculator", '{"expression": "1+1"}'),
                    ("c2", "calculator", '{"expression": "2*3"}'),
                    ("c3", "list_files", '{"path": null}')]),
        turn(content="done"),
    ])
    s = run(make_cfg(), make_logger(), provider)
    assert s["reason"] == "completed"
    assistant, tools = tool_msgs_after_assistant(provider.requests[1])
    assert [tc["id"] for tc in assistant["tool_calls"]] == ["c1", "c2", "c3"]
    assert [m["tool_call_id"] for m in tools] == ["c1", "c2", "c3"]
    assert "2" in tools[0]["content"] and "6" in tools[1]["content"]


def test_tool_error_is_returned_not_fatal(make_cfg, make_logger):
    """读不存在的文件：错误回传（ERROR: 前缀 + 可操作信息），循环继续。"""
    provider = MockProvider([
        turn(calls=[("c1", "read_file",
                     '{"path": "ghost.csv", "offset": null, "max_lines": null}')]),
        turn(content="recovered"),
    ])
    s = run(make_cfg(), make_logger(), provider)
    assert s["reason"] == "completed"
    _, tools = tool_msgs_after_assistant(provider.requests[1])
    assert tools[0]["content"].startswith("ERROR:")
    assert "not found" in tools[0]["content"]


def test_malformed_json_arguments_still_answered(make_cfg, make_logger):
    """模型吐非法 JSON 参数：该 id 仍必须被应答，且错误要求重发。"""
    provider = MockProvider([
        turn(calls=[("c1", "calculator", '{"expression": "1+1"')]),  # 缺右括号
        turn(content="done"),
    ])
    s = run(make_cfg(), make_logger(), provider)
    assert s["reason"] == "completed"
    _, tools = tool_msgs_after_assistant(provider.requests[1])
    assert tools[0]["tool_call_id"] == "c1"
    assert "invalid JSON" in tools[0]["content"]


def test_unknown_tool_name_answered_with_available_list(make_cfg, make_logger):
    provider = MockProvider([
        turn(calls=[("c1", "make_coffee", '{}')]),
        turn(content="ok"),
    ])
    s = run(make_cfg(), make_logger(), provider)
    assert s["reason"] == "completed"
    _, tools = tool_msgs_after_assistant(provider.requests[1])
    assert "Unknown tool" in tools[0]["content"] and "calculator" in tools[0]["content"]


def test_dangerous_command_denied_loop_continues(make_cfg, make_logger):
    """非交互环境下危险命令被拒，拒绝理由回传模型（而非崩溃/静默放行）。"""
    provider = MockProvider([
        turn(calls=[("c1", "bash", '{"command": "sudo rm -rf /"}')]),
        turn(content="I will not do that"),
    ])
    s = run(make_cfg(yolo=False), make_logger(), provider)
    assert s["reason"] == "completed"
    _, tools = tool_msgs_after_assistant(provider.requests[1])
    assert "blocked by safety policy" in tools[0]["content"]


def test_max_turns_stops_runaway_loop(make_cfg, make_logger):
    provider = MockProvider([
        turn(calls=[(f"c{i}", "calculator", '{"expression": "1+1"}')])
        for i in range(10)
    ])
    s = run(make_cfg(max_turns=3), make_logger(), provider)
    assert s["reason"] == "max_turns"
    assert len(provider.requests) == 3


def test_max_cost_circuit_breaker(make_cfg, make_logger):
    # 每轮 200K input + 30K output(gpt-5.5) ≈ $1.9，第二轮前熔断
    provider = MockProvider([
        turn(calls=[(f"c{i}", "calculator", '{"expression": "1+1"}')],
             prompt_tokens=200_000, completion=30_000)
        for i in range(5)
    ])
    s = run(make_cfg(max_cost_usd=1.0), make_logger(), provider)
    assert s["reason"] == "max_cost"
    assert len(provider.requests) == 1


def test_length_is_not_completion(make_cfg, make_logger):
    """finish_reason=length：先 nudge 续写，再次截断才以 truncated 结束。"""
    provider = MockProvider([
        turn(content="partial...", finish="length"),
        turn(content="full answer"),
    ])
    s = run(make_cfg(), make_logger(), provider)
    assert s["reason"] == "completed"
    nudge = provider.requests[1][-1]
    assert nudge["role"] == "user" and "cut off" in nudge["content"]

    provider2 = MockProvider([
        turn(content="partial...", finish="length"),
        turn(content="still partial", finish="length"),
    ])
    s2 = run(make_cfg(), make_logger(), provider2)
    assert s2["reason"] == "truncated"


def test_trajectory_is_complete_and_ordered(make_cfg, make_logger):
    cfg, logger = make_cfg(), make_logger()
    run_id, runs_dir = logger.run_id, cfg.runs_dir
    provider = MockProvider([
        turn(calls=[("c1", "calculator", '{"expression": "40+2"}')]),
        turn(content="answer is 42"),
    ])
    run(cfg, logger, provider)
    events = read_trajectory(runs_dir, run_id)
    types = [e["type"] for e in events]
    assert types == ["run_start",
                     "turn_start", "llm_request", "stream_request_start",
                     "llm_response", "tool_call", "tool_start",
                     "tool_progress", "tool_result", "tool_end", "transition",
                     "turn_start", "llm_request", "stream_request_start",
                     "assistant_delta", "llm_response", "run_end"]
    assert all(e["run_id"] == run_id for e in events)
    assert [e["step"] for e in events] == list(range(len(events)))
    transitions = [e for e in events if e["type"] == "transition"]
    assert transitions == [{
        **{k: transitions[0][k] for k in ("run_id", "step", "ts", "type")},
        "turn": 1,
        "kind": "continue",
        "reason": "next_turn",
        "tool_calls": 1,
    }]
    turn_starts = [e for e in events if e["type"] == "turn_start"]
    assert [e["transition"] for e in turn_starts] == [None, "next_turn"]
    assert [e["name"] for e in events if e["type"] == "tool_start"] == ["calculator"]
    assert [e["name"] for e in events if e["type"] == "tool_end"] == ["calculator"]
    assert [e["content"] for e in events if e["type"] == "assistant_delta"] == ["answer is 42"]
    end = events[-1]
    assert end["reason"] == "completed" and end["usage_total"]["prompt_tokens"] == 200


def test_length_recovery_is_explicit_transition(make_cfg, make_logger):
    cfg, logger = make_cfg(), make_logger()
    provider = MockProvider([
        turn(content="partial...", finish="length"),
        turn(content="full answer"),
    ])
    s = run(cfg, logger, provider)
    assert s["reason"] == "completed"
    events = read_trajectory(cfg.runs_dir, logger.run_id)
    transitions = [e for e in events if e["type"] == "transition"]
    assert [(e["reason"], e.get("attempt")) for e in transitions] == [("output_recovery", 1)]
    turn_starts = [e for e in events if e["type"] == "turn_start"]
    assert [e["transition"] for e in turn_starts] == [None, "output_recovery"]
    assert provider.requests[1][-1]["role"] == "user"
    assert "cut off" in provider.requests[1][-1]["content"]


def test_second_length_terminal_is_truncated_without_fake_completion(make_cfg, make_logger):
    cfg, logger = make_cfg(), make_logger()
    provider = MockProvider([
        turn(content="partial...", finish="length"),
        turn(content="still partial", finish="length"),
    ])
    s = run(cfg, logger, provider)
    assert s["reason"] == "truncated"
    events = read_trajectory(cfg.runs_dir, logger.run_id)
    assert [e["reason"] for e in events if e["type"] == "transition"] == ["output_recovery"]
    assert events[-1]["type"] == "run_end"
    assert events[-1]["reason"] == "truncated"
    assert events[-1]["final_message"] == "still partial"


def test_resume_rebuilds_exact_message_state(make_cfg, make_logger):
    """resume = 最后一次请求的 messages + 其后的 assistant/tool 增量。"""
    cfg, logger = make_cfg(), make_logger()
    provider = MockProvider([
        turn(calls=[("c1", "calculator", '{"expression": "40+2"}')]),
        turn(content="answer is 42"),
    ])
    run(cfg, logger, provider)
    events = read_trajectory(cfg.runs_dir, logger.run_id)
    rebuilt = build_resume_messages(events)
    # 最后一次请求含 4 条消息，其后增量 1 条 assistant 最终回复
    assert [m["role"] for m in rebuilt] == ["system", "user", "assistant", "tool", "assistant"]
    assert rebuilt[-1]["content"] == "answer is 42"
    assert rebuilt[2]["tool_calls"][0]["id"] == "c1"
    assert json.loads(rebuilt[2]["tool_calls"][0]["function"]["arguments"])
