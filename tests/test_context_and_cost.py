from conftest import MockProvider, turn

from harness.config import load_pricing
from harness.context import ContextManager
from harness.loop import run_agent
from harness.telemetry import CostLedger, Usage, read_trajectory


def msg_tool(content):
    return {"role": "tool", "tool_call_id": "x", "content": content}


def test_compact_clears_old_tool_results_keeps_recent():
    cm = ContextManager(budget_tokens=1000, keep_recent=2)
    messages = ([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
                + [msg_tool("Z" * 1000) for _ in range(5)])
    cm.observe(5000)  # 超预算
    edit = cm.maybe_compact(messages)
    assert edit and edit["cleared_messages"] == 3
    cleared = [m for m in messages if m.get("_cleared")]
    assert len(cleared) == 3
    assert all("cleared to save context" in m["content"] for m in cleared)
    # 最近 2 条保持原样
    assert messages[-1]["content"] == "Z" * 1000 and messages[-2]["content"] == "Z" * 1000


def test_compact_noop_under_budget_and_skips_short():
    cm = ContextManager(budget_tokens=1000, keep_recent=0)
    messages = [msg_tool("short")]
    cm.observe(500)
    assert cm.maybe_compact(messages) is None      # 未超预算
    cm.observe(5000)
    assert cm.maybe_compact(messages) is None      # 超预算但全是短消息，清了无意义


def test_loop_emits_context_edit_event(make_cfg, make_logger):
    """真实 usage 推高水位 → 下一轮触发清理并写 context_edit 事件。"""
    cfg = make_cfg(context_budget=500, context_keep_recent=1)
    logger = make_logger()
    run_id, runs_dir = logger.run_id, cfg.runs_dir
    big = '{"expression": "' + "1+" * 200 + '1"}'
    provider = MockProvider([
        turn(calls=[("c1", "calculator", big)], prompt_tokens=100),
        turn(calls=[("c2", "calculator", big)], prompt_tokens=10_000),  # 推过预算
        turn(content="done", prompt_tokens=10_000),
    ])
    s = run_agent("t", cfg, provider, logger)
    assert s["reason"] == "completed"
    events = read_trajectory(runs_dir, run_id)
    edits = [e for e in events if e["type"] == "context_edit"]
    assert len(edits) >= 1
    # 清理在发给模型的消息里生效：第三次请求中第一条 tool 消息已是占位符
    third_req = [e for e in events if e["type"] == "llm_request"][2]
    tool_contents = [m["content"] for m in third_req["messages"] if m["role"] == "tool"]
    assert any("cleared to save context" in c for c in tool_contents)
    assert not any(m.get("_cleared") for m in third_req["messages"])  # 内部标记不上线


def test_cost_formula_with_cache():
    ledger = CostLedger(load_pricing())
    cost = ledger.record("gpt-5.5", Usage(
        prompt_tokens=10_000, cached_tokens=4_000,
        completion_tokens=1_000, reasoning_tokens=800))
    # (6000*5 + 4000*0.5 + 1000*30) / 1e6 = 0.062；reasoning 含在 completion 内不重复计费
    assert abs(cost - 0.062) < 1e-9
    assert ledger.total.reasoning_tokens == 800


def test_long_context_surcharge():
    ledger = CostLedger(load_pricing())
    cost = ledger.record("gpt-5.5", Usage(prompt_tokens=300_000, completion_tokens=1_000))
    # 超 272K：input x2, output x1.5 → (300000*5*2 + 1000*30*1.5) / 1e6 = 3.045
    assert abs(cost - 3.045) < 1e-9
    assert ledger.long_context_hits == 1


def test_unknown_model_marks_pricing_unknown():
    ledger = CostLedger(load_pricing())
    cost = ledger.record("mystery-model", Usage(prompt_tokens=1000, completion_tokens=10))
    assert cost == 0.0 and ledger.pricing_unknown  # 绝不静默编造成本
