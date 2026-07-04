from conftest import MockProvider, turn

from harness.memory import load_memory_records, memory_path_info
from harness.memory_extract import (
    MemoryExtractionController,
    extract_memory_candidates,
    has_direct_memory_write,
)
from harness.session import AgentSession
from harness.telemetry import read_trajectory


def test_extract_memory_candidates_classifies_project_reference_and_user():
    candidates = extract_memory_candidates([
        {"role": "user", "content": "以后这个项目里，金额统一用 Decimal，不要用 float。原因是财务报表不能有误差。"},
        {"role": "user", "content": "生产仪表盘链接 https://grafana.example.com/d/abc 请记住"},
        {"role": "user", "content": "我是后端工程师，第一次接触 React，请记住"},
    ])

    assert [candidate.type for candidate in candidates] == [
        "project",
        "reference",
        "user",
    ]
    assert "Decimal" in candidates[0].content


def test_direct_memory_write_detection():
    messages = [
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "write_memory", "arguments": "{}"}}
        ]},
    ]

    assert has_direct_memory_write(messages)


def test_controller_extracts_and_deduplicates(tmp_path, make_cfg):
    cfg = make_cfg(workdir=tmp_path / "ws", runs_dir=tmp_path / "runs")
    controller = MemoryExtractionController(cfg)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "以后这个项目里，金额统一用 Decimal，不要用 float。"},
        {"role": "assistant", "content": "记住了。"},
    ]

    first = controller.extract(messages, force=True)
    second = controller.extract(messages + [{"role": "user", "content": "以后这个项目里，金额统一用 Decimal，不要用 float。"}], force=True)

    assert any(event["type"] == "memory_extract_saved" for event in first)
    assert any(event.get("reason") == "duplicates" for event in second)
    records = load_memory_records(memory_path_info(cfg).directory)
    assert len(records) == 1
    assert records[0].type == "project"


def test_session_auto_extracts_after_completed_turn(tmp_path, make_cfg):
    cfg = make_cfg(workdir=tmp_path / "ws", runs_dir=tmp_path / "runs")
    provider = MockProvider([turn(content="ok")])
    session = AgentSession.fresh(cfg, provider)

    result = session.submit("以后这个项目里，金额统一用 Decimal，不要用 float。")

    records = load_memory_records(memory_path_info(cfg).directory)
    assert records
    assert records[0].type == "project"
    events = read_trajectory(cfg.runs_dir, result.run_id)
    assert any(event["type"] == "memory_extract_saved" for event in events)


def test_session_memory_auto_extract_can_be_disabled(tmp_path, make_cfg):
    cfg = make_cfg(workdir=tmp_path / "ws", runs_dir=tmp_path / "runs")
    provider = MockProvider([turn(content="ok")])
    session = AgentSession.fresh(cfg, provider)
    session.set_memory_auto_extract(False)

    session.submit("以后这个项目里，金额统一用 Decimal，不要用 float。")

    assert load_memory_records(memory_path_info(cfg).directory) == []
