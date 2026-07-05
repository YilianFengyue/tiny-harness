from conftest import MockProvider, turn

from harness.agents import agent_tool_names, get_agent_definition, parse_agent_markdown
from harness.loop import run_agent
from harness.providers.base import ModelTurn, Provider
from harness.telemetry import read_trajectory
from harness.session import AgentSession


def test_builtin_agent_definitions_have_expected_tool_filters():
    explore = get_agent_definition("explore")
    general = get_agent_definition("general")

    assert explore is not None
    assert general is not None
    assert "read_file" in agent_tool_names(explore)
    assert "agent" not in agent_tool_names(explore)
    assert "agent" not in agent_tool_names(general)


def test_agent_tool_runs_foreground_subagent(make_cfg, make_logger):
    cfg, logger = make_cfg(), make_logger()
    provider = MockProvider([
        turn(calls=[
            ("a1", "agent", '{"description":"inspect code","prompt":"find the issue","subagent_type":"explore","run_in_background":null,"fork":null}'),
        ]),
        turn(content="Scope: inspect code\nResult: found issue in app.py"),
        turn(content="parent done"),
    ])

    summary = run_agent("delegate investigation", cfg, provider, logger)
    events = read_trajectory(cfg.runs_dir, logger.run_id)

    assert summary["reason"] == "completed"
    assert any(e["type"] == "agent_start" and e["agent_type"] == "explore"
               for e in events)
    assert any(e["type"] == "agent_done" and e["agent_type"] == "explore"
               for e in events)
    assert any(e["type"] == "tool_result" and e["name"] == "agent"
               and "Subagent explore completed" in e["result"]
               for e in events)


def test_read_only_subagent_cannot_write_files(make_cfg, make_logger):
    cfg, logger = make_cfg(), make_logger()
    provider = MockProvider([
        turn(calls=[
            ("a1", "agent", '{"description":"verify safely","prompt":"try to write","subagent_type":"verify","run_in_background":null,"fork":null}'),
        ]),
        turn(calls=[
            ("w1", "write_file", '{"path":"should_not_exist.txt","content":"bad"}'),
        ]),
        turn(content="Write was blocked; verification stayed read-only."),
        turn(content="parent done"),
    ])

    summary = run_agent("delegate verification", cfg, provider, logger)
    events = read_trajectory(cfg.runs_dir, logger.run_id)

    assert summary["reason"] == "completed"
    assert not (cfg.workdir / "should_not_exist.txt").exists()
    subagent_run = next(e["run_id"] for e in events if e["type"] == "agent_done")
    sub_events = read_trajectory(cfg.runs_dir, subagent_run)
    assert any(e["type"] == "tool_result" and e["name"] == "write_file"
               and e["ok"] is False
               and "not available to this agent" in e["result"]
               for e in sub_events)


def test_custom_agent_markdown_parses(tmp_path):
    path = tmp_path / "doc-writer.md"
    path.write_text("""---
name: doc-writer
description: writes docs
tools: [read_file, write_file]
disallowedTools:
  - bash
maxTurns: 7
background: true
---
You write concise docs.
""", encoding="utf-8")

    agent = parse_agent_markdown(path)

    assert agent is not None
    assert agent.agent_type == "doc-writer"
    assert agent.tools == ("read_file", "write_file")
    assert agent.disallowed_tools == ("bash",)
    assert agent.max_turns == 7
    assert agent.background is True
    assert "concise docs" in agent.system_prompt


class ParentChildProvider(Provider):
    def __init__(self, parent_turns, child_turns):
        self.parent = MockProvider(parent_turns)
        self.child = MockProvider(child_turns)

    def complete(self, messages, tools, on_retry=None):
        return self.parent.complete(messages, tools, on_retry)

    def spawn_child(self):
        return self.child


class BlockingChildProvider(MockProvider):
    def __init__(self, turns):
        super().__init__(turns)
        import threading
        self.release = threading.Event()

    def complete(self, messages, tools, on_retry=None):
        self.release.wait(timeout=5)
        return super().complete(messages, tools, on_retry)


def test_fork_agent_inherits_parent_message_snapshot(make_cfg, make_logger):
    cfg, logger = make_cfg(), make_logger()
    provider = ParentChildProvider(
        [
            turn(calls=[
                ("a1", "agent", '{"description":"fork inspect","prompt":"use inherited context","subagent_type":"explore","run_in_background":null,"fork":true}'),
            ]),
            turn(content="parent done"),
        ],
        [turn(content="child saw context")],
    )

    summary = run_agent("parent context line", cfg, provider, logger)

    assert summary["reason"] == "completed"
    child_messages = provider.child.requests[0]
    assert any(m.get("role") == "user" and "parent context line" in str(m.get("content"))
               for m in child_messages)
    assert any("Fork subagent directive" in str(m.get("content"))
               for m in child_messages)


def test_background_agent_completion_is_injected_next_session_turn(make_cfg):
    cfg = make_cfg()
    child = BlockingChildProvider([turn(content="background child result")])
    provider = ParentChildProvider(
        [
            turn(calls=[
                ("a1", "agent", '{"description":"background inspect","prompt":"inspect later","subagent_type":"explore","run_in_background":true,"fork":null}'),
            ]),
            turn(content="background launched"),
            turn(content="integrated background result"),
        ],
        [],
    )
    provider.child = child
    session = AgentSession.fresh(cfg, provider)

    first = session.submit("launch background")
    child.release.set()
    for _ in range(100):
        if session.background_agents.list()[0].status != "running":
            break
        import time
        time.sleep(0.01)
    second = session.submit("check background")

    assert first.summary["reason"] == "completed"
    assert second.summary["reason"] == "completed"
    assert any("Background subagent completed" in str(message.get("content"))
               for message in provider.parent.requests[-1])
    assert "background child result" in session.agents_summary()
