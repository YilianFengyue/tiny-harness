from conftest import MockProvider, turn

from harness.agents import agent_tool_names, get_agent_definition, parse_agent_markdown
from harness.loop import run_agent, tool_schemas_for_config
from harness.providers.base import ModelTurn, Provider
from harness.telemetry import read_trajectory
from harness.session import AgentSession
from harness.tools.registry import openai_tool_schemas


def test_builtin_agent_definitions_have_expected_tool_filters():
    explore = get_agent_definition("explore")
    general = get_agent_definition("general")

    assert explore is not None
    assert general is not None
    assert "read_file" in agent_tool_names(explore)
    assert "agent" not in agent_tool_names(explore)
    assert "agent" not in agent_tool_names(general)
    worker = get_agent_definition("worker")
    assert worker is not None
    assert "read_file" in agent_tool_names(worker)
    assert "agent" not in agent_tool_names(worker)


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


def test_agent_tool_schema_lists_project_agents(tmp_path):
    agents_dir = tmp_path / ".tiny-harness" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "audit-reviewer.md").write_text("""---
name: audit-reviewer
description: reviews audit semantics
tools: [read_file]
---
Review audit semantics.
""", encoding="utf-8")

    agent_schema = next(
        schema for schema in openai_tool_schemas(tmp_path)
        if schema["function"]["name"] == "agent"
    )
    description = agent_schema["function"]["description"]
    params = agent_schema["function"]["parameters"]["properties"]

    assert "audit-reviewer" in description
    assert "Project agents are loaded" in description
    assert "background" in params["run_in_background"]["description"]
    assert "Reserved" not in params["fork"]["description"]
    assert "agent_id" not in params


def test_coordinator_tool_schema_only_exposes_agent(make_cfg, tmp_path):
    schemas = tool_schemas_for_config(make_cfg(workdir=tmp_path, coordinator_mode=True))
    names = [schema["function"]["name"] for schema in schemas]
    agent_schema = next(schema for schema in schemas if schema["function"]["name"] == "agent")

    assert names == ["agent"]
    assert "read_file" in [schema["function"]["name"] for schema in openai_tool_schemas(tmp_path)]
    assert "worker" in agent_schema["function"]["description"]
    assert "agent_id" in agent_schema["function"]["parameters"]["properties"]


class ParentChildProvider(Provider):
    def __init__(self, parent_turns, child_turns):
        self.parent = MockProvider(parent_turns)
        self.child = MockProvider(child_turns)
        self.child_spawns = 0

    def complete(self, messages, tools, on_retry=None):
        return self.parent.complete(messages, tools, on_retry)

    def spawn_child(self):
        self.child_spawns += 1
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


def test_coordinator_forces_async_worker_and_xml_notification(make_cfg):
    cfg = make_cfg(coordinator_mode=True)
    child = BlockingChildProvider([turn(content="worker finished")])
    provider = ParentChildProvider(
        [
            turn(calls=[
                ("a1", "agent", '{"description":"research task","prompt":"inspect","subagent_type":null,"run_in_background":null,"fork":null}'),
            ]),
            turn(content="launched worker"),
            turn(content="integrated worker result"),
        ],
        [],
    )
    provider.child = child
    session = AgentSession.fresh(cfg, provider)

    first = session.submit("coordinate this")
    events = read_trajectory(cfg.runs_dir, first.run_id)
    child.release.set()
    for _ in range(100):
        if session.background_agents.list()[0].status != "running":
            break
        import time
        time.sleep(0.01)
    second = session.submit("continue")

    assert first.summary["reason"] == "completed"
    assert second.summary["reason"] == "completed"
    assert any(e["type"] == "run_start" and e["mode"] == "coordinator"
               for e in events)
    llm_request = next(e for e in events if e["type"] == "llm_request")
    assert llm_request["tools"] == ["agent"]
    assert any(e["type"] == "agent_background_start" and e["agent_type"] == "worker"
               for e in events)
    assert any("<task-notification>" in str(message.get("content"))
               for message in provider.parent.requests[-1])
    done = next(e for e in second.events if e["type"] == "agent_background_done")
    child_events = read_trajectory(cfg.runs_dir, done["run_id"])
    child_request = next(e for e in child_events if e["type"] == "llm_request")
    assert "read_file" in child_request["tools"]
    assert "agent" not in child_request["tools"]


def test_coordinator_send_message_resumes_same_worker_context(make_cfg):
    cfg = make_cfg(coordinator_mode=True)
    child = BlockingChildProvider([
        turn(content="first worker result with local context"),
        turn(content="second worker used prior context"),
    ])
    provider = ParentChildProvider(
        [
            turn(calls=[
                ("a1", "agent", '{"description":"research task","prompt":"inspect","subagent_type":null,"run_in_background":null,"fork":null,"agent_id":null}'),
            ]),
            turn(content="launched worker"),
        ],
        [],
    )
    provider.child = child
    session = AgentSession.fresh(cfg, provider)

    first = session.submit("coordinate this")
    record = session.background_agents.list()[0]
    worker_id = record.agent_id
    child.release.set()
    for _ in range(100):
        if record.status != "running":
            break
        import time
        time.sleep(0.01)
    first_worker_run = record.result.run_id
    provider.parent.turns.extend([
        turn(calls=[
            ("a2", "agent", '{"description":"follow up","prompt":"continue with the same findings","subagent_type":"worker","run_in_background":true,"fork":null,"agent_id":"' + worker_id + '"}'),
        ]),
        turn(content="resumed worker"),
    ])

    second = session.submit("ask same worker a follow-up")
    for _ in range(100):
        if record.status != "running":
            break
        import time
        time.sleep(0.01)

    assert first.summary["reason"] == "completed"
    assert second.summary["reason"] == "completed"
    assert provider.child_spawns == 1
    assert len(provider.child.requests) == 2
    second_child_request = provider.child.requests[1]
    assert any(m.get("role") == "assistant"
               and "first worker result with local context" in str(m.get("content"))
               for m in second_child_request)
    assert any("Coordinator SendMessage" in str(m.get("content"))
               and "continue with the same findings" in str(m.get("content"))
               for m in second_child_request)
    assert record.resume_count == 1
    assert record.result.run_id != first_worker_run
