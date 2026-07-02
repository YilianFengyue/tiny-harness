import time

from conftest import MockProvider, turn

from harness.loop import run_agent
from harness.telemetry import read_trajectory
from harness.tools.registry import REGISTRY, ToolResult, ToolRuntimeState, tool


def _install_test_tool(name, **meta):
    def deco(fn):
        return tool(
            name=name,
            description=f"test tool {name}",
            parameters={"type": "object", "properties": {}},
            **meta,
        )(fn)
    return deco


def test_context_modifier_updates_runtime_for_later_tool(make_cfg, make_logger):
    saved = dict(REGISTRY)
    try:
        @_install_test_tool("set_runtime_flag")
        def set_runtime_flag(ctx):
            def modifier(runtime: ToolRuntimeState):
                runtime.file_history.append({"flag": "ready"})
                return {"kind": "test_flag", "flag": "ready"}
            return ToolResult("flag set", context_modifier=modifier)

        @_install_test_tool("read_runtime_flag", read_only=True, concurrency_safe=True)
        def read_runtime_flag(ctx):
            flag = ctx.runtime.file_history[-1]["flag"]
            return f"flag={flag}"

        cfg, logger = make_cfg(), make_logger()
        provider = MockProvider([
            turn(calls=[
                ("c1", "set_runtime_flag", "{}"),
                ("c2", "read_runtime_flag", "{}"),
            ]),
            turn(content="done"),
        ])

        summary = run_agent("test runtime", cfg, provider, logger)
        events = read_trajectory(cfg.runs_dir, logger.run_id)

        assert summary["reason"] == "completed"
        assert any(e["type"] == "tool_context_modified" and e["kind"] == "test_flag"
                   for e in events)
        assert any(e["type"] == "tool_result" and e["name"] == "read_runtime_flag"
                   and "flag=ready" in e["result"] for e in events)
    finally:
        REGISTRY.clear()
        REGISTRY.update(saved)


def test_concurrency_safe_batch_starts_before_results(make_cfg, make_logger):
    saved = dict(REGISTRY)
    try:
        @_install_test_tool("slow_safe", read_only=True, concurrency_safe=True)
        def slow_safe(ctx):
            time.sleep(0.05)
            return "safe"

        cfg, logger = make_cfg(), make_logger()
        provider = MockProvider([
            turn(calls=[
                ("c1", "slow_safe", "{}"),
                ("c2", "slow_safe", "{}"),
            ]),
            turn(content="done"),
        ])

        run_agent("test safe concurrency", cfg, provider, logger)
        events = read_trajectory(cfg.runs_dir, logger.run_id)
        interesting = [e for e in events if e["type"] in ("tool_start", "tool_result")]

        assert [e["type"] for e in interesting[:4]] == [
            "tool_start", "tool_start", "tool_result", "tool_result"
        ]
        assert [e["tool_call_id"] for e in interesting if e["type"] == "tool_result"] == [
            "c1", "c2"
        ]
    finally:
        REGISTRY.clear()
        REGISTRY.update(saved)


def test_unsafe_tools_are_serialized(make_cfg, make_logger):
    saved = dict(REGISTRY)
    try:
        @_install_test_tool("slow_unsafe")
        def slow_unsafe(ctx):
            time.sleep(0.01)
            return "unsafe"

        cfg, logger = make_cfg(), make_logger()
        provider = MockProvider([
            turn(calls=[
                ("c1", "slow_unsafe", "{}"),
                ("c2", "slow_unsafe", "{}"),
            ]),
            turn(content="done"),
        ])

        run_agent("test unsafe serial", cfg, provider, logger)
        events = read_trajectory(cfg.runs_dir, logger.run_id)
        interesting = [e for e in events if e["type"] in ("tool_start", "tool_result")]

        assert [e["type"] for e in interesting[:4]] == [
            "tool_start", "tool_result", "tool_start", "tool_result"
        ]
    finally:
        REGISTRY.clear()
        REGISTRY.update(saved)
