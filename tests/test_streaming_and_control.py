from pathlib import Path

from conftest import MockProvider, turn

from harness.cancel import CancelledError
from harness.loop import run_agent
from harness.providers.base import ModelTurn, Provider
from harness.telemetry import read_trajectory


class PromptTooLongError(Exception):
    status_code = 400

    def __str__(self):
        return "context length exceeded: maximum context is too small"


class ReactiveProvider(Provider):
    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools, on_retry=None):
        raise AssertionError("stream path should be used")

    def stream(self, messages, tools, on_retry=None, cancel_token=None):
        self.calls += 1
        if self.calls == 1:
            raise PromptTooLongError()
        yield {"type": "assistant_delta", "content": "recovered"}
        return ModelTurn(content="recovered")


class CancelDuringStreamProvider(Provider):
    def complete(self, messages, tools, on_retry=None):
        raise AssertionError("stream path should be used")

    def stream(self, messages, tools, on_retry=None, cancel_token=None):
        yield {"type": "assistant_delta", "content": "partial"}
        raise CancelledError("cancelled while streaming")


def test_run_start_records_session_id(make_cfg, make_logger):
    cfg, logger = make_cfg(), make_logger()
    provider = MockProvider([turn(content="hello")])

    run_agent("hello", cfg, provider, logger)
    events = read_trajectory(cfg.runs_dir, logger.run_id)

    assert "session_id" in events[0]
    assert events[0]["session_id"] is None


def test_blocking_limit_stops_before_provider_call(make_cfg, make_logger):
    cfg, logger = make_cfg(context_hard_limit=50), make_logger()
    provider = MockProvider([])

    summary = run_agent("x" * 1000, cfg, provider, logger)
    events = read_trajectory(cfg.runs_dir, logger.run_id)

    assert summary["reason"] == "blocking_limit"
    assert not provider.requests
    assert any(e["type"] == "error" and e["where"] == "context" for e in events)


def test_prompt_too_long_reactive_compact_retries(make_cfg, make_logger):
    cfg, logger = make_cfg(context_hard_limit=1_000_000), make_logger()
    provider = ReactiveProvider()
    huge_tool = "z" * 30_000
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": huge_tool},
    ]

    summary = run_agent("continue", cfg, provider, logger, resume_messages=messages)
    events = read_trajectory(cfg.runs_dir, logger.run_id)

    assert summary["reason"] == "completed"
    assert provider.calls == 2
    assert any(e["type"] == "transition" and e["reason"] == "reactive_compact_retry"
               for e in events)
    assert any(e["type"] == "context_edit" and e["kind"] == "reactive_compact"
               for e in events)


def test_cancelled_stream_has_specific_terminal_reason(make_cfg, make_logger):
    cfg, logger = make_cfg(), make_logger()
    provider = CancelDuringStreamProvider()

    summary = run_agent("cancel me", cfg, provider, logger)
    events = read_trajectory(cfg.runs_dir, logger.run_id)

    assert summary["reason"] == "aborted_streaming"
    assert [e["content"] for e in events if e["type"] == "assistant_delta"] == ["partial"]
