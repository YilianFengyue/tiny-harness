import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.config import Config                      # noqa: E402
from harness.providers.base import ModelTurn, Provider, ToolCallRequest  # noqa: E402
from harness.telemetry import RunLogger, Usage         # noqa: E402
from harness.tools.registry import ToolContext         # noqa: E402


class MockProvider(Provider):
    """脚本化响应序列 + 记录收到的完整消息，供协议断言。"""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests: list[list[dict]] = []

    def complete(self, messages, tools, on_retry=None):
        self.requests.append([dict(m) for m in messages])
        if not self.turns:
            raise AssertionError("MockProvider exhausted: loop called more times than scripted")
        return self.turns.pop(0)


def turn(content=None, calls=None, finish=None, prompt_tokens=100,
         cached=0, completion=50, reasoning=0):
    """便捷构造 ModelTurn。calls: [(id, name, raw_json_args), ...]"""
    tool_calls = [ToolCallRequest.from_raw(i, n, a) for i, n, a in (calls or [])]
    return ModelTurn(
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish or ("tool_calls" if tool_calls else "stop"),
        usage=Usage(prompt_tokens=prompt_tokens, cached_tokens=cached,
                    completion_tokens=completion, reasoning_tokens=reasoning),
    )


@pytest.fixture
def ctx(tmp_path):
    work = tmp_path / "ws"
    work.mkdir()
    return ToolContext(workdir=work, bash_timeout=60, output_limit=20_000)


@pytest.fixture
def make_cfg(tmp_path):
    def _make(**kw):
        kw.setdefault("workdir", tmp_path / "ws")
        kw.setdefault("runs_dir", tmp_path / "runs")
        kw.setdefault("model", "gpt-5.5")
        return Config(**kw)
    return _make


@pytest.fixture
def make_logger(tmp_path):
    def _make():
        return RunLogger(tmp_path / "runs")
    return _make
