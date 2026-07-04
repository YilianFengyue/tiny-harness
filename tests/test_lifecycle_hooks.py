import pytest

from conftest import MockProvider, turn

from harness.config import Config
from harness.lifecycle_hooks import (
    SUPPORTED_LIFECYCLE_EVENTS,
    dispatch_lifecycle_hooks,
)
from harness.loop import run_agent
from harness.telemetry import RunLogger, read_trajectory


def test_lifecycle_hook_contract_starts_as_noop_dispatcher():
    result = dispatch_lifecycle_hooks(
        "PreToolUse",
        {"tool_name": "bash", "tool_input": {"command": "python -m pytest -q"}},
    )

    assert "SessionStart" in SUPPORTED_LIFECYCLE_EVENTS
    assert "Stop" in SUPPORTED_LIFECYCLE_EVENTS
    assert result.event == "PreToolUse"
    assert result.blocked is False
    assert result.updated_input is None
    assert result.additional_context is None


def test_lifecycle_hook_rejects_unknown_event():
    with pytest.raises(ValueError):
        dispatch_lifecycle_hooks("UnknownEvent")  # type: ignore[arg-type]


def test_command_hook_can_update_tool_input(tmp_path):
    cfg = _cfg_with_hook(
        tmp_path,
        "PreToolUse",
        "Bash",
        """
import json
print(json.dumps({"hookSpecificOutput": {"updatedInput": {"command": "python -c \\"print(42)\\""}}}))
""",
    )

    result = dispatch_lifecycle_hooks(
        "PreToolUse",
        {"tool_name": "bash", "tool_input": {"command": "python -c \"print(1)\""}},
        cfg,
    )

    assert result.updated_input == {"command": "python -c \"print(42)\""}
    assert [d.event for d in result.diagnostics] == ["PreToolUse", "PreToolUse"]


def test_pretooluse_hook_blocks_tool_execution(tmp_path):
    cfg = _cfg_with_hook(
        tmp_path,
        "PreToolUse",
        "Bash",
        """
import json
print(json.dumps({"decision": "block", "reason": "demo hook blocks bash"}))
""",
    )
    provider = MockProvider([
        turn(calls=[("c1", "bash", '{"command": "python -c \\"print(1)\\""}')]),
        turn(content="I saw the hook block."),
    ])
    logger = RunLogger(cfg.runs_dir)

    summary = run_agent("Try bash.", cfg, provider, logger)

    events = read_trajectory(cfg.runs_dir, summary["run_id"])
    assert summary["reason"] == "completed"
    assert any(e["type"] == "hook_end" and e["hook_event"] == "PreToolUse"
               and e["blocked"] for e in events)
    assert any(e["type"] == "tool_result" and e["error_kind"] == "hook_blocked"
               for e in events)
    assert not any(e["type"] == "tool_start" for e in events)


def test_stop_hook_adds_context_and_continues_once(tmp_path):
    cfg = _cfg_with_hook(
        tmp_path,
        "Stop",
        "*",
        """
import json
print(json.dumps({"additionalContext": "Mention hook-ok in the final answer."}))
""",
    )
    provider = MockProvider([
        turn(content="Initial final."),
        turn(content="hook-ok final."),
    ])
    logger = RunLogger(cfg.runs_dir)

    summary = run_agent("Finish once.", cfg, provider, logger)

    assert summary["final_message"] == "hook-ok final."
    assert provider.requests[1][-1]["role"] == "user"
    assert "Mention hook-ok" in provider.requests[1][-1]["content"]


def _cfg_with_hook(tmp_path, event: str, matcher: str, script: str) -> Config:
    workdir = tmp_path / "ws"
    settings_dir = workdir / ".tiny-harness"
    hook_dir = settings_dir / "hooks"
    hook_dir.mkdir(parents=True)
    script_path = hook_dir / f"{event.lower()}.py"
    script_path.write_text(script.strip() + "\n", encoding="utf-8")
    (settings_dir / "settings.local.json").write_text(
        json_text({
            "hooksTrusted": True,
            "hooks": {
                event: [{
                    "matcher": matcher,
                    "hooks": [{
                        "type": "command",
                        "command": f"python {script_path.relative_to(workdir).as_posix()}",
                        "timeout": 5,
                    }],
                }],
            },
        }),
        encoding="utf-8",
    )
    return Config.from_env(workdir=workdir, runs_dir=tmp_path / "runs")


def json_text(data: dict) -> str:
    import json
    return json.dumps(data, ensure_ascii=False, indent=2)
