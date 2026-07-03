import json
import threading

from conftest import MockProvider, turn

from harness.hooks import (
    evaluate_tool_permission,
    gate_tool_call,
    resolve_permission_decision,
)
from harness.loop import run_agent
from harness.permissions import (
    PermissionContext,
    PermissionRule,
    PermissionRuleValue,
    PermissionUpdate,
    ResolveOnce,
    apply_permission_updates,
    load_permission_context,
    match_bash_rule,
    permission_rule_value_from_string,
    permission_rule_value_to_string,
    persist_permission_updates,
)
from harness.telemetry import read_trajectory
from harness.tools.registry import ToolContext


def test_resolve_once_claim_is_first_wins():
    resolver = ResolveOnce()
    winners = []

    def claim(owner):
        if resolver.claim(owner, {"owner": owner}):
            winners.append(owner)

    threads = [threading.Thread(target=claim, args=(f"owner-{i}",)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(winners) == 1
    assert resolver.claimed_by == winners[0]
    assert resolver.value == {"owner": winners[0]}
    assert not resolver.claim("late")


def test_permission_rule_parse_round_trips_escaped_content():
    value = permission_rule_value_from_string(r"Bash(npm run test\(unit\))")
    assert value.tool_name == "Bash"
    assert value.rule_content == "npm run test(unit)"
    assert permission_rule_value_to_string(value) == r"Bash(npm run test\(unit\))"


def test_bash_wildcard_trailing_space_star_is_optional_subcommand():
    assert match_bash_rule("git:*", "git")
    assert match_bash_rule("git:*", "git status")
    assert match_bash_rule("npm *", "npm")
    assert match_bash_rule("npm *", "npm test")
    assert match_bash_rule("npm run *", "npm run")
    assert match_bash_rule("npm run *", "npm run build")
    assert not match_bash_rule("npm run *", "npm test")


def test_apply_updates_changes_memory_before_persist(tmp_path):
    workdir = tmp_path / "ws"
    workdir.mkdir()
    context = PermissionContext()
    updates = (
        PermissionUpdate("addRules", "local", "allow",
                         (PermissionRuleValue("Bash", "npm *"),)),
        PermissionUpdate("setMode", "session", mode="acceptEdits"),
    )

    in_memory = apply_permission_updates(context, updates)
    assert in_memory.mode == "acceptEdits"
    assert len(in_memory.rules) == 1
    assert not (workdir / ".tiny-harness" / "settings.local.json").exists()

    persist_permission_updates(workdir, updates)
    loaded = load_permission_context(workdir)
    assert loaded.mode == "default"
    assert any(rule.behavior == "allow" and rule.value.rule_content == "npm *"
               for rule in loaded.rules)


def test_persist_updates_preserves_unknown_settings_keys(tmp_path):
    workdir = tmp_path / "ws"
    settings = workdir / ".tiny-harness" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"theme": "quiet", "permissions": {"deny": []}}),
                        encoding="utf-8")

    persist_permission_updates(workdir, (
        PermissionUpdate("addRules", "project", "deny",
                         (PermissionRuleValue("Bash", "rm *"),)),
    ))

    data = json.loads(settings.read_text(encoding="utf-8"))
    assert data["theme"] == "quiet"
    assert data["permissions"]["deny"] == ["Bash(rm *)"]


def test_deny_and_ask_rules_precede_yolo_and_allow(make_cfg, tmp_path):
    workdir = tmp_path / "ws"
    (workdir / ".tiny-harness").mkdir(parents=True)
    (workdir / ".tiny-harness" / "settings.json").write_text(json.dumps({
        "permissions": {
            "allow": ["Bash(npm *)"],
            "ask": ["Bash(npm test)"],
            "deny": ["Bash(rm *)"],
        }
    }), encoding="utf-8")
    cfg = make_cfg(workdir=workdir, yolo=True, permission_mode="bypass")
    ctx = ToolContext(workdir=workdir)

    asked = evaluate_tool_permission("bash", {"command": "npm test"}, cfg, ctx)
    denied = gate_tool_call("bash", {"command": "rm tmp.txt"}, cfg, ctx)
    allowed = gate_tool_call("bash", {"command": "npm run lint"}, cfg, ctx)

    assert asked.behavior == "ask"
    assert denied.behavior == "deny"
    assert allowed.behavior == "allow"


def test_sensitive_path_safety_survives_bypass_and_yolo(make_cfg, tmp_path):
    workdir = tmp_path / "ws"
    workdir.mkdir()
    cfg = make_cfg(workdir=workdir, yolo=True, permission_mode="bypass")
    decision = gate_tool_call("write_file", {"path": ".env", "content": "SECRET=1"},
                              cfg, ToolContext(workdir=workdir))

    assert decision.behavior == "deny"
    assert decision.safety_check
    assert decision.reason_type == "safety_check"


def test_settings_mode_applies_when_cli_mode_is_default(make_cfg, tmp_path):
    workdir = tmp_path / "ws"
    (workdir / ".tiny-harness").mkdir(parents=True)
    (workdir / ".tiny-harness" / "settings.json").write_text(json.dumps({
        "permissions": {"mode": "acceptEdits"}
    }), encoding="utf-8")
    cfg = make_cfg(workdir=workdir)
    decision = gate_tool_call("write_file", {"path": "note.txt", "content": "hello"},
                              cfg, ToolContext(workdir=workdir))

    assert decision.allowed
    assert decision.reason_type == "mode"
    assert decision.mode == "acceptEdits"


def test_settings_json_allows_utf8_bom_from_powershell(make_cfg, tmp_path):
    workdir = tmp_path / "ws"
    settings = workdir / ".tiny-harness" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({
        "permissions": {
            "mode": "acceptEdits",
            "allow": ["Bash(python *)"],
            "ask": ["Bash(python -c *)"],
        }
    }), encoding="utf-8-sig")
    cfg = make_cfg(workdir=workdir)
    ctx = ToolContext(workdir=workdir)

    edit_decision = gate_tool_call("write_file", {"path": "note.txt", "content": "ok"},
                                   cfg, ctx)
    allowed = evaluate_tool_permission("bash", {"command": "python check.py"}, cfg, ctx)
    asked = evaluate_tool_permission("bash", {"command": "python -c \"print(1)\""},
                                     cfg, ctx)

    assert edit_decision.allowed
    assert edit_decision.mode == "acceptEdits"
    assert allowed.allowed and allowed.rule == "Bash(python *)"
    assert asked.behavior == "ask" and asked.rule == "Bash(python -c *)"


def test_passthrough_becomes_denied_ask_in_noninteractive_loop(make_cfg, make_logger):
    provider = MockProvider([
        turn(calls=[("c1", "write_file", '{"path": "note.txt", "content": "hello"}')]),
        turn(content="blocked"),
    ])

    summary = run_agent("write a file", make_cfg(), provider, make_logger())

    assert summary["reason"] == "completed"
    _, tool_message = provider.requests[1][-2], provider.requests[1][-1]
    assert tool_message["role"] == "tool"
    assert "blocked by safety policy" in tool_message["content"]


def test_tool_permission_event_explains_rule_and_mode(make_cfg, make_logger, tmp_path):
    workdir = tmp_path / "ws"
    (workdir / ".tiny-harness").mkdir(parents=True)
    (workdir / ".tiny-harness" / "settings.json").write_text(json.dumps({
        "permissions": {"deny": ["Bash(rm *)"]}
    }), encoding="utf-8")
    cfg = make_cfg(workdir=workdir, permission_mode="bypass")
    logger = make_logger()
    provider = MockProvider([
        turn(calls=[("c1", "bash", '{"command": "rm tmp.txt"}')]),
        turn(content="blocked"),
    ])

    run_agent("try denied command", cfg, provider, logger)
    events = read_trajectory(cfg.runs_dir, logger.run_id)
    permission = next(e for e in events if e["type"] == "tool_permission")

    assert permission["ok"] is False
    assert permission["decision"] == "deny"
    assert permission["reason_type"] == "rule"
    assert permission["rule"] == "Bash(rm *)"
    assert permission["mode"] == "bypass"


def test_ask_permission_emits_wait_and_resolved_events(make_cfg, make_logger, tmp_path):
    workdir = tmp_path / "ws"
    (workdir / ".tiny-harness").mkdir(parents=True)
    (workdir / ".tiny-harness" / "settings.json").write_text(json.dumps({
        "permissions": {"ask": ["Bash(python -c *)"]}
    }), encoding="utf-8")
    cfg = make_cfg(workdir=workdir)
    logger = make_logger()
    provider = MockProvider([
        turn(calls=[("c1", "bash", '{"command": "python -c \\"print(1)\\""}')]),
        turn(content="blocked"),
    ])

    run_agent("try ask command", cfg, provider, logger)
    events = read_trajectory(cfg.runs_dir, logger.run_id)

    assert any(e["type"] == "tool_permission_wait" and e["reason_type"] == "rule"
               for e in events)
    assert any(e["type"] == "tool_permission_resolved"
               and e["resolver"] == "noninteractive"
               and e["decision"] == "deny" for e in events)


def test_interactive_session_allow_updates_memory_context(make_cfg, monkeypatch, tmp_path):
    workdir = tmp_path / "ws"
    workdir.mkdir()
    cfg = make_cfg(workdir=workdir)
    ctx = ToolContext(workdir=workdir)
    decision = evaluate_tool_permission("bash", {"command": "python check.py"}, cfg, ctx)

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "s")
    final, events = resolve_permission_decision(
        "bash", {"command": "python check.py"}, cfg, ctx, decision)
    second = evaluate_tool_permission("bash", {"command": "python check.py"}, cfg, ctx)

    assert final.allowed
    assert any(e["type"] == "tool_permission_update" and not e["persisted"]
               for e in events)
    assert second.allowed
    assert second.reason_type == "rule"
