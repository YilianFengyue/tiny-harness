import json

from harness.config import Config
from harness.settings import (
    load_settings,
    merge_settings,
    parse_setting_sources_flag,
    read_settings_file,
    trusted_security_settings,
)


def test_settings_merge_arrays_objects_and_scalars():
    merged = merge_settings(
        {
            "model": "mini",
            "permissions": {"allow": ["Bash(ls)"], "deny": ["Bash(rm *)"]},
            "env": {"A": "1"},
        },
        {
            "model": "full",
            "permissions": {"allow": ["Bash(ls)", "Read(*)"]},
            "env": {"B": "2"},
        },
    )

    assert merged["model"] == "full"
    assert merged["permissions"]["allow"] == ["Bash(ls)", "Read(*)"]
    assert merged["permissions"]["deny"] == ["Bash(rm *)"]
    assert merged["env"] == {"A": "1", "B": "2"}


def test_load_settings_uses_source_order_and_policy_wins(tmp_path, monkeypatch):
    workdir = tmp_path / "ws"
    project = workdir / ".tiny-harness"
    project.mkdir(parents=True)
    home = tmp_path / "home"
    home.mkdir()
    managed = tmp_path / "managed"
    (managed / "managed-settings.d").mkdir(parents=True)
    monkeypatch.setenv("TINY_HARNESS_CONFIG_HOME", str(home))
    monkeypatch.setenv("TINY_HARNESS_MANAGED_SETTINGS_PATH", str(managed))

    (home / "settings.json").write_text(json.dumps({
        "model": "user-model",
        "permissions": {"allow": ["Bash(npm *)"]},
    }), encoding="utf-8")
    (project / "settings.json").write_text(json.dumps({
        "permissions": {"allow": ["Read(*)"]},
        "env": {"A": "project"},
    }), encoding="utf-8")
    (project / "settings.local.json").write_text(json.dumps({
        "model": "local-model",
        "env": {"A": "local"},
    }), encoding="utf-8")
    flag = tmp_path / "flag.json"
    flag.write_text(json.dumps({"max_turns": 12}), encoding="utf-8")
    (managed / "managed-settings.json").write_text(json.dumps({
        "model": "policy-model",
        "permissions": {"deny": ["Bash(rm *)"]},
    }), encoding="utf-8")
    (managed / "managed-settings.d" / "20-extra.json").write_text(json.dumps({
        "permissions": {"allow": ["Bash(python *)"]},
    }), encoding="utf-8")

    snapshot = load_settings(workdir, flag_settings_path=flag)

    assert [layer.source for layer in snapshot.sources] == [
        "userSettings",
        "projectSettings",
        "localSettings",
        "flagSettings",
        "policySettings",
    ]
    assert snapshot.effective["model"] == "policy-model"
    assert snapshot.effective["max_turns"] == 12
    assert snapshot.effective["env"]["A"] == "local"
    assert snapshot.effective["permissions"]["allow"] == [
        "Bash(npm *)",
        "Read(*)",
        "Bash(python *)",
    ]
    assert snapshot.effective["permissions"]["deny"] == ["Bash(rm *)"]
    assert snapshot.policy_origin == "file"


def test_policy_env_source_wins_over_managed_files(tmp_path, monkeypatch):
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("TINY_HARNESS_MANAGED_SETTINGS_PATH", str(managed))
    monkeypatch.setenv("TINY_HARNESS_POLICY_SETTINGS_JSON",
                       json.dumps({"model": "env-policy"}))
    (managed / "managed-settings.json").write_text(
        json.dumps({"model": "file-policy"}), encoding="utf-8")

    snapshot = load_settings(tmp_path / "ws")

    assert snapshot.effective["model"] == "env-policy"
    assert snapshot.policy_origin == "env"


def test_settings_file_reads_utf8_sig(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"model": "bom-ok"}), encoding="utf-8-sig")

    data, errors = read_settings_file(path, "userSettings")

    assert errors == []
    assert data["model"] == "bom-ok"


def test_trusted_security_settings_excludes_project(tmp_path):
    workdir = tmp_path / "ws"
    settings_dir = workdir / ".tiny-harness"
    settings_dir.mkdir(parents=True)
    (settings_dir / "settings.json").write_text(json.dumps({
        "skipDangerousModePermissionPrompt": True,
    }), encoding="utf-8")

    snapshot = load_settings(workdir)

    assert snapshot.effective["skipDangerousModePermissionPrompt"] is True
    assert trusted_security_settings(snapshot).get(
        "skipDangerousModePermissionPrompt") is None


def test_config_from_env_applies_settings_then_env_then_explicit(tmp_path, monkeypatch):
    workdir = tmp_path / "ws"
    settings_dir = workdir / ".tiny-harness"
    settings_dir.mkdir(parents=True)
    (settings_dir / "settings.json").write_text(json.dumps({
        "model": "project-model",
        "max_turns": 5,
        "permissions": {"mode": "acceptEdits"},
    }), encoding="utf-8")
    monkeypatch.setenv("TINY_HARNESS_MODEL", "env-model")

    cfg = Config.from_env(workdir=workdir, max_turns=9)

    assert cfg.model == "env-model"
    assert cfg.max_turns == 9
    assert cfg.permission_mode == "acceptEdits"
    assert cfg.settings_snapshot is not None


def test_parse_setting_sources_flag():
    assert parse_setting_sources_flag("user, project,local") == (
        "userSettings", "projectSettings", "localSettings")
