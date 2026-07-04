import json

from harness.config import Config
from harness.features import feature, feature_snapshot, feature_value


def test_feature_snapshot_reads_settings_and_env_override(tmp_path, monkeypatch):
    workdir = tmp_path / "ws"
    settings = workdir / ".tiny-harness" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({
        "features": {
            "settings_tui": True,
            "slow_mode": False,
            "sample_rate": 0.25,
        }
    }), encoding="utf-8")
    monkeypatch.setenv("TINY_HARNESS_FEATURES", "slow_mode,-settings_tui,app_state")

    cfg = Config.from_env(workdir=workdir)

    assert feature_snapshot(cfg)["settings_tui"] is False
    assert feature("app_state", cfg)
    assert feature("slow_mode", cfg)
    assert feature_value("sample_rate", cfg=cfg, default=1.0) == 0.25
