import json

from harness.config import Config
from harness.state import AppState, build_app_state, create_store


def test_store_notifies_only_when_reference_changes():
    state = AppState(model="m1")
    store = create_store(state)
    calls = []
    store.subscribe(lambda: calls.append(store.get_state()))

    store.set_state(lambda prev: prev)
    assert calls == []

    store.set_state(lambda prev: AppState(model=prev.model, status="running"))
    assert len(calls) == 1
    assert store.get_state().status == "running"


def test_build_app_state_includes_settings_sources_and_features(tmp_path):
    workdir = tmp_path / "ws"
    settings = workdir / ".tiny-harness" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({
        "features": {"app_state": True}
    }), encoding="utf-8")
    cfg = Config.from_env(workdir=workdir)

    state = build_app_state(cfg)

    assert state.model == cfg.model
    assert state.settings_sources == ("projectSettings",)
    assert state.features["app_state"] is True
