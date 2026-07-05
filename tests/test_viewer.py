from pathlib import Path


def test_viewer_knows_multi_agent_events_and_is_text():
    path = Path(__file__).resolve().parent.parent / "viewer" / "index.html"
    raw = path.read_bytes()
    text = raw.decode("utf-8")

    assert b"\x00" not in raw
    for event_type in (
        "agent_start",
        "agent_progress",
        "agent_done",
        "agent_error",
        "agent_background_start",
        "agent_background_done",
    ):
        assert event_type in text
    assert "function renderAgentEvent" in text
    assert "task-notification" in text
    assert "scratchpad" in text
    assert "@@CODEBLOCK_" in text
    assert "replace(/(\\d+)/g" not in text
