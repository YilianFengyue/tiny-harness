from conftest import MockProvider, turn

from harness.compact import compact_conversation, format_compact_summary
from harness.context import strip_internal_marks


def test_format_compact_summary_strips_analysis():
    raw = "<analysis>private notes</analysis>\n<summary>\nKeep this.\n</summary>"

    assert format_compact_summary(raw) == "Keep this."


def test_compact_conversation_inserts_boundary_and_summary(make_cfg):
    cfg = make_cfg()
    provider = MockProvider([
        turn(content="<analysis>draft</analysis><summary>Remember file app.py and next step.</summary>")
    ])
    messages = [{"role": "system", "content": "system"}]
    for i in range(10):
        messages.append({"role": "user", "content": f"user {i}"})
        messages.append({"role": "assistant", "content": f"assistant {i}"})

    result = compact_conversation(
        messages, provider, cfg, trigger="manual",
        custom_instructions="preserve app.py", keep_recent_messages=4)

    assert result.messages_summarized > 0
    assert messages[0]["role"] == "system"
    assert messages[1]["_kind"] == "compact_boundary"
    assert messages[2]["_kind"] == "compact_summary"
    assert "draft" not in messages[2]["content"]
    assert "Remember file app.py" in messages[2]["content"]

    wire = strip_internal_marks(messages)
    assert all(m.get("_kind") is None for m in wire)
    assert all("compact boundary" not in (m.get("content") or "") for m in wire)
