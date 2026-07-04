import json

from harness.config import Config
from harness.loop import build_initial_messages
from harness.memory import (
    MEMORY_INDEX,
    load_memory_records,
    memory_path_info,
    read_memory_index,
    rebuild_memory_index,
    render_memory_prompt,
    truncate_index,
    write_memory,
)


def test_memory_write_rebuilds_typed_index(tmp_path):
    directory = tmp_path / "memory"

    record = write_memory(
        directory,
        "feedback",
        "Run tests before summary",
        "User expects tests before final summaries.",
        "Rule: run focused tests before final summaries.\nWhy: confidence matters.",
    )

    assert record.type == "feedback"
    assert (directory / MEMORY_INDEX).exists()
    index = read_memory_index(directory)
    assert "## feedback" in index
    assert "Run tests before summary" in index
    records = load_memory_records(directory)
    assert records[0].description == "User expects tests before final summaries."


def test_memory_index_truncates_lines_then_bytes():
    text = "\n".join(f"line {i} {'x' * 20}" for i in range(20))

    line_limited = truncate_index(text, max_lines=5, max_bytes=10_000)
    assert line_limited.count("\n") == 5
    assert "line 6" not in line_limited

    byte_limited = truncate_index(text, max_lines=20, max_bytes=40)
    assert len(byte_limited.encode("utf-8")) <= 41


def test_project_auto_memory_directory_is_ignored_but_local_is_trusted(tmp_path):
    workdir = tmp_path / "ws"
    settings_dir = workdir / ".tiny-harness"
    settings_dir.mkdir(parents=True)
    project_target = tmp_path / "project_should_not_win"
    local_target = tmp_path / "local_memory"
    (settings_dir / "settings.json").write_text(json.dumps({
        "autoMemoryDirectory": str(project_target),
    }), encoding="utf-8")
    (settings_dir / "settings.local.json").write_text(json.dumps({
        "autoMemoryDirectory": str(local_target),
    }), encoding="utf-8")

    cfg = Config.from_env(workdir=workdir, runs_dir=tmp_path / "runs")
    info = memory_path_info(cfg)

    assert info.directory == local_target.resolve()
    assert info.source == "localSettings"
    assert info.ignored_project_directory == str(project_target)


def test_invalid_trusted_memory_directory_falls_back_to_default(tmp_path):
    workdir = tmp_path / "ws"
    settings_dir = workdir / ".tiny-harness"
    settings_dir.mkdir(parents=True)
    (settings_dir / "settings.local.json").write_text(json.dumps({
        "autoMemoryDirectory": "relative-memory",
    }), encoding="utf-8")

    cfg = Config.from_env(workdir=workdir, runs_dir=tmp_path / "runs")
    info = memory_path_info(cfg)

    assert info.source == "default"
    assert "absolute" in (info.warning or "")
    assert "relative-memory" not in str(info.directory)


def test_memory_prompt_injects_index_and_verification_guidance(tmp_path):
    workdir = tmp_path / "ws"
    directory = tmp_path / "mem"
    settings_dir = workdir / ".tiny-harness"
    settings_dir.mkdir(parents=True)
    (settings_dir / "settings.local.json").write_text(json.dumps({
        "autoMemoryDirectory": str(directory),
    }), encoding="utf-8")
    cfg = Config.from_env(workdir=workdir, runs_dir=tmp_path / "runs")
    write_memory(
        directory,
        "project",
        "Decimal money",
        "Amounts should be handled with Decimal.",
        "Rule: use Decimal for financial totals.\nWhy: avoid binary float drift.",
    )

    prompt = render_memory_prompt(cfg)
    messages = build_initial_messages("fix totals", cfg)

    assert "# Memory" in prompt
    assert "Treat memory as clues, not truth." in prompt
    assert "Decimal money" in messages[0]["content"]
    assert "verify the current workspace state" in messages[0]["content"]


def test_rebuild_memory_index_keeps_empty_index_small(tmp_path):
    directory = tmp_path / "memory"

    text = rebuild_memory_index(directory)

    assert "(no memories)" in text
    assert (directory / MEMORY_INDEX).read_text(encoding="utf-8") == text
