from harness.config import Config
from harness.tools import ToolContext, execute_tool


def test_memory_tools_write_list_read_and_forget(tmp_path):
    workdir = tmp_path / "ws"
    cfg = Config.from_env(workdir=workdir, runs_dir=tmp_path / "runs")
    ctx = ToolContext(workdir=workdir)
    ctx.runtime.config = cfg

    written = execute_tool("write_memory", {
        "type": "feedback",
        "name": "Use Decimal",
        "description": "Use Decimal for money.",
        "content": "Rule: use Decimal for money.\nWhy: avoid float drift.",
    }, ctx)
    assert written.ok
    assert "[◈ Memory Saved]" in written.text

    listed = execute_tool("list_memories", {"type": "feedback", "query": "decimal"}, ctx)
    assert listed.ok
    assert "Use Decimal" in listed.text

    mem_id = listed.text.split()[1]
    read = execute_tool("read_memory", {"id": mem_id}, ctx)
    assert read.ok
    assert "avoid float drift" in read.text

    removed = execute_tool("forget_memory", {"id": mem_id}, ctx)
    assert removed.ok
    assert "[◈ Memory Removed]" in removed.text


def test_memory_tool_rejects_invalid_type(tmp_path):
    workdir = tmp_path / "ws"
    cfg = Config.from_env(workdir=workdir, runs_dir=tmp_path / "runs")
    ctx = ToolContext(workdir=workdir)
    ctx.runtime.config = cfg

    result = execute_tool("write_memory", {
        "type": "random",
        "name": "Bad",
        "description": "Bad",
        "content": "Bad",
    }, ctx)

    assert not result.ok
    assert "invalid memory type" in result.text
