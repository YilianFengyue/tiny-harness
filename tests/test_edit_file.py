from harness.tools.registry import execute_tool


def apply(result, ctx):
    if result.context_modifier:
        result.context_modifier(ctx.runtime)
    return result


def test_edit_requires_read_file_first(ctx):
    (ctx.workdir / "app.py").write_text("x = 1\n", encoding="utf-8")
    r = execute_tool("edit_file", {
        "path": "app.py", "old_string": "x = 1", "new_string": "x = 2",
        "replace_all": None,
    }, ctx)
    assert not r.ok
    assert "must be read" in r.text


def test_edit_file_precise_replacement_updates_runtime(ctx):
    (ctx.workdir / "app.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    apply(execute_tool("read_file", {"path": "app.py", "offset": None, "max_lines": None}, ctx), ctx)

    r = execute_tool("edit_file", {
        "path": "app.py",
        "old_string": "    return 1",
        "new_string": "    return 2",
        "replace_all": None,
    }, ctx)
    apply(r, ctx)

    assert r.ok
    assert "replaced 1 occurrence" in r.text
    assert "return 2" in (ctx.workdir / "app.py").read_text(encoding="utf-8")
    assert ctx.runtime.file_history[-1]["action"] == "edit"


def test_edit_rejects_ambiguous_match_without_replace_all(ctx):
    (ctx.workdir / "app.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    apply(execute_tool("read_file", {"path": "app.py", "offset": None, "max_lines": None}, ctx), ctx)

    r = execute_tool("edit_file", {
        "path": "app.py", "old_string": "x = 1", "new_string": "x = 2",
        "replace_all": False,
    }, ctx)
    assert not r.ok
    assert "Found 2 matches" in r.text

    r2 = execute_tool("edit_file", {
        "path": "app.py", "old_string": "x = 1", "new_string": "x = 2",
        "replace_all": True,
    }, ctx)
    assert r2.ok
    assert (ctx.workdir / "app.py").read_text(encoding="utf-8").count("x = 2") == 2


def test_write_existing_file_requires_fresh_read(ctx):
    path = ctx.workdir / "note.txt"
    path.write_text("old\n", encoding="utf-8")

    r = execute_tool("write_file", {"path": "note.txt", "content": "new\n"}, ctx)
    assert not r.ok and "must be read" in r.text

    apply(execute_tool("read_file", {"path": "note.txt", "offset": None, "max_lines": None}, ctx), ctx)
    path.write_text("external change\n", encoding="utf-8")
    r2 = execute_tool("write_file", {"path": "note.txt", "content": "new\n"}, ctx)
    assert not r2.ok and "changed since it was read" in r2.text

    apply(execute_tool("read_file", {"path": "note.txt", "offset": None, "max_lines": None}, ctx), ctx)
    r3 = execute_tool("write_file", {"path": "note.txt", "content": "new\n"}, ctx)
    assert r3.ok and "updated" in r3.text


def test_show_diff_uses_runtime_history_outside_git(ctx):
    (ctx.workdir / "app.py").write_text("x = 1\n", encoding="utf-8")
    apply(execute_tool("read_file", {"path": "app.py", "offset": None, "max_lines": None}, ctx), ctx)
    r = execute_tool("edit_file", {
        "path": "app.py", "old_string": "x = 1", "new_string": "x = 2",
        "replace_all": None,
    }, ctx)
    apply(r, ctx)

    d = execute_tool("show_diff", {"path": "app.py"}, ctx)
    assert d.ok
    assert "-x = 1" in d.text and "+x = 2" in d.text
