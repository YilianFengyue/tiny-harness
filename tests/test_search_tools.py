from harness.tools.registry import execute_tool


def test_glob_files_finds_matching_paths(ctx):
    (ctx.workdir / "src").mkdir()
    (ctx.workdir / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    (ctx.workdir / "src" / "app.txt").write_text("hi\n", encoding="utf-8")

    r = execute_tool("glob_files", {"pattern": "**/*.py", "path": None, "max_results": None}, ctx)

    assert r.ok
    assert "src/app.py" in r.text
    assert "app.txt" not in r.text


def test_grep_returns_path_line_content(ctx):
    (ctx.workdir / "a.py").write_text("def target():\n    return 1\n", encoding="utf-8")
    (ctx.workdir / "b.py").write_text("def other():\n    return 2\n", encoding="utf-8")

    r = execute_tool("grep", {
        "pattern": "target", "path": None, "include": "*.py",
        "max_results": 10, "context_lines": None,
    }, ctx)

    assert r.ok
    assert "a.py:1:" in r.text
    assert "target" in r.text
    assert "b.py" not in r.text


def test_grep_invalid_regex_is_recoverable(ctx, monkeypatch):
    import harness.tools.search as search
    monkeypatch.setattr(search.shutil, "which", lambda name: None)
    (ctx.workdir / "a.py").write_text("x\n", encoding="utf-8")

    r = execute_tool("grep", {
        "pattern": "[", "path": None, "include": "*.py",
        "max_results": None, "context_lines": None,
    }, ctx)

    assert not r.ok
    assert "invalid regex" in r.text


def test_file_info_reports_lines(ctx):
    (ctx.workdir / "data.txt").write_text("a\nb\n", encoding="utf-8")
    r = execute_tool("file_info", {"path": "data.txt"}, ctx)
    assert r.ok
    assert "size=" in r.text and "lines=2" in r.text
