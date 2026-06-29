from harness.tools.registry import execute_tool


def test_write_read_roundtrip(ctx):
    w = execute_tool("write_file", {"path": "out/answer.txt", "content": "42.5\n"}, ctx)
    assert w.ok
    r = execute_tool("read_file", {"path": "out/answer.txt", "offset": None, "max_lines": None}, ctx)
    assert r.ok and "42.5" in r.text and "1 lines total" in r.text


def test_path_escape_rejected(ctx):
    for path in ["../escape.txt", "../../etc/passwd", "a/../../escape.txt"]:
        r = execute_tool("write_file", {"path": path, "content": "x"}, ctx)
        assert not r.ok and "escapes the workspace" in r.text, path
    outside = ctx.workdir.parent / "absolute.txt"
    r = execute_tool("read_file", {"path": str(outside), "offset": None, "max_lines": None}, ctx)
    assert not r.ok and "escapes the workspace" in r.text


def test_missing_file_error_is_actionable(ctx):
    (ctx.workdir / "sales_data.csv").write_text("a,b,c\n", encoding="utf-8")
    r = execute_tool("read_file", {"path": "data.csv", "offset": None, "max_lines": None}, ctx)
    assert not r.ok
    # 错误信息必须包含目录里实际存在的文件，让模型能自纠
    assert "sales_data.csv" in r.text


def test_paging(ctx):
    (ctx.workdir / "big.txt").write_text(
        "\n".join(f"line{i}" for i in range(1, 1001)), encoding="utf-8")
    r = execute_tool("read_file", {"path": "big.txt", "offset": 1, "max_lines": 10}, ctx)
    assert r.ok and "line10" in r.text and "line11" not in r.text
    assert "offset=11" in r.text  # 续读指引
    r2 = execute_tool("read_file", {"path": "big.txt", "offset": 991, "max_lines": 100}, ctx)
    assert "line1000" in r2.text and "more lines" not in r2.text


def test_list_files(ctx):
    (ctx.workdir / "a.txt").write_text("x", encoding="utf-8")
    (ctx.workdir / "sub").mkdir()
    r = execute_tool("list_files", {"path": None}, ctx)
    assert r.ok and "a.txt" in r.text and "sub" in r.text
