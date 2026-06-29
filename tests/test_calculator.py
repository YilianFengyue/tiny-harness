from harness.tools.registry import execute_tool


def calc(ctx, expr):
    return execute_tool("calculator", {"expression": expr}, ctx)


def test_basic_arithmetic(ctx):
    assert calc(ctx, "(12.5 + 7) / 3").text.endswith("6.5")
    assert "= 8" in calc(ctx, "2**3").text


def test_functions_and_constants(ctx):
    assert "= 2" in calc(ctx, "sqrt(4)").text
    assert "3.14" in calc(ctx, "pi").text
    assert "= 6" in calc(ctx, "sum([1, 2, 3])").text
    assert "= 3" in calc(ctx, "max(1, 3, 2)").text


def test_rejects_code_injection(ctx):
    for evil in ["__import__('os').system('id')",
                 "().__class__.__bases__",
                 "open('/etc/passwd')",
                 "exec('x=1')"]:
        r = calc(ctx, evil)
        assert not r.ok, f"should reject: {evil}"


def test_rejects_unknown_name_and_strings(ctx):
    assert not calc(ctx, "a + 1").ok
    assert not calc(ctx, "'abc' + 'd'").ok


def test_rejects_huge_exponent(ctx):
    r = calc(ctx, "9 ** 999999")
    assert not r.ok and "overflow" in r.text


def test_syntax_error_is_recoverable(ctx):
    r = calc(ctx, "1 +")
    assert not r.ok and "syntax error" in r.text


def test_list_result_guidance(ctx):
    r = calc(ctx, "[1, 2]")
    assert not r.ok and "sum()" in r.text
