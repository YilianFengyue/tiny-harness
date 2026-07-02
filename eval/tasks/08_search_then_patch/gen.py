from pathlib import Path


PARSER = '''def legacy_parse(payload: str) -> dict:
    return {"engine": "legacy", "value": payload.strip().lower()}


def parse_v2(payload: str) -> dict:
    return {"engine": "v2", "value": payload.strip().casefold()}
'''


def _module(i: int) -> str:
    if i == 17:
        call = "legacy_parse(payload)"
        import_line = "from parser import legacy_parse, parse_v2"
    else:
        call = "parse_v2(payload)"
        import_line = "from parser import parse_v2"
    return f'''{import_line}


MODULE_ID = {i}
SENTINEL = "module-{i:02d}-stable"


def handle(payload: str) -> dict:
    result = {call}
    result["module"] = MODULE_ID
    return result
'''


CHECK = '''import importlib


for i in range(1, 31):
    mod = importlib.import_module(f"modules.module_{i:02d}")
    result = mod.handle("  Hello  ")
    assert result["engine"] == "v2", (i, result)
    assert result["value"] == "hello", (i, result)
    assert result["module"] == i, result

print("search patch ok")
'''


def generate(workdir: Path) -> None:
    (workdir / "parser.py").write_text(PARSER, encoding="utf-8")
    modules = workdir / "modules"
    modules.mkdir()
    (modules / "__init__.py").write_text("", encoding="utf-8")
    for i in range(1, 31):
        (modules / f"module_{i:02d}.py").write_text(_module(i), encoding="utf-8")
    (workdir / "check.py").write_text(CHECK, encoding="utf-8")
