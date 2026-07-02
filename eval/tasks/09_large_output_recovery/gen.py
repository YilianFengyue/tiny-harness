from pathlib import Path


FAIL_IDS = [137, 274, 411, 548, 685, 822, 959, 1096, 1233, 1370, 1507, 1644,
            1781, 1918, 2055]


SCRIPT = f'''FAIL_IDS = {FAIL_IDS!r}


for i in range(1, 2201):
    filler = "x" * 72
    if i in FAIL_IDS:
        print(f"case={{i:04d}} status=FAIL code=E{{i % 7}} detail=retry-budget-exhausted payload={{filler}}")
    else:
        print(f"case={{i:04d}} status=PASS code=OK detail=stable payload={{filler}}")
'''


def generate(workdir: Path) -> None:
    (workdir / "generate_log.py").write_text(SCRIPT, encoding="utf-8")
