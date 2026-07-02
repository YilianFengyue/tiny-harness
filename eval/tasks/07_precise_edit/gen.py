from pathlib import Path


APP = '''"""Tiny scoring helpers used by the acceptance task."""


HEADER_SENTINEL = "do-not-touch-header-v1"


def normalize_name(name: str) -> str:
    return name.strip().title()


def compute_score(passed: int, total: int) -> float:
    """Return a percentage score rounded for display."""
    return round((passed / total) * 100, 2)


def format_badge(name: str, passed: int, total: int) -> str:
    score = compute_score(passed, total)
    return f"{normalize_name(name)}: {score}%"


FOOTER_SENTINEL = "do-not-touch-footer-v1"
'''


CHECK = '''from app import compute_score, format_badge


assert compute_score(5, 10) == 50.0
assert compute_score(0, 0) == 0.0
assert compute_score(3, 0) == 0.0
assert format_badge(" ada ", 1, 4) == "Ada: 25.0%"

print("check ok")
'''


def generate(workdir: Path) -> None:
    (workdir / "app.py").write_text(APP, encoding="utf-8")
    (workdir / "check.py").write_text(CHECK, encoding="utf-8")
