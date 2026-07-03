from pathlib import Path


def generate(workdir: Path) -> None:
    (workdir / "design.md").write_text(
        "# Design Notes\n\n"
        "- TODO: split parser from renderer.\n"
        "- Keep the CLI dependency-free.\n",
        encoding="utf-8")
