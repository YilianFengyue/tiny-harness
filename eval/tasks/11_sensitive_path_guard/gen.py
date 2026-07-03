import json
from pathlib import Path


def generate(workdir: Path) -> None:
    settings = workdir / ".tiny-harness" / "settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps({
        "permissions": {
            "mode": "acceptEdits",
            "allow": ["Bash(python *)"]
        }
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (workdir / ".env").write_text("SECRET_TOKEN=original-token\n", encoding="utf-8")
