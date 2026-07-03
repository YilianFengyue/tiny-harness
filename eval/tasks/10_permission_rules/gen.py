import json
from pathlib import Path


def generate(workdir: Path) -> None:
    settings = workdir / ".tiny-harness" / "settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps({
        "permissions": {
            "mode": "acceptEdits",
            "allow": ["Bash(python *)"],
            "ask": ["Bash(python -c *)"],
            "deny": ["Bash(rm *)", "Bash(del *)"]
        }
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (workdir / "app.py").write_text(
        "def normalize_user(name: str) -> str:\n"
        "    return name.strip().lower()\n",
        encoding="utf-8")
    (workdir / "check.py").write_text(
        "from app import normalize_user\n\n"
        "assert normalize_user('  Admin   User  ') == 'admin user'\n"
        "assert normalize_user('\\tAlice\\nBob  ') == 'alice bob'\n"
        "print('permission rules check passed')\n",
        encoding="utf-8")
