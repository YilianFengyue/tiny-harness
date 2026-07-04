import json
import pathlib
import sys

payload = json.loads(sys.stdin.read() or "{}")
workdir = pathlib.Path(payload.get("workdir") or ".")
if not (workdir / "ACCEPTANCE.md").exists():
    print(json.dumps({
        "additionalContext": (
            "Before final answer, create ACCEPTANCE.md with the bug fixed, "
            "commands run, and test result."
        )
    }))
else:
    print("{}")
