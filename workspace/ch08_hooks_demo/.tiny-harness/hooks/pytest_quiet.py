import json
import sys

payload = json.loads(sys.stdin.read() or "{}")
tool_input = payload.get("tool_input") or {}
command = str(tool_input.get("command") or "")
if command.strip() == "python -m pytest":
    print(json.dumps({
        "hookSpecificOutput": {
            "updatedInput": {"command": "python -m pytest -q"}
        },
        "additionalContext": "pytest command normalized to quiet mode."
    }))
else:
    print("{}")
