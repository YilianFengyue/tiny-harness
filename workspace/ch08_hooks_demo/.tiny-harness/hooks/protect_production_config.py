import json
import sys

payload = json.loads(sys.stdin.read() or "{}")
tool_input = payload.get("tool_input") or {}
path = str(tool_input.get("path") or "").replace("\\", "/")
if path.endswith("src/production_config.py"):
    print(json.dumps({
        "decision": "block",
        "reason": "CH08 demo hook: production_config.py is protected."
    }))
else:
    print("{}")
