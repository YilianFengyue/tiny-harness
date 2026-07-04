import json
import pathlib
import sys
from datetime import datetime, timezone

payload = json.loads(sys.stdin.read() or "{}")
workdir = pathlib.Path(payload.get("workdir") or ".")
tool_input = payload.get("tool_input") or {}
path = tool_input.get("path") or "(unknown)"
audit = workdir / ".tiny-harness" / "hook-audit.log"
audit.parent.mkdir(parents=True, exist_ok=True)
audit.write_text(
    (audit.read_text(encoding="utf-8") if audit.exists() else "")
    + f"{datetime.now(timezone.utc).isoformat()} {payload.get('tool_name')} {path}\n",
    encoding="utf-8",
)
print(json.dumps({
    "additionalContext": f"Audit recorded for {path}."
}))
