import json
import sys

payload = json.loads(sys.stdin.read() or "{}")
prompt = payload.get("prompt") or ""
if "金额" in prompt or "invoice" in prompt.lower() or "bug" in prompt.lower():
    print(json.dumps({
        "additionalContext": (
            "CH08 demo rule: verify the invoice bug with pytest, prefer Decimal "
            "for money, and write ACCEPTANCE.md before the final answer."
        )
    }))
else:
    print("{}")
