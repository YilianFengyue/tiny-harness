# CH09 Multi-Agent Demo: LedgerGuard Audit

This workspace is intentionally broken. It is designed to validate TinyAgent's
multi-agent mode from the TUI:

- foreground sub-agent delegation
- explicit fork context inheritance
- background verification agents
- project custom agent loading from `.tiny-harness/agents/*.md`
- normal coding loop: search, read, edit, test, write acceptance notes

## Goal

Fix the LedgerGuard audit report so all tests pass.

Run the failing tests from this directory:

```powershell
python -m pytest -q
```

Expected initial state: tests fail.

## TUI Acceptance Script

Start TinyAgent from repository root:

```powershell
python main.py chat --workdir ./workspace/ch09_multi_agent_demo --max-turns 18 --max-cost 1.0
```

Suggested prompt:

```text
This is a CH09 multi-agent acceptance task.

Fix this LedgerGuard audit workspace until `python -m pytest -q` passes.

Use multi-agent mode deliberately:
1. Launch an explore sub-agent to map the failing tests and relevant source files.
2. Launch a fork=true sub-agent to independently analyze the money/risk logic using the parent context.
3. Use the project custom subagent_type `audit-reviewer` to review the intended audit semantics.
4. Implement the fix yourself in the main agent after reading the files.
5. Launch a verify sub-agent with run_in_background=true before your final answer.
6. Run the tests yourself, write ACCEPTANCE.md with what changed, and read it back.

Do not skip actual verification. If a background agent completes, incorporate its result.
```

Useful TUI checks:

- `Ctrl+O`: open Build details and inspect the `agent` tool lifecycle.
- `/agents`: list background sub-agents and their latest result.
- `/trace`: get the latest trajectory path.

## Acceptance Criteria

- `python -m pytest -q` passes.
- `ACCEPTANCE.md` exists and names the files changed.
- Trajectory contains at least one `agent_start` or `agent_background_start`.
- TUI Build details show sub-agent type, status, run id, and trajectory path.

