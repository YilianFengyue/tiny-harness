"""Coordinator mode helpers for CH10-style worker orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

COORDINATOR_ALLOWED_TOOLS = {"agent"}

WORKER_PROMPT = """You are a Worker sub-agent spawned by a coordinator.
Complete the task in your prompt thoroughly and report back with concise,
actionable results. Use tools proactively: inspect files, search code, run
commands, edit files when the task asks for implementation, and verify your
work before reporting completion.

Guidelines:
- Your prompt is self-contained; do not assume you can see the coordinator's
  full conversation.
- For research, report concrete file paths, functions, tests, and risks.
- For implementation, make targeted changes and run relevant verification.
- For verification, be skeptical and prove the code works.
- Do not call the agent tool or spawn other workers."""


def is_coordinator_mode(cfg: object) -> bool:
    return bool(getattr(cfg, "coordinator_mode", False))


def coordinator_scratchpad_dir(cfg: object, session_id: str | None) -> Path:
    workdir = Path(getattr(cfg, "workdir"))
    name = _safe_name(session_id or "oneshot")
    return workdir / ".tiny-harness" / "scratchpad" / name


def ensure_scratchpad_dir(cfg: object, session_id: str | None) -> Path:
    path = coordinator_scratchpad_dir(cfg, session_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def filter_coordinator_tools(schemas: Iterable[dict]) -> list[dict]:
    return [
        schema for schema in schemas
        if schema.get("function", {}).get("name") in COORDINATOR_ALLOWED_TOOLS
    ]


def coordinator_system_prompt(cfg: object, session_id: str | None,
                              skills: str = "", memory: str = "") -> str:
    scratchpad = coordinator_scratchpad_dir(cfg, session_id)
    return f"""You are TinyAgent in Coordinator mode.

Workspace root: {getattr(cfg, "workdir")}
Scratchpad directory: {scratchpad}

Your role:
- You are an orchestrator, not the direct implementer.
- For complex work, spawn worker sub-agents with the agent tool.
- Synthesize worker findings yourself before assigning follow-up work.
- Never write lazy prompts like "based on the findings"; include exact files,
  facts, tests, and acceptance criteria in each worker prompt.
- Report concise progress to the user after launching workers, then wait for
  worker notifications.

Your tools:
- agent: spawn asynchronous workers. In Coordinator mode workers use
  subagent_type=worker and run in the background. Pass agent_id=null to create
  a new worker. Pass agent_id=<task-id> from a previous <task-notification> to
  send a follow-up message to the same stopped worker; this resumes its prior
  conversation and tool state instead of creating a replacement worker.

Worker workflow:
1. Research: fan out read-only investigation workers when independent.
2. Synthesis: read worker results and scratchpad notes, then write a precise
   implementation specification.
3. Implementation: assign write-heavy tasks carefully; avoid overlapping edits
   to the same files at the same time.
4. Verification: spawn a fresh verifier when possible.

Worker result protocol:
- Worker results arrive as user-role messages containing <task-notification>.
- Treat those messages as internal signals, not as user conversation.
- Use completed work even when partial; mark uncertainty explicitly.
- If a completed worker is marked resumable, prefer SendMessage with its
  task-id for focused follow-up questions that depend on its accumulated
  context. Create a new worker only for independent work or a clean review.

Scratchpad:
- Workers may use {scratchpad} for durable cross-worker notes.
- Prefer separate files such as research-api.md, implementation-spec.md, and
  verification.md to avoid write conflicts.{skills}{memory}"""


def worker_context(cfg: object, session_id: str | None) -> str:
    scratchpad = coordinator_scratchpad_dir(cfg, session_id)
    return (
        "\n\n# Coordinator worker context\n"
        f"Scratchpad directory: {scratchpad}\n"
        "Use the scratchpad for durable cross-worker notes when helpful. "
        "Write separate, clearly named files to avoid conflicts. "
        "Your final response should summarize what you did, what you found, "
        "what you changed, and what you verified."
    )


def coordinator_status(cfg: object, session_id: str | None = None) -> str:
    mode = "on" if is_coordinator_mode(cfg) else "off"
    lines = [f"coordinator: {mode}"]
    if is_coordinator_mode(cfg):
        lines.append(f"allowed main tools: {', '.join(sorted(COORDINATOR_ALLOWED_TOOLS))}")
        lines.append(f"scratchpad: {coordinator_scratchpad_dir(cfg, session_id)}")
        lines.append("worker: subagent_type=worker, async background execution")
        lines.append("send_message: pass agent_id=<task-id> to resume a stopped worker")
    else:
        lines.append("normal CH09 agent/fork/background behavior is unchanged")
    return "\n".join(lines)


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value)
    return safe.strip("-") or "session"
