"""In-process background sub-agent registry."""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

from .agents import AgentDefinition
from .cancel import CancellationToken

if TYPE_CHECKING:
    from .subagent import SubagentRunResult
    from .tools.registry import ToolContext


@dataclass
class BackgroundAgentRecord:
    agent_id: str
    agent_type: str
    description: str
    prompt: str
    status: str = "running"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    result: "SubagentRunResult | None" = None
    error: str | None = None
    notified: bool = False
    fork: bool = False

    def as_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "description": self.description,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "notified": self.notified,
            "fork": self.fork,
            "run_id": self.result.run_id if self.result else self.agent_id,
            "trajectory_path": self.result.trajectory_path if self.result else None,
            "final_message": self.result.final_message if self.result else None,
        }


class BackgroundAgentManager:
    def __init__(self) -> None:
        self._records: dict[str, BackgroundAgentRecord] = {}
        self._lock = threading.RLock()

    def start(self, agent: AgentDefinition, prompt: str, parent_ctx: "ToolContext",
              *, description: str, fork: bool = False,
              fork_messages: list[dict] | None = None,
              emit: Callable[[dict], None] | None = None) -> BackgroundAgentRecord:
        agent_id = f"bg-{int(time.time() * 1000)}-{len(self._records) + 1}"
        record = BackgroundAgentRecord(
            agent_id=agent_id,
            agent_type=agent.agent_type,
            description=description,
            prompt=prompt,
            fork=fork,
        )
        with self._lock:
            self._records[agent_id] = record

        token = CancellationToken()

        def run() -> None:
            try:
                from .subagent import run_subagent

                result = run_subagent(
                    agent,
                    prompt,
                    parent_ctx,
                    description=description,
                    emit=emit,
                    cancel_token=token,
                    fork_messages=fork_messages,
                )
                with self._lock:
                    record.result = result
                    record.status = result.status
                    record.finished_at = time.time()
            except Exception as exc:
                with self._lock:
                    record.status = "error"
                    record.error = f"{type(exc).__name__}: {exc}"
                    record.finished_at = time.time()
                    record.prompt = record.prompt + "\n\n" + traceback.format_exc(limit=5)

        thread = threading.Thread(target=run, name=f"tiny-agent-{agent_id}", daemon=True)
        thread.start()
        return record

    def list(self) -> list[BackgroundAgentRecord]:
        with self._lock:
            return list(self._records.values())

    def drain_completed(self) -> list[BackgroundAgentRecord]:
        ready: list[BackgroundAgentRecord] = []
        with self._lock:
            for record in self._records.values():
                if record.notified:
                    continue
                if record.status == "running":
                    continue
                record.notified = True
                ready.append(record)
        return ready


def format_background_agents(manager: BackgroundAgentManager | None) -> str:
    if manager is None:
        return "No background agents."
    records = manager.list()
    if not records:
        return "No background agents."
    lines = ["Background agents:"]
    for record in records:
        elapsed = ((record.finished_at or time.time()) - record.started_at)
        kind = " fork" if record.fork else ""
        run = record.result.run_id if record.result else record.agent_id
        lines.append(
            f"- {record.agent_type}{kind} {record.status} "
            f"{elapsed:.1f}s run={run} desc={record.description}")
        if record.error:
            lines.append(f"  error: {record.error}")
        elif record.result and record.result.final_message:
            first = record.result.final_message.strip().splitlines()[0][:180]
            lines.append(f"  result: {first}")
    return "\n".join(lines)


def notification_message(record: BackgroundAgentRecord) -> str:
    if record.result:
        return (
            "[Background subagent completed]\n"
            f"agent_type: {record.agent_type}\n"
            f"status: {record.status}\n"
            f"run_id: {record.result.run_id}\n"
            f"trajectory: {record.result.trajectory_path}\n\n"
            f"{record.result.final_message or '(no final message)'}"
        )
    return (
        "[Background subagent failed]\n"
        f"agent_type: {record.agent_type}\n"
        f"status: {record.status}\n"
        f"error: {record.error or 'unknown error'}"
    )
