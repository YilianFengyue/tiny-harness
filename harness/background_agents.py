"""In-process background sub-agent registry."""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from xml.sax.saxutils import escape
from typing import Callable, TYPE_CHECKING

from .agents import AgentDefinition
from .cancel import CancellationToken

if TYPE_CHECKING:
    from .subagent import SubagentRunResult, SubagentRuntime
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
    runtime: "SubagentRuntime | None" = None
    resume_count: int = 0
    resumable: bool = True

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
            "resume_count": self.resume_count,
            "resumable": self.resumable and self.runtime is not None,
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
        from .subagent import create_subagent_runtime

        runtime = create_subagent_runtime(
            agent,
            prompt,
            parent_ctx,
            description=description,
            fork_messages=fork_messages,
            agent_id=agent_id,
        )
        record = BackgroundAgentRecord(
            agent_id=agent_id,
            agent_type=agent.agent_type,
            description=description,
            prompt=prompt,
            fork=fork,
            runtime=runtime,
        )
        with self._lock:
            self._records[agent_id] = record
        self._launch(record, prompt, parent_ctx, description=description,
                     fork=fork, emit=emit)
        return record

    def send_message(self, agent_id: str, prompt: str, parent_ctx: "ToolContext",
                     *, description: str | None = None,
                     emit: Callable[[dict], None] | None = None
                     ) -> tuple[BackgroundAgentRecord | None, str | None]:
        with self._lock:
            record = self._records.get(agent_id)
            if record is None:
                return None, f"unknown worker agent_id {agent_id!r}"
            if record.status == "running":
                return record, f"worker {agent_id} is still running"
            if record.runtime is None or not record.resumable:
                return record, f"worker {agent_id} is not resumable"
            from .subagent import append_resume_message

            append_resume_message(record.runtime, prompt)
            record.resume_count = record.runtime.resume_count
            record.description = description or record.description
            record.prompt = record.prompt + "\n\n[Coordinator SendMessage]\n" + prompt
            record.status = "running"
            record.started_at = time.time()
            record.finished_at = None
            record.error = None
            record.notified = False
        self._launch(record, prompt, parent_ctx,
                     description=description or record.description,
                     fork=record.fork, emit=emit)
        return record, None

    def _launch(self, record: BackgroundAgentRecord, prompt: str,
                parent_ctx: "ToolContext", *, description: str,
                fork: bool = False,
                emit: Callable[[dict], None] | None = None) -> None:
        token = CancellationToken()

        def run() -> None:
            try:
                from .subagent import run_subagent_runtime

                if record.runtime is None:
                    raise RuntimeError("background agent has no runtime")
                result = run_subagent_runtime(
                    record.runtime,
                    prompt,
                    parent_ctx,
                    description=description,
                    emit=emit,
                    cancel_token=token,
                    fork=fork,
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

        thread = threading.Thread(
            target=run,
            name=f"tiny-agent-{record.agent_id}",
            daemon=True,
        )
        thread.start()

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
        resume = f" resume={record.resume_count}" if record.resume_count else ""
        resumable = " resumable" if record.resumable and record.runtime else ""
        run = record.result.run_id if record.result else record.agent_id
        lines.append(
            f"- {record.agent_type}{kind}{resume}{resumable} {record.status} "
            f"{elapsed:.1f}s run={run} desc={record.description}")
        if record.error:
            lines.append(f"  error: {record.error}")
        elif record.result and record.result.final_message:
            first = record.result.final_message.strip().splitlines()[0][:180]
            lines.append(f"  result: {first}")
    return "\n".join(lines)


def notification_message(record: BackgroundAgentRecord,
                         coordinator_mode: bool = False) -> str:
    if coordinator_mode:
        return _task_notification(record)
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


def _task_notification(record: BackgroundAgentRecord) -> str:
    task_id = record.agent_id
    status = record.status if record.status in {"completed", "failed", "killed"} else "failed"
    if record.result and record.result.status != "completed":
        status = "failed"
    if record.error:
        status = "failed"
    summary = (
        f"Worker {record.agent_type} {status}: {record.description}"
        if status != "completed"
        else f"Worker {record.agent_type} completed: {record.description}"
    )
    result = record.result
    text = [
        "<task-notification>",
        f"<task-id>{escape(task_id)}</task-id>",
        f"<status>{escape(status)}</status>",
        f"<summary>{escape(summary)}</summary>",
        f"<resumable>{str(record.resumable and record.runtime is not None).lower()}</resumable>",
        f"<resume-count>{record.resume_count}</resume-count>",
    ]
    if result:
        text.extend([
            f"<run-id>{escape(result.run_id)}</run-id>",
            f"<trajectory>{escape(result.trajectory_path)}</trajectory>",
            f"<result>{escape(result.final_message or '(no final message)')}</result>",
            "<usage>",
            f"<total_tokens>{result.usage.prompt_tokens + result.usage.completion_tokens}</total_tokens>",
            f"<tool_uses>{result.tool_count}</tool_uses>",
            "</usage>",
        ])
    elif record.error:
        text.append(f"<result>{escape(record.error)}</result>")
    text.append("</task-notification>")
    return "\n".join(text)
