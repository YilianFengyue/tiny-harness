"""Workspace-scoped persistent chat session index.

The message history itself stays in run trajectories. This module stores the
small bit of routing metadata needed to find the latest resumable run for a
workspace after the TUI process exits.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class StoredSession:
    session_id: str
    workdir: str
    title: str
    last_run_id: str | None = None
    created_at: str = ""
    updated_at: str = ""
    turns: int = 0
    cost_usd: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "StoredSession":
        return cls(
            session_id=str(data.get("session_id") or ""),
            workdir=str(data.get("workdir") or ""),
            title=str(data.get("title") or "Session"),
            last_run_id=data.get("last_run_id"),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
            turns=int(data.get("turns") or 0),
            cost_usd=float(data.get("cost_usd") or 0.0),
        )

    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "workdir": self.workdir,
            "title": self.title,
            "last_run_id": self.last_run_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turns": self.turns,
            "cost_usd": round(self.cost_usd, 6),
        }


def session_index_path(workdir: Path) -> Path:
    return Path(workdir).resolve() / ".tiny-harness" / "sessions" / "index.json"


def list_workspace_sessions(workdir: Path) -> list[StoredSession]:
    data = _read_index(workdir)
    sessions = [
        StoredSession.from_dict(item)
        for item in data.get("sessions", [])
        if isinstance(item, dict) and item.get("session_id")
    ]
    expected_workdir = str(Path(workdir).resolve())
    sessions = [s for s in sessions if s.workdir in {"", expected_workdir}]
    return sorted(sessions, key=lambda s: s.updated_at or s.created_at, reverse=True)


def latest_workspace_session(workdir: Path) -> StoredSession | None:
    for session in list_workspace_sessions(workdir):
        if session.last_run_id:
            return session
    return None


def upsert_workspace_session(
    workdir: Path,
    *,
    session_id: str,
    title: str,
    last_run_id: str | None,
    turns: int,
    cost_usd: float,
) -> StoredSession:
    data = _read_index(workdir)
    now = _now()
    resolved_workdir = str(Path(workdir).resolve())
    sessions = [
        StoredSession.from_dict(item)
        for item in data.get("sessions", [])
        if isinstance(item, dict) and item.get("session_id")
    ]
    by_id = {item.session_id: item for item in sessions}
    old = by_id.get(session_id)
    stored = StoredSession(
        session_id=session_id,
        workdir=resolved_workdir,
        title=(title.strip() or (old.title if old else "Session"))[:120],
        last_run_id=last_run_id or (old.last_run_id if old else None),
        created_at=old.created_at if old and old.created_at else now,
        updated_at=now,
        turns=max(turns, old.turns if old else 0),
        cost_usd=max(float(cost_usd), old.cost_usd if old else 0.0),
    )
    by_id[session_id] = stored
    ordered = sorted(by_id.values(), key=lambda s: s.updated_at or s.created_at,
                     reverse=True)[:50]
    _write_index(workdir, ordered)
    return stored


def delete_workspace_session(workdir: Path, session_id: str) -> bool:
    sessions = list_workspace_sessions(workdir)
    kept = [s for s in sessions if s.session_id != session_id]
    if len(kept) == len(sessions):
        return False
    _write_index(workdir, kept)
    return True


def _read_index(workdir: Path) -> dict:
    path = session_index_path(workdir)
    if not path.exists():
        return {"sessions": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {"sessions": []}
    return data if isinstance(data, dict) else {"sessions": []}


def _write_index(workdir: Path, sessions: Iterable[StoredSession]) -> None:
    path = session_index_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"sessions": [session.as_dict() for session in sessions]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
