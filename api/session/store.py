"""In-memory session store with inactivity-based expiration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import numpy as np


@dataclass
class SessionState:
    """Runtime session state for one user interaction stream."""

    session_id: str
    history_vectors: list[dict[str, np.ndarray]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    played_indices: list[int] = field(default_factory=list)
    current_profile_vector: np.ndarray | None = None
    previous_profile_vector: np.ndarray | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SessionStore:
    """Simple in-memory store for session lifecycle management."""

    def __init__(self, ttl_minutes: int = 30) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._ttl = timedelta(minutes=ttl_minutes)

    def cleanup_expired(self) -> None:
        """Remove sessions that are inactive longer than configured TTL."""
        now = datetime.now(timezone.utc)
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.updated_at > self._ttl
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)

    def create(self) -> SessionState:
        """Create and register a new session."""
        self.cleanup_expired()
        session = SessionState(session_id=str(uuid4()))
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str, touch: bool = True) -> SessionState | None:
        """Fetch an existing session by id."""
        self.cleanup_expired()
        session = self._sessions.get(session_id)
        if session and touch:
            self.touch(session)
        return session

    def touch(self, session: SessionState) -> None:
        """Refresh the last-access timestamp for a session."""
        session.updated_at = datetime.now(timezone.utc)

    def delete(self, session_id: str) -> bool:
        """Delete session if present and return deletion status."""
        self.cleanup_expired()
        return self._sessions.pop(session_id, None) is not None
