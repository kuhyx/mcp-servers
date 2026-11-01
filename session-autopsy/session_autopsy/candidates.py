"""Candidate and trend records shared by every detector, plus id helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from session_autopsy.records import SessionRecord

MAX_EVIDENCE_SESSIONS = 10


@dataclass
class Candidate:
    """One ranked automation opportunity."""

    id: str
    kind: str
    title: str
    sig: str
    sessions: int
    occurrences: int
    avg_tokens: int
    per_week: float
    est_weekly_savings: int
    score: float
    action: str
    session_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Serialize for ``candidates --json``.

        Returns:
            A JSON-compatible dict of all fields.
        """
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "sig": self.sig,
            "sessions": self.sessions,
            "occurrences": self.occurrences,
            "avg_tokens": self.avg_tokens,
            "per_week": round(self.per_week, 2),
            "est_weekly_savings": self.est_weekly_savings,
            "score": round(self.score, 3),
            "action": self.action,
            "session_ids": self.session_ids,
        }


@dataclass
class TrendInfo:
    """Weekly cache-read-per-turn averages and their direction."""

    weekly: list[tuple[str, int]] = field(default_factory=list)
    slope_per_week: float = 0.0
    top_sessions: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Everything the report renders: ranked candidates and the trend."""

    candidates: list[Candidate] = field(default_factory=list)
    trend: TrendInfo = field(default_factory=TrendInfo)


def hex_id(text: str) -> str:
    """Derive the stable 8-hex candidate id suffix for a signature.

    Args:
        text: The normalized signature.

    Returns:
        First 8 hex chars of its SHA-256.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def recent_session_ids(records: list[SessionRecord]) -> list[str]:
    """Most recent contributing session ids for the traces subcommand.

    Args:
        records: Contributing session records (may repeat).

    Returns:
        Up to :data:`MAX_EVIDENCE_SESSIONS` unique ids, newest first.
    """
    unique: dict[str, str] = {}
    for record in records:
        unique[record.session_id] = record.meta.started_at or ""
    ranked = sorted(unique.items(), key=lambda item: item[1], reverse=True)
    return [session_id for session_id, _ in ranked[:MAX_EVIDENCE_SESSIONS]]
