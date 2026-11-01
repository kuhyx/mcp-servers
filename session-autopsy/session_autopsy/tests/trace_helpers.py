"""Builders shared by the trace-extraction tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from session_autopsy.candidates import Candidate
from session_autopsy.tests.conftest import FileFacts, record, write_transcript

if TYPE_CHECKING:
    from pathlib import Path

    from session_autopsy.records import SessionRecord

DENIED = "rm: cannot remove '/home/kuhy/.cache/yay/pkg': Permission denied"


def trace_candidate(kind: str, sig: str, session_ids: list[str]) -> Candidate:
    """A candidate pointing at the given evidence sessions."""
    return Candidate(
        id=f"{kind}-x",
        kind=kind,
        title="t",
        sig=sig,
        sessions=len(session_ids),
        occurrences=1,
        avg_tokens=1,
        per_week=0.0,
        est_weekly_savings=0,
        score=1.0,
        action="a",
        session_ids=session_ids,
    )


def trace_session(
    tmp_path: Path, session_id: str, lines: list[dict[str, object] | str]
) -> SessionRecord:
    """Write a transcript and a record whose path points at it."""
    path = write_transcript(tmp_path / "projects", session_id, lines)
    return record(session_id, file=FileFacts(path=str(path)))
