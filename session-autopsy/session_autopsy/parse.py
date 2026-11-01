"""Streaming parser turning one transcript (plus subagents) into a record.

Transcripts can reach 12 MB, so every file is read line-by-line; no line's
parse result outlives the accumulator fields it updates.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from session_autopsy.accumulator import Accumulator
from session_autopsy.parse_constants import (
    MAX_BASH_FIRST_LINES,
    MAX_ERROR_SIGS,
    MAX_PROMPT_SIGS,
)
from session_autopsy.parse_helpers import top
from session_autopsy.records import Observations, SessionRecord

if TYPE_CHECKING:
    from pathlib import Path


def subagent_files(transcript_path: Path) -> list[Path]:
    """Locate the nested subagent transcripts of a session.

    Args:
        transcript_path: Path to the main ``<uuid>.jsonl`` transcript.

    Returns:
        Sorted ``agent-*.jsonl`` paths under ``<uuid>/subagents/``.
    """
    nested = transcript_path.parent / transcript_path.stem / "subagents"
    if not nested.is_dir():
        return []
    return sorted(nested.glob("agent-*.jsonl"))


def parse_session(transcript_path: Path) -> SessionRecord:
    """Parse one session transcript (and its subagents) into a record.

    Subagent token usage and tool calls fold into the parent session's
    totals; subagent prompts never count as external (they are sidechains).

    Args:
        transcript_path: Path to the main ``<uuid>.jsonl`` transcript.

    Returns:
        The populated :class:`SessionRecord`.
    """
    acc = Accumulator()
    subagents = subagent_files(transcript_path)
    for path in [transcript_path, *subagents]:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                acc.feed_line(raw_line)
        acc.finish()
    stat = transcript_path.stat()
    # The accumulator owns the metadata and counts records outright, so they are
    # handed over rather than copied field by field. Only the subagent count is
    # not something a transcript line can carry: it comes from the file listing.
    acc.counts.subagent_count = len(subagents)
    return SessionRecord(
        session_id=transcript_path.stem,
        project_slug=transcript_path.parent.name,
        transcript_path=str(transcript_path),
        file_size=stat.st_size,
        file_mtime=stat.st_mtime,
        analyzed_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        meta=acc.meta,
        counts=acc.counts,
        tokens=acc.tokens,
        turns=acc.turns,
        obs=Observations(
            tool_histogram=dict(acc.tool_histogram),
            bash_first_lines=top(acc.bash_first_lines, MAX_BASH_FIRST_LINES),
            bash_sequences=acc.bash_sequences,
            skill_invocations=acc.skill_invocations,
            error_signatures=top(acc.error_signatures, MAX_ERROR_SIGS),
            typed_prompt_signatures=top(acc.typed_prompt_signatures, MAX_PROMPT_SIGS),
        ),
    )
