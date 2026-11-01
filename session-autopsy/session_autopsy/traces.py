"""Export compact transcript excerpts for a candidate.

This is what keeps the LLM compile step cheap: instead of re-reading multi-MB
transcripts, /compile-candidate reads a few hundred normalized lines.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from session_autopsy.signatures import command_signature, normalize_signature
from session_autopsy.trace_io import (
    MAX_RESULT_LINES,
    MAX_TEXT_CHARS,
    clip,
    content_blocks,
    is_external_prompt,
    iter_lines,
    result_texts,
)
from session_autopsy.trace_skill import extract_skill

if TYPE_CHECKING:
    from session_autopsy.candidates import Candidate
    from session_autopsy.records import SessionRecord

MAX_ERR_HITS_PER_SESSION = 5


def render_traces(
    candidate: Candidate,
    records_by_id: dict[str, SessionRecord],
    max_invocations: int,
) -> str:
    """Render trace excerpts for a candidate across its evidence sessions.

    Args:
        candidate: The candidate to trace.
        records_by_id: Stored records keyed by session id (for paths).
        max_invocations: Stop after this many captured spans/hits.

    Returns:
        The full trace document.
    """
    lines = [f"# traces {candidate.id} — {candidate.title}", ""]
    captured = 0
    for session_id in candidate.session_ids:
        record = records_by_id.get(session_id)
        if record is None:
            continue
        if captured >= max_invocations:
            break
        session_lines = trace_session(candidate, record)
        if session_lines:
            captured += 1
            lines.append(
                f"## session {session_id} ({record.meta.started_at or 'undated'})"
            )
            lines.extend(session_lines)
            lines.append("")
    if captured == 0:
        lines.append("no matching spans found (transcripts may have been deleted)")
    return "\n".join(lines) + "\n"


def trace_session(candidate: Candidate, record: SessionRecord) -> list[str]:
    """Extract this candidate's excerpt lines from one session.

    Args:
        candidate: The candidate to trace.
        record: The session's stored record (holds the transcript path).

    Returns:
        Excerpt lines, or an empty list when nothing matches.
    """
    path = existing_path(record)
    if path is None:
        return []
    extractors = {
        "skill": extract_skill,
        "err": extract_error,
        "ngram": extract_ngram,
        "prompt": extract_prompt,
    }
    return extractors[candidate.kind](path, candidate.sig)


def existing_path(record: SessionRecord) -> Path | None:
    """Resolve the record's transcript path if it still exists.

    Args:
        record: The stored session record.

    Returns:
        The path, or None when the transcript is gone.
    """
    path = Path(record.transcript_path)
    if path.is_file():
        return path
    return None


def extract_error(path: Path, sig: str) -> list[str]:
    """Excerpt occurrences of one error signature with the causing command.

    Args:
        path: Transcript path.
        sig: The normalized error signature to match.

    Returns:
        CMD/ERR line pairs, capped per session.
    """
    lines: list[str] = []
    last_bash = ""
    hits = 0
    for obj in iter_lines(path):
        for block in content_blocks(obj) if obj.get("type") == "assistant" else []:
            if block.get("type") == "tool_use" and block.get("name") == "Bash":
                tool_input = block.get("input")
                command = (
                    tool_input.get("command") if isinstance(tool_input, dict) else None
                )
                if isinstance(command, str):
                    last_bash = command
        if obj.get("type") != "user" or hits >= MAX_ERR_HITS_PER_SESSION:
            continue
        for text in result_texts(obj):
            for result_line in text.splitlines():
                if (
                    normalize_signature(result_line) == sig
                    and hits < MAX_ERR_HITS_PER_SESSION
                ):
                    hits += 1
                    lines.append(f"CMD | {clip(last_bash)}")
                    lines.append(f"ERR | {clip(result_line)}")
    return lines


def extract_ngram(path: Path, sig: str) -> list[str]:
    """Excerpt the first occurrence of a command sequence.

    Args:
        path: Transcript path.
        sig: The " → "-joined command-signature gram.

    Returns:
        The raw commands of the first matching window, or empty.
    """
    gram = sig.split(" → ")
    window: list[tuple[str, str]] = []
    for obj in iter_lines(path):
        for block in content_blocks(obj) if obj.get("type") == "assistant" else []:
            if block.get("type") != "tool_use" or block.get("name") != "Bash":
                continue
            tool_input = block.get("input")
            command = (
                tool_input.get("command") if isinstance(tool_input, dict) else None
            )
            if not isinstance(command, str):
                continue
            window.append((command_signature(command), command))
            window = window[-len(gram) :]
            if [entry[0] for entry in window] == gram:
                return [f"CMD | {clip(raw)}" for _, raw in window]
    return []


def extract_prompt(path: Path, sig: str) -> list[str]:
    """Excerpt occurrences of a repeated typed prompt and what followed.

    Args:
        path: Transcript path.
        sig: The normalized prompt signature.

    Returns:
        PROMPT lines followed by the next few tool names.
    """
    lines: list[str] = []
    following = 0
    for obj in iter_lines(path):
        if is_external_prompt(obj):
            message = obj.get("message")
            content = str(message.get("content")) if isinstance(message, dict) else ""
            if normalize_signature(content) == sig:
                lines.append(f"PROMPT | {clip(content, MAX_TEXT_CHARS)}")
                following = MAX_RESULT_LINES
            else:
                following = 0
            continue
        if following <= 0:
            continue
        for block in content_blocks(obj) if obj.get("type") == "assistant" else []:
            if block.get("type") == "tool_use" and following > 0:
                following -= 1
                lines.append(f"THEN | {block.get('name')}")
    return lines
