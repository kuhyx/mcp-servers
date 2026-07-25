"""Export compact transcript excerpts for a candidate.

This is what keeps the LLM compile step cheap: instead of re-reading multi-MB
transcripts, /compile-candidate reads a few hundred normalized lines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from python_pkg.session_autopsy.signatures import command_signature, normalize_signature

if TYPE_CHECKING:
    from collections.abc import Iterator

    from python_pkg.session_autopsy.detectors import Candidate
    from python_pkg.session_autopsy.records import SessionRecord

MAX_LINES_PER_SPAN = 200
MAX_TEXT_CHARS = 200
MAX_RESULT_LINES = 3
MAX_LINE_CHARS = 160
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
        session_lines = _trace_session(candidate, record)
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


def _trace_session(candidate: Candidate, record: SessionRecord) -> list[str]:
    """Extract this candidate's excerpt lines from one session.

    Args:
        candidate: The candidate to trace.
        record: The session's stored record (holds the transcript path).

    Returns:
        Excerpt lines, or an empty list when nothing matches.
    """
    path = _existing_path(record)
    if path is None:
        return []
    extractors = {
        "skill": _extract_skill,
        "err": _extract_error,
        "ngram": _extract_ngram,
        "prompt": _extract_prompt,
    }
    return extractors[candidate.kind](path, candidate.sig)


def _existing_path(record: SessionRecord) -> Path | None:
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


def _iter_lines(path: Path) -> Iterator[dict[str, object]]:
    """Yield parsed dict lines of a transcript, skipping malformed ones.

    Args:
        path: Transcript path.

    Yields:
        Each line's parsed dict.
    """
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _clip(text: str, limit: int = MAX_LINE_CHARS) -> str:
    """Single-line clip for trace output.

    Args:
        text: Raw text.
        limit: Character cap.

    Returns:
        The first line, capped.
    """
    first = text.strip().splitlines()[0] if text.strip() else ""
    return first[:limit]


def _content_blocks(obj: dict[str, object]) -> list[dict[str, object]]:
    """Return a line's message content blocks.

    Args:
        obj: A parsed transcript line.

    Returns:
        The content block dicts, or an empty list.
    """
    message = obj.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def _is_external_prompt(obj: dict[str, object]) -> bool:
    """Whether a line is a real typed user prompt.

    Args:
        obj: A parsed transcript line.

    Returns:
        True for external, non-sidechain, string-content user lines.
    """
    if (
        obj.get("type") != "user"
        or obj.get("userType") != "external"
        or obj.get("isSidechain") is True
    ):
        return False
    message = obj.get("message")
    return isinstance(message, dict) and isinstance(message.get("content"), str)


def _result_texts(obj: dict[str, object]) -> list[str]:
    """Collect the textual parts of a line's toolUseResult.

    Args:
        obj: A parsed transcript line.

    Returns:
        Zero or more raw result texts.
    """
    result = obj.get("toolUseResult")
    texts = []
    if isinstance(result, str):
        texts.append(result)
    elif isinstance(result, dict):
        for key in ("stderr", "stdout"):
            value = result.get(key)
            if isinstance(value, str) and value:
                texts.append(value)
    return texts


class _SpanBuffer:
    """Excerpt lines for the skill spans found in one transcript.

    Owning ``in_span`` and the per-span line budget here is what keeps
    :func:`_extract_skill` down to a readable walk: every "am I inside a span and
    is there room left" question is answered by :meth:`add`.
    """

    def __init__(self) -> None:
        """Start outside any span with nothing captured."""
        self.lines: list[str] = []
        self.in_span = False
        self._span_lines = 0

    def start(self, *, matched: bool, header: str) -> None:
        """Open a span for a Skill invocation, or leave the one in progress.

        Args:
            matched: Whether the invoked skill is the one being excerpted.
            header: The span's first line, kept only when ``matched``.
        """
        self.in_span = matched
        self._span_lines = 0
        if matched:
            self.lines.append(header)

    def close(self) -> None:
        """End the current span, if any."""
        self.in_span = False

    def add(self, line: str) -> None:
        """Capture one line, inside a span and within its budget.

        Args:
            line: The formatted excerpt line.
        """
        if self.in_span and self._span_lines < MAX_LINES_PER_SPAN:
            self._span_lines += 1
            self.lines.append(line)


def _capture_results(span: _SpanBuffer, obj: dict[str, object]) -> None:
    """Capture the head of one tool result.

    Args:
        span: Buffer for the span in progress.
        obj: A ``user`` transcript line carrying a tool result.
    """
    for text in _result_texts(obj)[:1]:
        for result_line in text.splitlines()[:MAX_RESULT_LINES]:
            span.add(f"RESULT | {_clip(result_line)}")


def _capture_block(
    span: _SpanBuffer, block: dict[str, object], skill_name: str
) -> None:
    """Capture one assistant content block, opening a span when Skill runs.

    Args:
        span: Buffer for the span in progress.
        block: One content block of an ``assistant`` line.
        skill_name: The skill being excerpted.
    """
    block_type = block.get("type")
    if block_type == "tool_use":
        name = str(block.get("name") or "?")
        if name == "Skill":
            tool_input = block.get("input")
            invoked = tool_input.get("skill") if isinstance(tool_input, dict) else None
            span.start(
                matched=invoked == skill_name, header=f"TOOL Skill | {skill_name}"
            )
            return
        payload = _clip(json.dumps(block.get("input", {}))[:MAX_LINE_CHARS])
        span.add(f"TOOL {name} | {payload}")
    elif block_type == "text":
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            span.add(f"TEXT | {_clip(text, MAX_TEXT_CHARS)}")


def _extract_skill(path: Path, skill_name: str) -> list[str]:
    """Excerpt every span of one skill in a transcript.

    Args:
        path: Transcript path.
        skill_name: The skill to capture.

    Returns:
        TOOL/TEXT/RESULT lines for each span, capped per span.
    """
    span = _SpanBuffer()
    for obj in _iter_lines(path):
        if _is_external_prompt(obj):
            span.close()
        elif obj.get("type") == "user":
            if span.in_span:
                _capture_results(span, obj)
        elif obj.get("type") == "assistant":
            for block in _content_blocks(obj):
                _capture_block(span, block, skill_name)
    return span.lines


def _extract_error(path: Path, sig: str) -> list[str]:
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
    for obj in _iter_lines(path):
        for block in _content_blocks(obj) if obj.get("type") == "assistant" else []:
            if block.get("type") == "tool_use" and block.get("name") == "Bash":
                tool_input = block.get("input")
                command = (
                    tool_input.get("command") if isinstance(tool_input, dict) else None
                )
                if isinstance(command, str):
                    last_bash = command
        if obj.get("type") != "user" or hits >= MAX_ERR_HITS_PER_SESSION:
            continue
        for text in _result_texts(obj):
            for result_line in text.splitlines():
                if (
                    normalize_signature(result_line) == sig
                    and hits < MAX_ERR_HITS_PER_SESSION
                ):
                    hits += 1
                    lines.append(f"CMD | {_clip(last_bash)}")
                    lines.append(f"ERR | {_clip(result_line)}")
    return lines


def _extract_ngram(path: Path, sig: str) -> list[str]:
    """Excerpt the first occurrence of a command sequence.

    Args:
        path: Transcript path.
        sig: The " → "-joined command-signature gram.

    Returns:
        The raw commands of the first matching window, or empty.
    """
    gram = sig.split(" → ")
    window: list[tuple[str, str]] = []
    for obj in _iter_lines(path):
        for block in _content_blocks(obj) if obj.get("type") == "assistant" else []:
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
                return [f"CMD | {_clip(raw)}" for _, raw in window]
    return []


def _extract_prompt(path: Path, sig: str) -> list[str]:
    """Excerpt occurrences of a repeated typed prompt and what followed.

    Args:
        path: Transcript path.
        sig: The normalized prompt signature.

    Returns:
        PROMPT lines followed by the next few tool names.
    """
    lines: list[str] = []
    following = 0
    for obj in _iter_lines(path):
        if _is_external_prompt(obj):
            message = obj.get("message")
            content = str(message.get("content")) if isinstance(message, dict) else ""
            if normalize_signature(content) == sig:
                lines.append(f"PROMPT | {_clip(content, MAX_TEXT_CHARS)}")
                following = MAX_RESULT_LINES
            else:
                following = 0
            continue
        if following <= 0:
            continue
        for block in _content_blocks(obj) if obj.get("type") == "assistant" else []:
            if block.get("type") == "tool_use" and following > 0:
                following -= 1
                lines.append(f"THEN | {block.get('name')}")
    return lines
