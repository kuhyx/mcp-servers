"""Skill-span excerpting: the SpanBuffer and the transcript walk using it."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from session_autopsy.trace_io import (
    MAX_LINE_CHARS,
    MAX_RESULT_LINES,
    MAX_TEXT_CHARS,
    clip,
    content_blocks,
    is_external_prompt,
    iter_lines,
    result_texts,
)

if TYPE_CHECKING:
    from pathlib import Path

MAX_LINES_PER_SPAN = 200


class SpanBuffer:
    """Excerpt lines for the skill spans found in one transcript.

    Owning ``in_span`` and the per-span line budget here is what keeps
    :func:`extract_skill` down to a readable walk: every "am I inside a span and
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


def capture_results(span: SpanBuffer, obj: dict[str, object]) -> None:
    """Capture the head of one tool result.

    Args:
        span: Buffer for the span in progress.
        obj: A ``user`` transcript line carrying a tool result.
    """
    for text in result_texts(obj)[:1]:
        for result_line in text.splitlines()[:MAX_RESULT_LINES]:
            span.add(f"RESULT | {clip(result_line)}")


def capture_block(span: SpanBuffer, block: dict[str, object], skill_name: str) -> None:
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
        payload = clip(json.dumps(block.get("input", {}))[:MAX_LINE_CHARS])
        span.add(f"TOOL {name} | {payload}")
    elif block_type == "text":
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            span.add(f"TEXT | {clip(text, MAX_TEXT_CHARS)}")


def extract_skill(path: Path, skill_name: str) -> list[str]:
    """Excerpt every span of one skill in a transcript.

    Args:
        path: Transcript path.
        skill_name: The skill to capture.

    Returns:
        TOOL/TEXT/RESULT lines for each span, capped per span.
    """
    span = SpanBuffer()
    for obj in iter_lines(path):
        if is_external_prompt(obj):
            span.close()
        elif obj.get("type") == "user":
            if span.in_span:
                capture_results(span, obj)
        elif obj.get("type") == "assistant":
            for block in content_blocks(obj):
                capture_block(span, block, skill_name)
    return span.lines
