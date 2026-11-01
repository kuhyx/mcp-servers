"""Reading helpers for transcript excerpts: line iteration, clipping, blocks."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

MAX_TEXT_CHARS = 200
MAX_RESULT_LINES = 3
MAX_LINE_CHARS = 160


def iter_lines(path: Path) -> Iterator[dict[str, object]]:
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


def clip(text: str, limit: int = MAX_LINE_CHARS) -> str:
    """Single-line clip for trace output.

    Args:
        text: Raw text.
        limit: Character cap.

    Returns:
        The first line, capped.
    """
    first = text.strip().splitlines()[0] if text.strip() else ""
    return first[:limit]


def content_blocks(obj: dict[str, object]) -> list[dict[str, object]]:
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


def is_external_prompt(obj: dict[str, object]) -> bool:
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


def result_texts(obj: dict[str, object]) -> list[str]:
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
