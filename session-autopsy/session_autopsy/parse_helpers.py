"""Pure helpers for the transcript parser: field readers and line miners."""

from __future__ import annotations

from typing import TYPE_CHECKING

from session_autopsy.parse_constants import (
    ERROR_LINE_RE,
    GENERATIVE_INPUT_KEYS,
    GENERIC_ERROR_SIGS,
    RESULT_SCAN_LINES,
    RESULT_TAIL_LINES,
    RETRY_PROMPT_MAX_LEN,
)
from session_autopsy.signatures import normalize_signature

if TYPE_CHECKING:
    from collections import Counter

    from session_autopsy.records import SessionMeta


def note_title(meta: SessionMeta, obj: dict[str, object]) -> None:
    """Record the session's AI-generated title (last one wins).

    Args:
        meta: Session metadata to update.
        obj: An ``ai-title`` transcript line.
    """
    for key in ("title", "aiTitle"):
        value = obj.get(key)
        if isinstance(value, str) and value:
            meta.title = value


def note_metadata(meta: SessionMeta, obj: dict[str, object]) -> None:
    """Capture session metadata from the first message line carrying it.

    Args:
        meta: Session metadata to update.
        obj: A message-bearing transcript line.
    """
    timestamp = obj.get("timestamp")
    if isinstance(timestamp, str):
        if meta.started_at is None:
            meta.started_at = timestamp
        meta.ended_at = timestamp
    if meta.cwd is None and isinstance(obj.get("cwd"), str):
        meta.cwd = str(obj["cwd"])
    if meta.git_branch is None and isinstance(obj.get("gitBranch"), str):
        meta.git_branch = str(obj["gitBranch"])
    if meta.slug is None and isinstance(obj.get("slug"), str):
        meta.slug = str(obj["slug"])


def is_retry_prompt(content: str) -> bool:
    """Whether a typed prompt is a "limit reached, continue" retry.

    Args:
        content: The prompt text.

    Returns:
        True for short prompts mentioning a limit plus reset/continue.
    """
    lowered = content.lower()
    return (
        len(content) <= RETRY_PROMPT_MAX_LEN
        and "limit" in lowered
        and ("reset" in lowered or "continue" in lowered)
    )


def error_signatures(result: object) -> list[str]:
    """Mine a ``toolUseResult`` for repeated error-line signatures.

    Args:
        result: The top-level ``toolUseResult`` value (str, dict, or absent).

    Returns:
        One normalized signature per informative error line found.
    """
    texts: list[str] = []
    if isinstance(result, str):
        texts.append(result)
    elif isinstance(result, dict):
        for key in ("stderr", "stdout"):
            value = result.get(key)
            if isinstance(value, str) and value:
                texts.append(value)
    found: list[str] = []
    for text in texts:
        all_lines = text.splitlines()
        scan = all_lines[:RESULT_SCAN_LINES]
        if len(all_lines) > RESULT_SCAN_LINES:
            scan += all_lines[-RESULT_TAIL_LINES:]
        for line in scan:
            if ERROR_LINE_RE.search(line):
                sig = normalize_signature(line)
                if sig and sig not in GENERIC_ERROR_SIGS:
                    found.append(sig)
    return found


def generated_chars(block: dict[str, object]) -> int:
    """Characters of model-generated content hidden in a tool_use input.

    Write/Edit-style inputs carry the prose or code the model wrote; counting
    them stops generation-heavy skills from masquerading as tool-only.

    Args:
        block: A ``tool_use`` content block.

    Returns:
        Character count of generative input fields.
    """
    tool_input = block.get("input")
    if not isinstance(tool_input, dict):
        return 0
    return sum(
        len(value)
        for key in GENERATIVE_INPUT_KEYS
        if isinstance(value := tool_input.get(key), str)
    )


def usage_int(usage: object, key: str) -> int:
    """Read one int field from a possibly-absent usage dict.

    Args:
        usage: The ``message.usage`` value.
        key: Field name to read.

    Returns:
        The int value, or 0.
    """
    if isinstance(usage, dict):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


def top(counter: Counter[str], limit: int) -> dict[str, int]:
    """Keep only the most common counter entries, deterministically ordered.

    Args:
        counter: Any string counter.
        limit: Maximum number of entries to keep.

    Returns:
        A dict of the top entries, ties broken alphabetically.
    """
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return dict(ranked[:limit])


def message_id(message: dict[str, object]) -> str | None:
    """Fallback turn key for lines that carry no ``requestId``.

    Args:
        message: The ``message`` object of an assistant line.

    Returns:
        The message id when present, else None.
    """
    value = message.get("id")
    return value if isinstance(value, str) else None
