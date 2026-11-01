"""Transcript discovery and per-file parsing for ``scan``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from session_autopsy.cli_io import emit_err
from session_autopsy.parse import parse_session

if TYPE_CHECKING:
    from pathlib import Path

    from session_autopsy.records import SessionRecord


def find_transcripts(root: Path) -> list[Path]:
    """List every top-level session transcript under the projects dir.

    Subagent transcripts (``**/subagents/agent-*.jsonl``) are excluded;
    they are folded into their parent session by the parser.

    Args:
        root: The projects directory.

    Returns:
        Sorted transcript paths.
    """
    return sorted(path for path in root.glob("*/*.jsonl") if path.is_file())


def needs_analysis(path: Path, index: dict[str, tuple[int, float]]) -> bool:
    """Decide whether a transcript changed since its stored record.

    Args:
        path: Transcript path.
        index: Stored ``{path: (size, mtime)}`` map.

    Returns:
        True when the transcript is new or its size/mtime moved.
    """
    stored = index.get(str(path))
    if stored is None:
        return True
    stat = path.stat()
    return (stat.st_size, stat.st_mtime) != stored


def parse_one(path: Path) -> SessionRecord | None:
    """Worker-side parse wrapper that never raises across the pool.

    Args:
        path: Transcript path to parse.

    Returns:
        The record, or None when the file vanished or was unreadable.
    """
    try:
        return parse_session(path)
    except OSError as error:
        emit_err(f"scan: skipping {path}: {error}")
        return None
