"""Filesystem locations for the session-autopsy store.

Both locations are overridable via environment variables so tests (and any
future relocation) never touch the real ``~/.claude`` tree.
"""

from __future__ import annotations

import os
from pathlib import Path


def autopsy_home() -> Path:
    """Return the directory holding sessions.jsonl, REPORT.md and state.json.

    Returns:
        The ``CLAUDE_AUTOPSY_HOME`` override or ``~/.claude/autopsy``.
    """
    return Path(os.environ.get("CLAUDE_AUTOPSY_HOME", "~/.claude/autopsy")).expanduser()


def projects_dir() -> Path:
    """Return the directory containing Claude Code session transcripts.

    Returns:
        The ``CLAUDE_PROJECTS_DIR`` override or ``~/.claude/projects``.
    """
    return Path(
        os.environ.get("CLAUDE_PROJECTS_DIR", "~/.claude/projects")
    ).expanduser()
