"""Stable text signatures for messages, commands and errors.

Two modules need these: :mod:`python_pkg.session_autopsy.parse` while streaming
a transcript, and :mod:`python_pkg.session_autopsy.traces` when re-reading one to
pull evidence. They live here rather than in either of those so neither has to
import the other.
"""

from __future__ import annotations

import re

MAX_SIG_LEN = 120

_PATH_RE = re.compile(r"(?:~|/)[\w.@+/-]{4,}")
_HEX_RE = re.compile(r"[0-9a-fA-F]{8,}")
_NUM_RE = re.compile(r"\b\d+\b")
_WS_RE = re.compile(r"\s+")
# Wrapper commands whose real command is the following word.
_WRAPPERS = frozenset({"cd", "sudo", "env", "nice", "command", "timeout"})


def normalize_signature(text: str) -> str:
    """Collapse a message/command/error into a stable cross-session signature.

    Takes the first line only and masks paths, long hex runs, and numbers so
    that reoccurrences of the same event count as one signature.

    Args:
        text: Raw text (may be multi-line).

    Returns:
        The normalized first line, truncated to :data:`MAX_SIG_LEN` chars.
    """
    stripped = text.strip()
    if not stripped:
        return ""
    first = stripped.splitlines()[0]
    first = _PATH_RE.sub("<PATH>", first)
    first = _HEX_RE.sub("<HEX>", first)
    first = _NUM_RE.sub("<N>", first)
    return _WS_RE.sub(" ", first).strip()[:MAX_SIG_LEN]


def command_signature(command: str) -> str:
    """Reduce a Bash command to its leading command word(s).

    Wrapper commands (``cd``, ``sudo``, ``timeout`` …) keep their first
    non-flag, non-numeric argument so sequences stay meaningful.

    Args:
        command: The raw (possibly multi-line) Bash tool input.

    Returns:
        A short signature such as ``"git"`` or ``"cd <PATH>"``, or ``""``.
    """
    normalized = normalize_signature(command)
    words = normalized.split()
    if not words:
        return ""
    head = words[0]
    if head not in _WRAPPERS:
        return head
    for word in words[1:]:
        if not word.startswith(("-", "<N>")):
            return f"{head} {word}"
    return head
