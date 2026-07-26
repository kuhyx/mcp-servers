"""Cell labels for the candidate table in REPORT.md.

Split out of ``report.py`` to keep it inside the repo's 500-line-per-file
limit. Both functions are pure label logic with no formatting dependencies,
which is why these two came out and the rest of the table rendering did not:
the detail block needs ``fmt_tokens`` and would make the import circular.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python_pkg.session_autopsy.detectors import Candidate


def action_cell(cand: Candidate, handled: set[str]) -> str:
    """The table's action column for one candidate.

    Args:
        cand: The candidate.
        handled: Ids with a recorded verdict.

    Returns:
        The suggested action, or a pointer to where its verdict lives.
    """
    if cand.id in handled:
        return "already handled — see Reviewed/scoreboard"
    return cand.action


def verdict_label(entry: dict[str, object]) -> str:
    """Human label for a reviewed entry's verdict.

    Args:
        entry: A compiled.json entry.

    Returns:
        ``"dropped (workflow removed)"`` or ``"keep LLM"``.
    """
    if entry.get("verdict") == "dropped":
        return "dropped (workflow removed)"
    return "keep LLM"
