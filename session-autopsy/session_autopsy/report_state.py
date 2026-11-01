"""state.json and compiled.json handling for the autopsy report."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from session_autopsy.candidates import Candidate

STATE_FILE = "state.json"
COMPILED_FILE = "compiled.json"
# The ⚙ badge / SessionStart nudge count only candidates at least this big —
# nobody reviews 180 weak n-grams, so the count must mean "worth your time".
SIGNIFICANT_WEEKLY_TOKENS = 100_000


def handled_ids(compiled: list[dict[str, object]]) -> set[str]:
    """Candidate ids that already have a verdict (compiled, keep-llm, dropped).

    A handled candidate never counts toward the unreviewed nudge and its
    table row stops suggesting a review — its story lives in the Reviewed
    section and the scoreboard instead.

    Args:
        compiled: Entries from compiled.json.

    Returns:
        The candidate ids with any recorded verdict.
    """
    return {
        str(entry.get("candidate_id"))
        for entry in compiled
        if entry.get("candidate_id")
    }


def significant_ids(candidates: list[Candidate]) -> list[str]:
    """Ids of the candidates big enough to surface in the badge.

    Args:
        candidates: Ranked candidates.

    Returns:
        Ids with at least :data:`SIGNIFICANT_WEEKLY_TOKENS` estimated weekly
        tokens at stake, in rank order.
    """
    return [
        cand.id
        for cand in candidates
        if cand.est_weekly_savings >= SIGNIFICANT_WEEKLY_TOKENS
    ]


def load_state(home: Path) -> dict[str, object]:
    """Read state.json, tolerating absence and corruption.

    Args:
        home: The autopsy home directory.

    Returns:
        The stored state dict, or an empty dict.
    """
    path = home / STATE_FILE
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        return data
    return {}


def load_compiled(home: Path) -> list[dict[str, object]]:
    """Read compiled.json entries, tolerating absence and corruption.

    Args:
        home: The autopsy home directory.

    Returns:
        The list of ``{candidate_id, skill, script, compiled_at}`` entries.
    """
    path = home / COMPILED_FILE
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]
    return []


def write_state(
    home: Path, candidate_ids: list[str], now: datetime, *, mark_reviewed: bool = False
) -> int:
    """Update state.json and return the unreviewed-candidate count.

    Args:
        home: The autopsy home directory.
        candidate_ids: Every current candidate id, ranked.
        now: Current UTC time.
        mark_reviewed: When True, every current candidate becomes reviewed.

    Returns:
        The number of candidates not yet marked reviewed.
    """
    state = load_state(home)
    stored = state.get("reviewed_ids")
    reviewed = (
        {item for item in stored if isinstance(item, str)}
        if isinstance(stored, list)
        else set()
    )
    if mark_reviewed:
        reviewed = set(candidate_ids)
    reviewed &= set(candidate_ids)
    unreviewed = len([cid for cid in candidate_ids if cid not in reviewed])
    payload = {
        "unreviewed_count": unreviewed,
        "candidate_ids": candidate_ids,
        "reviewed_ids": sorted(reviewed),
        "last_generated": now.isoformat(timespec="seconds"),
    }
    home.mkdir(parents=True, exist_ok=True)
    (home / STATE_FILE).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return unreviewed
