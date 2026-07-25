"""Small numeric and time helpers shared by the detectors and the report.

These are pure functions over plain numbers, timestamps and token totals — no
knowledge of what a candidate is. Keeping them here leaves
:mod:`python_pkg.session_autopsy.detectors` to the heuristics themselves.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python_pkg.session_autopsy.records import SessionRecord

TRAILING_WEEKS = 8
MAX_SIMILARITY_INVOCATIONS = 5
MIN_SLOPE_SAMPLES = 2


def parse_timestamp(value: str | None) -> datetime | None:
    """Parse an ISO-8601 transcript timestamp (with trailing ``Z``).

    Args:
        value: The timestamp string, or None.

    Returns:
        A timezone-aware datetime, or None when absent/invalid.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def per_turn_avg(records: list[SessionRecord]) -> int:
    """Average output+cache-write tokens per assistant turn, corpus-wide.

    Args:
        records: All session records.

    Returns:
        The average, or 0 for an empty corpus.
    """
    turns = sum(record.counts.assistant_msgs for record in records)
    if turns == 0:
        return 0
    spent = sum(record.tokens.output + record.tokens.cache_write for record in records)
    return spent // turns


def per_week(timestamps: list[datetime | None], now: datetime) -> float:
    """Occurrences per week over the trailing window.

    Args:
        timestamps: One entry per occurrence (None when the session had no
            parseable start time; those never count as recent).
        now: Current UTC time.

    Returns:
        Recent occurrences divided by :data:`TRAILING_WEEKS`.
    """
    cutoff = now - timedelta(weeks=TRAILING_WEEKS)
    recent = sum(1 for stamp in timestamps if stamp is not None and stamp >= cutoff)
    return recent / TRAILING_WEEKS


def mean(values: list[float]) -> float:
    """Arithmetic mean, 0.0 for an empty list.

    Args:
        values: Sample values.

    Returns:
        The mean.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def mean_pairwise_similarity(sequences: list[list[str]]) -> float:
    """Mean SequenceMatcher ratio over the first few invocation pairs.

    Args:
        sequences: One Bash-signature sequence per invocation.

    Returns:
        Mean similarity in [0, 1]; 0.0 when fewer than two non-empty
        sequences exist (nothing to compare means no evidence of stability).
    """
    sample = [seq for seq in sequences if seq][:MAX_SIMILARITY_INVOCATIONS]
    ratios = [
        SequenceMatcher(None, sample[i], sample[j]).ratio()
        for i in range(len(sample))
        for j in range(i + 1, len(sample))
    ]
    return mean(ratios)


def slope(values: list[float]) -> float:
    """Least-squares slope of evenly spaced samples.

    Args:
        values: One sample per week, in order.

    Returns:
        Tokens-per-turn change per week; 0.0 with fewer than two samples.
    """
    count = len(values)
    if count < MIN_SLOPE_SAMPLES:
        return 0.0
    mean_x = (count - 1) / 2
    mean_y = sum(values) / count
    numerator = sum((i - mean_x) * (val - mean_y) for i, val in enumerate(values))
    denominator = sum((i - mean_x) ** 2 for i in range(count))
    return numerator / denominator
