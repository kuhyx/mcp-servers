"""N-gram detector: repeated Bash command sequences across sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from session_autopsy import stats
from session_autopsy.candidates import Candidate, hex_id, recent_session_ids

if TYPE_CHECKING:
    from datetime import datetime

    from session_autopsy.records import SessionRecord

NGRAM_SIZES = range(3, 7)
NGRAM_MIN_OCCURRENCES = 8
NGRAM_MIN_SESSIONS = 4
NGRAM_MIN_DISTINCT_RATIO = 0.5


def ngram_candidates(
    records: list[SessionRecord], now: datetime, per_turn_avg: int
) -> list[Candidate]:
    """Repeated cross-session Bash command sequences outside skill spans.

    Args:
        records: All session records.
        now: Current UTC time.
        per_turn_avg: Corpus-wide tokens per assistant turn.

    Returns:
        Sequence candidates, longest surviving grams only.
    """
    counts: dict[tuple[str, ...], int] = {}
    contributors: dict[tuple[str, ...], list[SessionRecord]] = {}
    for record in records:
        seen_here: set[tuple[str, ...]] = set()
        for sequence in record.obs.bash_sequences:
            for size in NGRAM_SIZES:
                for start in range(len(sequence) - size + 1):
                    gram = tuple(sequence[start : start + size])
                    counts[gram] = counts.get(gram, 0) + 1
                    if gram not in seen_here:
                        seen_here.add(gram)
                        contributors.setdefault(gram, []).append(record)
    surviving = {
        gram: count
        for gram, count in counts.items()
        if count >= NGRAM_MIN_OCCURRENCES
        and len(contributors[gram]) >= NGRAM_MIN_SESSIONS
        and len(set(gram)) / len(gram) >= NGRAM_MIN_DISTINCT_RATIO
    }
    top = drop_subgrams(surviving)
    candidates = []
    for gram in sorted(top):
        sessions = contributors[gram]
        joined = " → ".join(gram)
        occurrences = counts[gram]
        per_week = stats.per_week(
            [stats.parse_timestamp(record.meta.started_at) for record in sessions], now
        )
        candidates.append(
            Candidate(
                id=f"ngram-{hex_id(joined)}",
                kind="ngram",
                title=joined[:70],
                sig=joined,
                sessions=len({record.session_id for record in sessions}),
                occurrences=occurrences,
                avg_tokens=per_turn_avg * len(gram),
                per_week=per_week,
                est_weekly_savings=int(per_turn_avg * len(gram) * per_week),
                score=1.0,
                action="script this sequence",
                session_ids=recent_session_ids(sessions),
            ),
        )
    return candidates


def drop_subgrams(surviving: dict[tuple[str, ...], int]) -> set[tuple[str, ...]]:
    """Remove grams contained in a longer surviving gram.

    Args:
        surviving: Grams that crossed the frequency thresholds.

    Returns:
        The maximal grams only.
    """
    grams = sorted(surviving, key=len, reverse=True)
    kept: set[tuple[str, ...]] = set()
    for gram in grams:
        if not any(contains(longer, gram) for longer in kept):
            kept.add(gram)
    return kept


def contains(longer: tuple[str, ...], shorter: tuple[str, ...]) -> bool:
    """Whether ``shorter`` appears contiguously inside ``longer``.

    Args:
        longer: The candidate container gram.
        shorter: The gram to look for.

    Returns:
        True when contained.
    """
    if len(shorter) >= len(longer):
        return False
    return any(
        longer[i : i + len(shorter)] == shorter
        for i in range(len(longer) - len(shorter) + 1)
    )
