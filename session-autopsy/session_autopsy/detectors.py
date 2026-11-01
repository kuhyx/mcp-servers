"""Pure heuristics turning session records into ranked automation candidates.

No LLM anywhere: the compile-worthiness score comes from turn composition and
cross-invocation command similarity, savings from measured token usage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from session_autopsy import stats
from session_autopsy.candidates import (
    AnalysisResult,
    Candidate,
    TrendInfo,
    hex_id,
    recent_session_ids,
)
from session_autopsy.detect_ngrams import ngram_candidates
from session_autopsy.detect_skills import skill_candidates

if TYPE_CHECKING:
    from datetime import datetime

    from session_autopsy.records import SessionRecord

ERROR_MIN_OCCURRENCES = 10
ERROR_MIN_SESSIONS = 3
PROMPT_MIN_OCCURRENCES = 4
PROMPT_MIN_WORDS = 3
TREND_TOP_SESSIONS = 3


@dataclass(frozen=True)
class _CounterSpec:
    """Which signature-counter field to mine, and the thresholds to mine it at."""

    field_name: str
    """Attribute on ``record.obs`` holding the ``sig -> count`` map."""
    kind: str
    """Candidate kind tag."""
    min_occurrences: int
    """Total-count threshold."""
    min_sessions: int
    """Distinct-session threshold (ORed with the occurrence threshold)."""
    action: str
    """Suggested action text."""


_ERROR_SPEC = _CounterSpec(
    field_name="error_signatures",
    kind="err",
    min_occurrences=ERROR_MIN_OCCURRENCES,
    min_sessions=ERROR_MIN_SESSIONS,
    action="fix once: eliminate the failure's root cause",
)
_PROMPT_SPEC = _CounterSpec(
    field_name="typed_prompt_signatures",
    kind="prompt",
    min_occurrences=PROMPT_MIN_OCCURRENCES,
    min_sessions=1,
    action="promote to slash command or script",
)


def analyze(records: list[SessionRecord], now: datetime) -> AnalysisResult:
    """Run every detector and rank the merged candidates.

    Args:
        records: All stored session records.
        now: Current UTC time (injected for determinism in tests).

    Returns:
        Ranked candidates and the context trend.
    """
    per_turn_avg = stats.per_turn_avg(records)
    candidates = [
        *skill_candidates(records, now),
        *ngram_candidates(records, now, per_turn_avg),
        *_error_candidates(records, now, per_turn_avg),
        *_prompt_candidates(records, now, per_turn_avg),
    ]
    candidates.sort(key=lambda cand: (-cand.est_weekly_savings, cand.id))
    return AnalysisResult(candidates=candidates, trend=_trend(records))


def _counter_candidates(
    records: list[SessionRecord],
    now: datetime,
    per_turn_avg: int,
    spec: _CounterSpec,
) -> list[Candidate]:
    """Shared aggregation for signature-counter fields (errors, prompts).

    Args:
        records: All session records.
        now: Current UTC time.
        per_turn_avg: Corpus-wide tokens per assistant turn.
        spec: Which counter field to mine, and at what thresholds.

    Returns:
        Candidates for every signature crossing either threshold.
    """
    totals: dict[str, int] = {}
    contributors: dict[str, list[SessionRecord]] = {}
    for record in records:
        counter: dict[str, int] = getattr(record.obs, spec.field_name)
        for sig, count in counter.items():
            totals[sig] = totals.get(sig, 0) + count
            contributors.setdefault(sig, []).append(record)
    candidates = []
    for sig, total in sorted(totals.items()):
        sessions = contributors[sig]
        if total < spec.min_occurrences and len(sessions) < spec.min_sessions:
            continue
        timestamps = [
            stats.parse_timestamp(record.meta.started_at) for record in sessions
        ]
        per_week = stats.per_week(timestamps, now) * (total / len(sessions))
        candidates.append(
            Candidate(
                id=f"{spec.kind}-{hex_id(sig)}",
                kind=spec.kind,
                title=sig[:70],
                sig=sig,
                sessions=len({record.session_id for record in sessions}),
                occurrences=total,
                avg_tokens=per_turn_avg,
                per_week=per_week,
                est_weekly_savings=int(per_turn_avg * per_week),
                score=1.0,
                action=spec.action,
                session_ids=recent_session_ids(sessions),
            ),
        )
    return candidates


def _error_candidates(
    records: list[SessionRecord], now: datetime, per_turn_avg: int
) -> list[Candidate]:
    """Repeated error signatures worth a fix-once script.

    Args:
        records: All session records.
        now: Current UTC time.
        per_turn_avg: Corpus-wide tokens per assistant turn (crude cost proxy).

    Returns:
        Fix-once candidates.
    """
    return _counter_candidates(records, now, per_turn_avg, _ERROR_SPEC)


def _prompt_candidates(
    records: list[SessionRecord], now: datetime, per_turn_avg: int
) -> list[Candidate]:
    """Repeated typed prompts worth promoting to a command or script.

    Args:
        records: All session records.
        now: Current UTC time.
        per_turn_avg: Corpus-wide tokens per assistant turn.

    Returns:
        Prompt-promotion candidates (short generic prompts filtered out).
    """
    raw = _counter_candidates(records, now, per_turn_avg, _PROMPT_SPEC)
    return [
        cand
        for cand in raw
        if len(cand.sig.split()) >= PROMPT_MIN_WORDS
        and cand.occurrences >= PROMPT_MIN_OCCURRENCES
    ]


def _trend(records: list[SessionRecord]) -> TrendInfo:
    """Weekly cache-read-per-turn trend plus the heaviest sessions.

    Args:
        records: All session records.

    Returns:
        The populated trend info (empty when timestamps are missing).
    """
    weekly_turns: dict[str, int] = {}
    weekly_reads: dict[str, int] = {}
    heaviest: list[tuple[str, int]] = []
    for record in records:
        stamp = stats.parse_timestamp(record.meta.started_at)
        if stamp is None or record.counts.assistant_msgs == 0:
            continue
        week = f"{stamp.isocalendar().year}-W{stamp.isocalendar().week:02d}"
        weekly_turns[week] = weekly_turns.get(week, 0) + record.counts.assistant_msgs
        weekly_reads[week] = weekly_reads.get(week, 0) + record.tokens.cache_read
        heaviest.append(
            (
                record.meta.slug or record.session_id,
                record.tokens.cache_read // record.counts.assistant_msgs,
            )
        )
    weekly = [
        (week, weekly_reads[week] // weekly_turns[week])
        for week in sorted(weekly_turns)
    ]
    heaviest.sort(key=lambda item: -item[1])
    return TrendInfo(
        weekly=weekly,
        slope_per_week=stats.slope([float(avg) for _, avg in weekly]),
        top_sessions=heaviest[:TREND_TOP_SESSIONS],
    )
