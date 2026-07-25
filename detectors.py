"""Pure heuristics turning session records into ranked automation candidates.

No LLM anywhere: the compile-worthiness score comes from turn composition and
cross-invocation command similarity, savings from measured token usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import TYPE_CHECKING

from python_pkg.session_autopsy import stats

if TYPE_CHECKING:
    from datetime import datetime

    from python_pkg.session_autopsy.records import SessionRecord, SkillInvocation

MIN_SKILL_INVOCATIONS = 3
NGRAM_SIZES = range(3, 7)
NGRAM_MIN_OCCURRENCES = 8
NGRAM_MIN_SESSIONS = 4
NGRAM_MIN_DISTINCT_RATIO = 0.5
ERROR_MIN_OCCURRENCES = 10
ERROR_MIN_SESSIONS = 3
PROMPT_MIN_OCCURRENCES = 4
PROMPT_MIN_WORDS = 3
MAX_EVIDENCE_SESSIONS = 10
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


@dataclass
class Candidate:
    """One ranked automation opportunity."""

    id: str
    kind: str
    title: str
    sig: str
    sessions: int
    occurrences: int
    avg_tokens: int
    per_week: float
    est_weekly_savings: int
    score: float
    action: str
    session_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Serialize for ``candidates --json``.

        Returns:
            A JSON-compatible dict of all fields.
        """
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "sig": self.sig,
            "sessions": self.sessions,
            "occurrences": self.occurrences,
            "avg_tokens": self.avg_tokens,
            "per_week": round(self.per_week, 2),
            "est_weekly_savings": self.est_weekly_savings,
            "score": round(self.score, 3),
            "action": self.action,
            "session_ids": self.session_ids,
        }


@dataclass
class TrendInfo:
    """Weekly cache-read-per-turn averages and their direction."""

    weekly: list[tuple[str, int]] = field(default_factory=list)
    slope_per_week: float = 0.0
    top_sessions: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Everything the report renders: ranked candidates and the trend."""

    candidates: list[Candidate] = field(default_factory=list)
    trend: TrendInfo = field(default_factory=TrendInfo)


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
        *_skill_candidates(records, now),
        *_ngram_candidates(records, now, per_turn_avg),
        *_error_candidates(records, now, per_turn_avg),
        *_prompt_candidates(records, now, per_turn_avg),
    ]
    candidates.sort(key=lambda cand: (-cand.est_weekly_savings, cand.id))
    return AnalysisResult(candidates=candidates, trend=_trend(records))


def _hex_id(text: str) -> str:
    """Derive the stable 8-hex candidate id suffix for a signature.

    Args:
        text: The normalized signature.

    Returns:
        First 8 hex chars of its SHA-256.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def _skill_candidates(records: list[SessionRecord], now: datetime) -> list[Candidate]:
    """Rank every skill with enough invocations by measured spend.

    No compile-vs-keep verdict is attempted here: real-corpus probing showed no
    cheap scalar separates mechanical skills from judgment-heavy ones (a giant
    Write turn hides among mechanical read turns). The score is published as
    evidence; the verdict belongs to /compile-candidate reading real traces.

    Args:
        records: All session records.
        now: Current UTC time.

    Returns:
        One candidate per skill with at least MIN_SKILL_INVOCATIONS uses.
    """
    by_skill: dict[str, list[tuple[SessionRecord, SkillInvocation]]] = {}
    for record in records:
        for invocation in record.obs.skill_invocations:
            by_skill.setdefault(invocation.name, []).append((record, invocation))
    return [
        _score_skill(name, pairs, now)
        for name, pairs in sorted(by_skill.items())
        if len(pairs) >= MIN_SKILL_INVOCATIONS
    ]


def _score_skill(
    name: str,
    pairs: list[tuple[SessionRecord, SkillInvocation]],
    now: datetime,
) -> Candidate:
    """Build the candidate for one skill from all its invocations.

    Args:
        name: Skill name.
        pairs: (session record, invocation) for every invocation.
        now: Current UTC time.

    Returns:
        The scored candidate (score = determinism * similarity).
    """
    determinism = stats.mean(
        [
            invocation.tool_only_turns / total
            for _, invocation in pairs
            if (total := invocation.tool_only_turns + invocation.text_turns) > 0
        ],
    )
    similarity = stats.mean_pairwise_similarity(
        [invocation.bash_sig for _, invocation in pairs]
    )
    score = determinism * similarity
    avg_tokens = int(
        stats.mean(
            [float(inv.tokens_output + inv.tokens_cache_write) for _, inv in pairs]
        )
    )
    per_week = stats.per_week(
        [stats.parse_timestamp(record.meta.started_at) for record, _ in pairs], now
    )
    session_ids = _recent_session_ids([record for record, _ in pairs])
    action = f"review for compilation: /compile-candidate skill-{name}"
    return Candidate(
        id=f"skill-{name}",
        kind="skill",
        title=f"Skill '{name}' ({len(pairs)} invocations)",
        sig=name,
        sessions=len({record.session_id for record, _ in pairs}),
        occurrences=len(pairs),
        avg_tokens=avg_tokens,
        per_week=per_week,
        est_weekly_savings=int(avg_tokens * per_week),
        score=score,
        action=action,
        session_ids=session_ids,
    )


def _recent_session_ids(records: list[SessionRecord]) -> list[str]:
    """Most recent contributing session ids for the traces subcommand.

    Args:
        records: Contributing session records (may repeat).

    Returns:
        Up to :data:`MAX_EVIDENCE_SESSIONS` unique ids, newest first.
    """
    unique: dict[str, str] = {}
    for record in records:
        unique[record.session_id] = record.meta.started_at or ""
    ranked = sorted(unique.items(), key=lambda item: item[1], reverse=True)
    return [session_id for session_id, _ in ranked[:MAX_EVIDENCE_SESSIONS]]


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
                id=f"{spec.kind}-{_hex_id(sig)}",
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
                session_ids=_recent_session_ids(sessions),
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


def _ngram_candidates(
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
    top = _drop_subgrams(surviving)
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
                id=f"ngram-{_hex_id(joined)}",
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
                session_ids=_recent_session_ids(sessions),
            ),
        )
    return candidates


def _drop_subgrams(surviving: dict[tuple[str, ...], int]) -> set[tuple[str, ...]]:
    """Remove grams contained in a longer surviving gram.

    Args:
        surviving: Grams that crossed the frequency thresholds.

    Returns:
        The maximal grams only.
    """
    grams = sorted(surviving, key=len, reverse=True)
    kept: set[tuple[str, ...]] = set()
    for gram in grams:
        if not any(_contains(longer, gram) for longer in kept):
            kept.add(gram)
    return kept


def _contains(longer: tuple[str, ...], shorter: tuple[str, ...]) -> bool:
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
