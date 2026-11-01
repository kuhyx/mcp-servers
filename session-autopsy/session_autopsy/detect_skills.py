"""Skill detector: rank skills by measured spend across their invocations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from session_autopsy import stats
from session_autopsy.candidates import Candidate, recent_session_ids

if TYPE_CHECKING:
    from datetime import datetime

    from session_autopsy.records import SessionRecord, SkillInvocation

MIN_SKILL_INVOCATIONS = 3


def skill_candidates(records: list[SessionRecord], now: datetime) -> list[Candidate]:
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
        score_skill(name, pairs, now)
        for name, pairs in sorted(by_skill.items())
        if len(pairs) >= MIN_SKILL_INVOCATIONS
    ]


def score_skill(
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
    session_ids = recent_session_ids([record for record, _ in pairs])
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
