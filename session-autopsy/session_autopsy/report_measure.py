"""Token formatting and the before/after scoreboard for compiled skills."""

from __future__ import annotations

from typing import TYPE_CHECKING

from session_autopsy.stats import parse_timestamp

if TYPE_CHECKING:
    from session_autopsy.records import SessionRecord

THOUSAND = 1_000
MILLION = 1_000_000


def fmt_tokens(count: int) -> str:
    """Format a token count for humans.

    Args:
        count: Token count.

    Returns:
        ``"842"``, ``"74k"``, or ``"1.2M"``.
    """
    if count >= MILLION:
        return f"{count / MILLION:.1f}M"
    if count >= THOUSAND:
        return f"{count // THOUSAND}k"
    return str(count)


def measure_lines(
    records: list[SessionRecord],
    compiled: list[dict[str, object]],
    only_skill: str | None,
) -> list[str]:
    """Compare tokens/invocation before vs after each compilation.

    Args:
        records: All session records.
        compiled: Entries from compiled.json.
        only_skill: Restrict output to one skill name, when given.

    Returns:
        One human-readable line per compiled skill.
    """
    lines = []
    for entry in compiled:
        skill = str(entry.get("skill") or "")
        if (
            not skill
            or not entry.get("script")
            or (only_skill is not None and skill != only_skill)
        ):
            continue
        compiled_at = parse_timestamp(str(entry.get("compiled_at") or "") or None)
        before: list[int] = []
        after: list[int] = []
        for record in records:
            stamp = parse_timestamp(record.meta.started_at)
            for invocation in record.obs.skill_invocations:
                if invocation.name != skill:
                    continue
                spent = invocation.tokens_output + invocation.tokens_cache_write
                if (
                    compiled_at is not None
                    and stamp is not None
                    and stamp >= compiled_at
                ):
                    after.append(spent)
                else:
                    before.append(spent)
        lines.append(_measure_line(skill, before, after))
    if not lines:
        lines.append(
            "nothing compiled yet — run /compile-candidate on a ranked candidate"
        )
    return lines


def _measure_line(skill: str, before: list[int], after: list[int]) -> str:
    """Render one scoreboard line.

    Args:
        skill: Skill name.
        before: Tokens/invocation samples before compilation.
        after: Samples after compilation.

    Returns:
        The formatted line, with a delta when both sides have data.
    """
    avg_before = sum(before) // len(before) if before else 0
    if not after:
        return (
            f"- {skill}: before {fmt_tokens(avg_before)}/inv "
            f"(n={len(before)}) — no post-compile invocations yet"
        )
    avg_after = sum(after) // len(after)
    delta = ""
    if avg_before > 0:
        delta = f" — {100 * (avg_after - avg_before) // avg_before:+d}%"
    return (
        f"- {skill}: before {fmt_tokens(avg_before)}/inv (n={len(before)}) "
        f"→ after {fmt_tokens(avg_after)}/inv (n={len(after)}){delta}"
    )
