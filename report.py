"""Render REPORT.md, maintain state.json, and measure compiled skills."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from python_pkg.session_autopsy.stats import parse_timestamp

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from python_pkg.session_autopsy.detectors import AnalysisResult, Candidate
    from python_pkg.session_autopsy.records import SessionRecord

REPORT_FILE = "REPORT.md"
STATE_FILE = "state.json"
COMPILED_FILE = "compiled.json"
TRACES_HINT = "PYTHONPATH=~/testsAndMisc python3 -m python_pkg.session_autopsy traces"
THOUSAND = 1_000
MILLION = 1_000_000
TABLE_TOP = 40
DETAIL_TOP = 25


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


def render_report(
    records: list[SessionRecord],
    result: AnalysisResult,
    compiled: list[dict[str, object]],
    now: datetime,
) -> str:
    """Render the full REPORT.md content.

    Args:
        records: All session records.
        result: Detector output.
        compiled: Entries from compiled.json.
        now: Current UTC time.

    Returns:
        The Markdown document.
    """
    parts = [_header(records, now), _candidate_table(result.candidates)]
    parts.extend(
        _candidate_detail(candidate) for candidate in result.candidates[:DETAIL_TOP]
    )
    parts.append(_reviewed_keep_llm(compiled))
    parts.append(_trend_section(result))
    parts.append(
        "## Compiled scoreboard\n\n"
        + "\n".join(measure_lines(records, compiled, None))
        + "\n"
    )
    return "\n".join(parts)


def _header(records: list[SessionRecord], now: datetime) -> str:
    """Render the totals header.

    Args:
        records: All session records.
        now: Current UTC time.

    Returns:
        The header block.
    """
    turns = sum(record.counts.assistant_msgs for record in records)
    output = sum(record.tokens.output for record in records)
    cache_write = sum(record.tokens.cache_write for record in records)
    cache_read = sum(record.tokens.cache_read for record in records)
    per_turn = cache_read // turns if turns else 0
    return (
        "# Session Autopsy Report\n\n"
        f"Generated: {now.isoformat(timespec='seconds')} | Sessions: {len(records)} | "
        f"Assistant msgs: {turns}\n"
        f"Tokens: {fmt_tokens(output)} output | "
        f"{fmt_tokens(cache_write)} cache-write | "
        f"{fmt_tokens(cache_read)} cache-read (avg {fmt_tokens(per_turn)}/turn)\n"
    )


def _candidate_table(candidates: list[Candidate]) -> str:
    """Render the ranked candidate table.

    Args:
        candidates: Ranked candidates.

    Returns:
        The Markdown table (or a placeholder when empty).
    """
    if not candidates:
        return "## Ranked automation candidates\n\nnone yet — need more sessions\n"
    rows = [
        f"| {rank} | {cand.id} | {cand.kind} | "
        f"{cand.occurrences}x / {cand.sessions} sessions "
        f"| {fmt_tokens(cand.est_weekly_savings)}/wk | {cand.action} |"
        for rank, cand in enumerate(candidates[:TABLE_TOP], start=1)
    ]
    overflow = ""
    if len(candidates) > TABLE_TOP:
        overflow = (
            f"\n… plus {len(candidates) - TABLE_TOP} below the fold — "
            "see `candidates --json` for all.\n"
        )
    return (
        "## Ranked automation candidates\n\n"
        "| # | id | kind | evidence | est. weekly tokens | action |\n"
        "|---|----|------|----------|--------------------|--------|\n"
        + "\n".join(rows)
        + "\n"
        + overflow
    )


def _candidate_detail(candidate: Candidate) -> str:
    """Render one candidate's detail block.

    Args:
        candidate: The candidate.

    Returns:
        The detail block with the exact traces command.
    """
    return (
        f"### {candidate.id} — {candidate.title}\n\n"
        f"- score {candidate.score:.2f}, avg "
        f"{fmt_tokens(candidate.avg_tokens)} tokens/occurrence, "
        f"{candidate.per_week:.1f}/week\n"
        f"- signature: `{candidate.sig}`\n"
        f"- traces: `{TRACES_HINT} {candidate.id}`\n"
    )


def _reviewed_keep_llm(compiled: list[dict[str, object]]) -> str:
    """Render candidates already reviewed and ruled judgment-heavy.

    These come from /compile-candidate verdicts in compiled.json, not from a
    heuristic: real-trace review decided the workflow needs the LLM.

    Args:
        compiled: Entries from compiled.json.

    Returns:
        The section text.
    """
    kept = [entry for entry in compiled if entry.get("verdict") == "keep-llm"]
    if not kept:
        return "## Reviewed — keeping LLM\n\nnone reviewed yet\n"
    lines = [
        f"- {entry.get('candidate_id')} — "
        f"reviewed {entry.get('compiled_at')} — keep LLM"
        for entry in kept
    ]
    return "## Reviewed — keeping LLM\n\n" + "\n".join(lines) + "\n"


def _trend_section(result: AnalysisResult) -> str:
    """Render the context-bloat trend section.

    Args:
        result: Detector output holding the trend.

    Returns:
        The section text.
    """
    trend = result.trend
    if not trend.weekly:
        return "## Context-bloat trend\n\nno dated sessions yet\n"
    weeks = "\n".join(
        f"- {week}: {fmt_tokens(avg)}/turn" for week, avg in trend.weekly[-8:]
    )
    offenders = ", ".join(
        f"{name} ({fmt_tokens(avg)}/turn)" for name, avg in trend.top_sessions
    )
    return (
        "## Context-bloat trend\n\n"
        f"slope: {fmt_tokens(int(trend.slope_per_week))}/turn per week\n"
        f"heaviest sessions: {offenders}\n\n{weeks}\n"
    )


def write_report(
    home: Path,
    records: list[SessionRecord],
    result: AnalysisResult,
    now: datetime,
) -> int:
    """Write REPORT.md and refresh state.json.

    Args:
        home: The autopsy home directory.
        records: All session records.
        result: Detector output.
        now: Current UTC time.

    Returns:
        The unreviewed-candidate count (for the caller's summary line).
    """
    home.mkdir(parents=True, exist_ok=True)
    compiled = load_compiled(home)
    (home / REPORT_FILE).write_text(
        render_report(records, result, compiled, now), encoding="utf-8"
    )
    return write_state(home, [cand.id for cand in result.candidates], now)
