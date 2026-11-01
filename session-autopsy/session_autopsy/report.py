"""Render REPORT.md and write it (with state.json) to the autopsy home."""

from __future__ import annotations

from typing import TYPE_CHECKING

from session_autopsy.report_cells import action_cell, verdict_label
from session_autopsy.report_measure import fmt_tokens, measure_lines
from session_autopsy.report_state import (
    handled_ids,
    load_compiled,
    significant_ids,
    write_state,
)
from session_autopsy.report_turns import turn_efficiency

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from session_autopsy.candidates import AnalysisResult, Candidate
    from session_autopsy.records import SessionRecord

REPORT_FILE = "REPORT.md"
TRACES_HINT = "cd ~/src/mcp-servers/session-autopsy && PYTHONPATH=. python3 -m session_autopsy traces"
TABLE_TOP = 40
DETAIL_TOP = 25


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
    handled = handled_ids(compiled)
    parts = [
        _header(records, now),
        turn_efficiency(records),
        _candidate_table(result.candidates, handled),
    ]
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


def _candidate_table(candidates: list[Candidate], handled: set[str]) -> str:
    """Render the ranked candidate table.

    Args:
        candidates: Ranked candidates.
        handled: Ids with a recorded verdict — their action column stops
            suggesting a review.

    Returns:
        The Markdown table (or a placeholder when empty).
    """
    if not candidates:
        return "## Ranked automation candidates\n\nnone yet — need more sessions\n"
    rows = [
        f"| {rank} | {cand.id} | {cand.kind} | "
        f"{cand.occurrences}x / {cand.sessions} sessions "
        f"| {fmt_tokens(cand.est_weekly_savings)}/wk | "
        f"{action_cell(cand, handled)} |"
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
    kept = [
        entry for entry in compiled if entry.get("verdict") in ("keep-llm", "dropped")
    ]
    if not kept:
        return "## Reviewed — keeping LLM\n\nnone reviewed yet\n"
    lines = [
        f"- {entry.get('candidate_id')} — "
        f"reviewed {entry.get('compiled_at')} — {verdict_label(entry)}"
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
    pending = [
        cid
        for cid in significant_ids(result.candidates)
        if cid not in handled_ids(compiled)
    ]
    return write_state(home, pending, now)
