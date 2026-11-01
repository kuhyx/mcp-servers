"""The turn-efficiency block of REPORT.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

from session_autopsy.report_measure import fmt_tokens

if TYPE_CHECKING:
    from session_autopsy.records import SessionRecord


def turn_efficiency(records: list[SessionRecord]) -> str:
    """Render the turn-efficiency block.

    Cache read is charged per API round-trip on the whole conversation so far, so
    turns the model did not need to take are the single largest recoverable cost —
    far larger than message length. This block prices them.

    Args:
        records: All session records.

    Returns:
        The turn-efficiency section.
    """
    api = sum(r.turns.api_turns for r in records)
    if not api:
        return "## Turn efficiency\n\nno tool turns recorded yet\n"
    batched = sum(r.turns.batched_turns for r in records)
    poll = sum(r.turns.poll_turns for r in records)
    lint = sum(r.turns.lint_test_turns for r in records)
    vcs = sum(r.turns.vcs_check_turns for r in records)
    calls = sum(r.turns.tool_calls for r in records)
    cache_read = sum(r.tokens.cache_read for r in records)
    mechanical = poll + lint + vcs
    per_turn = cache_read // api if api else 0
    return (
        "## Turn efficiency\n\n"
        f"{api} tool turns carrying {calls} calls | "
        f"batched: {batched} ({batched / api:.0%}) | "
        f"avg context re-read: {fmt_tokens(per_turn)}/turn\n\n"
        "| mechanical turn kind | turns | est. cache-read |\n"
        "| --- | ---: | ---: |\n"
        f"| polling (progress, liveness, clocks) | {poll} "
        f"| {fmt_tokens(poll * per_turn)} |\n"
        f"| lint / test (exit-code adjudicated) | {lint} "
        f"| {fmt_tokens(lint * per_turn)} |\n"
        f"| read-only VCS checks | {vcs} | {fmt_tokens(vcs * per_turn)} |\n"
        f"| **total** | **{mechanical}** ({mechanical / api:.0%}) | "
        f"**{fmt_tokens(mechanical * per_turn)}** |\n\n"
        "These turns decided nothing: a script could have taken them and reported "
        "once. Batching independent calls cuts the rest.\n"
    )
