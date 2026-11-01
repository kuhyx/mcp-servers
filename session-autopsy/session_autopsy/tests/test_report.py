"""Tests for REPORT.md rendering and its sections."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from session_autopsy import report, report_state, report_turns
from session_autopsy.candidates import Candidate
from session_autopsy.detectors import analyze
from session_autopsy.records import (
    ActivityCounts,
    Observations,
    TokenTotals,
    TurnEfficiency,
)
from session_autopsy.tests.conftest import (
    compiled_entry,
    invocation,
    record,
    session_meta,
)

if TYPE_CHECKING:
    from pathlib import Path

NOW = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)


def _fake_candidate(index: int) -> Candidate:
    """A minimal ranked candidate for table-rendering tests."""
    return Candidate(
        id=f"err-{index:08d}",
        kind="err",
        title=f"error {index}",
        sig=f"sig {index}",
        sessions=1,
        occurrences=2,
        avg_tokens=10,
        per_week=0.1,
        est_weekly_savings=100 - index,
        score=1.0,
        action="fix once",
        session_ids=["s1"],
    )


def test_candidate_table_empty_and_overflow() -> None:
    """Empty corpus placeholder, overflow note, and handled-action swap."""
    assert "none yet" in report._candidate_table([], set())
    table = report._candidate_table(
        [_fake_candidate(index) for index in range(45)], {"err-00000001"}
    )
    assert "plus 5 below the fold" in table
    assert "already handled — see Reviewed/scoreboard" in table


def test_reviewed_keep_llm_section() -> None:
    """Empty and populated reviewed-keep-LLM sections."""
    assert "none reviewed yet" in report._reviewed_keep_llm([])
    section = report._reviewed_keep_llm(
        [
            compiled_entry(verdict="keep-llm", candidate_id="skill-x"),
            compiled_entry(verdict="dropped", candidate_id="skill-y"),
        ]
    )
    assert "skill-x" in section
    assert "skill-y — reviewed" in section
    assert "dropped (workflow removed)" in section


def test_render_and_write_report(tmp_path: Path) -> None:
    """Full render includes all sections; write_report persists + counts."""
    records = [
        record(
            "a",
            obs=Observations(
                skill_invocations=[invocation(output=500_000)] * 3,
                error_signatures={"boom: x": 12},
            ),
        ),
    ]
    result = analyze(records, NOW)
    compiled = [
        compiled_entry(),
        compiled_entry(
            verdict="keep-llm", candidate_id="skill-explain-diff", script=None
        ),
    ]
    text = report.render_report(records, result, compiled, NOW)
    assert "# Session Autopsy Report" in text
    assert "skill-finish" in text
    assert "Reviewed — keeping LLM" in text
    assert "Context-bloat trend" in text
    assert "Compiled scoreboard" in text
    unreviewed = report.write_report(tmp_path, records, result, NOW)
    assert unreviewed == 1
    assert len(result.candidates) > 1
    assert (tmp_path / report.REPORT_FILE).is_file()
    (tmp_path / report_state.COMPILED_FILE).write_text(
        json.dumps([compiled_entry()]), encoding="utf-8"
    )
    assert report.write_report(tmp_path, records, result, NOW) == 0
    assert "already handled" in (tmp_path / report.REPORT_FILE).read_text(
        encoding="utf-8"
    )


def test_trend_and_header_empty_states() -> None:
    """Undated corpus → trend placeholder; zero turns → zero per-turn."""
    records = [
        record(
            "a",
            meta=session_meta(started_at=None),
            counts=ActivityCounts(assistant_msgs=0),
        )
    ]
    result = analyze(records, NOW)
    assert "no dated sessions yet" in report._trend_section(result)
    header = report._header(records, NOW)
    assert "avg 0/turn" in header


def test_state_file_is_valid_json(tmp_path: Path) -> None:
    """state.json holds the ranked ids and the timestamp."""
    report_state.write_state(tmp_path, ["x"], NOW)
    payload = json.loads(
        (tmp_path / report_state.STATE_FILE).read_text(encoding="utf-8")
    )
    assert payload["candidate_ids"] == ["x"]
    assert payload["last_generated"].startswith("2026-07-24")


def test_turn_efficiency_section_prices_mechanical_turns() -> None:
    """The point of the section: put a token number on turns nobody needed."""
    rec = record("s1", tokens=TokenTotals(cache_read=100_000_000))
    rec.turns = TurnEfficiency(
        api_turns=100,
        tool_calls=120,
        batched_turns=12,
        poll_turns=20,
        lint_test_turns=10,
        vcs_check_turns=5,
    )
    text = report_turns.turn_efficiency([rec])
    assert "100 tool turns carrying 120 calls" in text
    assert "batched: 12 (12%)" in text
    assert "**35** (35%)" in text  # 20 + 10 + 5 mechanical turns


def test_turn_efficiency_section_without_tool_turns() -> None:
    assert "no tool turns recorded yet" in report_turns.turn_efficiency([record("s1")])
