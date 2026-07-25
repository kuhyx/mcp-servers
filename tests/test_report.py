"""Tests for report rendering, state.json, and the measure scoreboard."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import TYPE_CHECKING

from python_pkg.session_autopsy import report
from python_pkg.session_autopsy.detectors import Candidate, analyze
from python_pkg.session_autopsy.records import ActivityCounts, Observations
from python_pkg.session_autopsy.tests.conftest import (
    invocation,
    record,
    session_meta,
)

if TYPE_CHECKING:
    from pathlib import Path

NOW = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)


def test_fmt_tokens() -> None:
    """Plain, k, and M ranges."""
    assert report.fmt_tokens(842) == "842"
    assert report.fmt_tokens(74_500) == "74k"
    assert report.fmt_tokens(1_234_567) == "1.2M"


def test_load_state_variants(tmp_path: Path) -> None:
    """Missing, corrupt, and wrong-type state files read as empty."""
    assert report.load_state(tmp_path) == {}
    state_path = tmp_path / report.STATE_FILE
    state_path.write_text("{bad", encoding="utf-8")
    assert report.load_state(tmp_path) == {}
    state_path.write_text("[1]", encoding="utf-8")
    assert report.load_state(tmp_path) == {}
    state_path.write_text('{"unreviewed_count": 2}', encoding="utf-8")
    assert report.load_state(tmp_path) == {"unreviewed_count": 2}


def test_load_compiled_variants(tmp_path: Path) -> None:
    """Missing, corrupt, wrong-type, and mixed compiled files."""
    assert report.load_compiled(tmp_path) == []
    compiled_path = tmp_path / report.COMPILED_FILE
    compiled_path.write_text("{bad", encoding="utf-8")
    assert report.load_compiled(tmp_path) == []
    compiled_path.write_text('{"a": 1}', encoding="utf-8")
    assert report.load_compiled(tmp_path) == []
    compiled_path.write_text('[{"skill": "finish"}, 3]', encoding="utf-8")
    assert report.load_compiled(tmp_path) == [{"skill": "finish"}]


def test_write_state_lifecycle(tmp_path: Path) -> None:
    """Unreviewed counting, mark-reviewed, stale-id pruning, junk tolerance."""
    assert report.write_state(tmp_path, ["a", "b"], NOW) == 2
    assert report.write_state(tmp_path, ["a", "b"], NOW, mark_reviewed=True) == 0
    assert report.write_state(tmp_path, ["a", "b", "c"], NOW) == 1
    assert report.write_state(tmp_path, ["c"], NOW) == 1
    state_path = tmp_path / report.STATE_FILE
    state_path.write_text('{"reviewed_ids": "junk"}', encoding="utf-8")
    assert report.write_state(tmp_path, ["z"], NOW) == 1


def _compiled_entry(**over: object) -> dict[str, object]:
    """A compiled.json entry with sensible defaults."""
    entry: dict[str, object] = {
        "candidate_id": "skill-finish",
        "skill": "finish",
        "script": "/x/finish.sh",
        "verdict": "compiled",
        "compiled_at": "2026-07-10T00:00:00+00:00",
    }
    entry.update(over)
    return entry


def test_measure_lines_variants() -> None:
    """Placeholder, skips, before-only, delta, and after-only cases."""
    before = record(
        "old",
        meta=session_meta(started_at="2026-07-01T00:00:00Z"),
        obs=Observations(
            skill_invocations=[
                invocation(output=100000, cache_write=0),
                invocation(name="other"),
            ]
        ),
    )
    after = record(
        "new",
        meta=session_meta(started_at="2026-07-20T00:00:00Z"),
        obs=Observations(skill_invocations=[invocation(output=10000, cache_write=0)]),
    )
    assert report.measure_lines([], [], None) == [
        "nothing compiled yet — run /compile-candidate on a ranked candidate",
    ]
    skipped = [
        _compiled_entry(skill=""),
        _compiled_entry(script=None),
        _compiled_entry(skill="other"),
    ]
    assert len(report.measure_lines([before], skipped, "finish")) == 1
    only_before = report.measure_lines([before], [_compiled_entry()], None)
    assert "no post-compile invocations yet" in only_before[0]
    delta = report.measure_lines([before, after], [_compiled_entry()], None)
    assert "→ after" in delta[0]
    assert "-90%" in delta[0]
    after_only = report.measure_lines([after], [_compiled_entry()], None)
    assert "before 0/inv (n=0)" in after_only[0]
    assert "%" not in after_only[0]


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
    """Empty corpus placeholder and the beyond-top-40 overflow note."""
    assert "none yet" in report._candidate_table([])
    table = report._candidate_table([_fake_candidate(index) for index in range(45)])
    assert "plus 5 below the fold" in table


def test_reviewed_keep_llm_section() -> None:
    """Empty and populated reviewed-keep-LLM sections."""
    assert "none reviewed yet" in report._reviewed_keep_llm([])
    section = report._reviewed_keep_llm(
        [_compiled_entry(verdict="keep-llm", candidate_id="skill-x")]
    )
    assert "skill-x" in section


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
        _compiled_entry(),
        _compiled_entry(
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
    report.write_state(tmp_path, ["x"], NOW)
    payload = json.loads((tmp_path / report.STATE_FILE).read_text(encoding="utf-8"))
    assert payload["candidate_ids"] == ["x"]
    assert payload["last_generated"].startswith("2026-07-24")
