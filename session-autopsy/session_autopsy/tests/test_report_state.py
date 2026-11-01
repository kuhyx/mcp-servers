"""Tests for state.json handling, token formatting, and the measure scoreboard."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from session_autopsy import report_measure, report_state
from session_autopsy.records import (
    Observations,
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


def test_fmt_tokens() -> None:
    """Plain, k, and M ranges."""
    assert report_measure.fmt_tokens(842) == "842"
    assert report_measure.fmt_tokens(74_500) == "74k"
    assert report_measure.fmt_tokens(1_234_567) == "1.2M"


def test_load_state_variants(tmp_path: Path) -> None:
    """Missing, corrupt, and wrong-type state files read as empty."""
    assert report_state.load_state(tmp_path) == {}
    state_path = tmp_path / report_state.STATE_FILE
    state_path.write_text("{bad", encoding="utf-8")
    assert report_state.load_state(tmp_path) == {}
    state_path.write_text("[1]", encoding="utf-8")
    assert report_state.load_state(tmp_path) == {}
    state_path.write_text('{"unreviewed_count": 2}', encoding="utf-8")
    assert report_state.load_state(tmp_path) == {"unreviewed_count": 2}


def test_load_compiled_variants(tmp_path: Path) -> None:
    """Missing, corrupt, wrong-type, and mixed compiled files."""
    assert report_state.load_compiled(tmp_path) == []
    compiled_path = tmp_path / report_state.COMPILED_FILE
    compiled_path.write_text("{bad", encoding="utf-8")
    assert report_state.load_compiled(tmp_path) == []
    compiled_path.write_text('{"a": 1}', encoding="utf-8")
    assert report_state.load_compiled(tmp_path) == []
    compiled_path.write_text('[{"skill": "finish"}, 3]', encoding="utf-8")
    assert report_state.load_compiled(tmp_path) == [{"skill": "finish"}]


def test_write_state_lifecycle(tmp_path: Path) -> None:
    """Unreviewed counting, mark-reviewed, stale-id pruning, junk tolerance."""
    assert report_state.write_state(tmp_path, ["a", "b"], NOW) == 2
    assert report_state.write_state(tmp_path, ["a", "b"], NOW, mark_reviewed=True) == 0
    assert report_state.write_state(tmp_path, ["a", "b", "c"], NOW) == 1
    assert report_state.write_state(tmp_path, ["c"], NOW) == 1
    state_path = tmp_path / report_state.STATE_FILE
    state_path.write_text('{"reviewed_ids": "junk"}', encoding="utf-8")
    assert report_state.write_state(tmp_path, ["z"], NOW) == 1


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
    assert report_measure.measure_lines([], [], None) == [
        "nothing compiled yet — run /compile-candidate on a ranked candidate",
    ]
    skipped = [
        compiled_entry(skill=""),
        compiled_entry(script=None),
        compiled_entry(skill="other"),
    ]
    assert len(report_measure.measure_lines([before], skipped, "finish")) == 1
    only_before = report_measure.measure_lines([before], [compiled_entry()], None)
    assert "no post-compile invocations yet" in only_before[0]
    delta = report_measure.measure_lines([before, after], [compiled_entry()], None)
    assert "→ after" in delta[0]
    assert "-90%" in delta[0]
    after_only = report_measure.measure_lines([after], [compiled_entry()], None)
    assert "before 0/inv (n=0)" in after_only[0]
    assert "%" not in after_only[0]
