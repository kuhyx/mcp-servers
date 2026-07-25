"""Tests for the pure candidate detectors."""

from __future__ import annotations

from datetime import datetime, timezone

from python_pkg.session_autopsy import detectors, stats
from python_pkg.session_autopsy.detectors import analyze
from python_pkg.session_autopsy.records import (
    ActivityCounts,
    Observations,
    TokenTotals,
)
from python_pkg.session_autopsy.stats import parse_timestamp
from python_pkg.session_autopsy.tests.conftest import (
    invocation,
    record,
    session_meta,
)

NOW = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
OLD = "2026-01-01T00:00:00.000Z"


def test_parse_timestamp() -> None:
    """None, empty, junk, and Z-suffixed ISO forms."""
    assert parse_timestamp(None) is None
    assert parse_timestamp("") is None
    assert parse_timestamp("junk") is None
    stamp = parse_timestamp("2026-07-20T10:00:00.000Z")
    assert stamp is not None
    assert stamp.tzinfo is not None


def test_analyze_empty_corpus() -> None:
    """No records → no candidates, empty trend, zero per-turn average."""
    result = analyze([], NOW)
    assert result.candidates == []
    assert result.trend.weekly == []
    assert result.trend.slope_per_week == 0.0


def test_skill_candidates_threshold_and_score() -> None:
    """Skills need >=3 invocations; zero-turn spans don't skew determinism."""
    records = [
        record(
            "a",
            obs=Observations(
                skill_invocations=[invocation(), invocation(tool_only=0, text=0)]
            ),
        ),
        record("b", obs=Observations(skill_invocations=[invocation()])),
        record("c", obs=Observations(skill_invocations=[invocation(name="rare")])),
    ]
    result = analyze(records, NOW)
    skills = [cand for cand in result.candidates if cand.kind == "skill"]
    assert [cand.id for cand in skills] == ["skill-finish"]
    finish = skills[0]
    assert finish.occurrences == 3
    assert finish.sessions == 2
    assert finish.score == 0.6
    assert finish.avg_tokens == 150_000
    assert finish.per_week == 3 / 8
    assert finish.est_weekly_savings == int(150_000 * 3 / 8)
    assert "compile-candidate skill-finish" in finish.action


def test_skill_similarity_zero_without_sequences() -> None:
    """All-empty bash signatures give zero similarity evidence."""
    records = [
        record("a", obs=Observations(skill_invocations=[invocation(bash_sig=[])])),
        record("b", obs=Observations(skill_invocations=[invocation(bash_sig=[])])),
        record("c", obs=Observations(skill_invocations=[invocation(bash_sig=[])])),
    ]
    result = analyze(records, NOW)
    assert result.candidates[0].score == 0.0


def test_old_invocations_have_zero_weekly_rate() -> None:
    """Occurrences outside the trailing window contribute no weekly rate."""
    records = [
        record(
            "a",
            meta=session_meta(started_at=OLD),
            obs=Observations(skill_invocations=[invocation()]),
        ),
        record(
            "b",
            meta=session_meta(started_at=OLD),
            obs=Observations(skill_invocations=[invocation()]),
        ),
        record(
            "c",
            meta=session_meta(started_at=None),
            obs=Observations(skill_invocations=[invocation()]),
        ),
    ]
    result = analyze(records, NOW)
    assert result.candidates[0].per_week == 0.0
    assert result.candidates[0].est_weekly_savings == 0


def test_error_candidates_thresholds() -> None:
    """Errors qualify by total occurrences OR by session spread."""
    records = [
        record("a", obs=Observations(error_signatures={"boom: <PATH>": 10, "meh": 2})),
        record("b", obs=Observations(error_signatures={"spread: x": 1})),
        record("c", obs=Observations(error_signatures={"spread: x": 1})),
        record("d", obs=Observations(error_signatures={"spread: x": 1})),
    ]
    result = analyze(records, NOW)
    errs = {cand.sig: cand for cand in result.candidates if cand.kind == "err"}
    assert set(errs) == {"boom: <PATH>", "spread: x"}
    assert errs["boom: <PATH>"].occurrences == 10
    assert errs["spread: x"].sessions == 3


def test_prompt_candidates_filters() -> None:
    """Prompts need >=4 occurrences and >=3 words."""
    records = [
        record(
            "a",
            obs=Observations(
                typed_prompt_signatures={
                    "do the big thing": 4,
                    "hi there friend": 3,
                    "short one": 9,
                }
            ),
        ),
    ]
    result = analyze(records, NOW)
    prompts = [cand.sig for cand in result.candidates if cand.kind == "prompt"]
    assert prompts == ["do the big thing"]


def test_ngram_candidates_and_subgram_drop() -> None:
    """Frequent distinct sequences survive; subgrams and degenerates drop."""
    seq = ["a", "b", "c", "d"]
    degenerate = ["x", "x", "x", "x", "x", "x"]
    records = [
        record(sid, obs=Observations(bash_sequences=[seq, seq, degenerate]))
        for sid in ("s1", "s2", "s3", "s4")
    ]
    result = analyze(records, NOW)
    ngrams = [cand for cand in result.candidates if cand.kind == "ngram"]
    assert [cand.sig for cand in ngrams] == ["a → b → c → d"]
    assert ngrams[0].occurrences == 8
    assert ngrams[0].sessions == 4


def test_ngram_below_session_threshold() -> None:
    """Many occurrences in too few sessions do not qualify."""
    records = [record("only", obs=Observations(bash_sequences=[["a", "b", "c"]] * 20))]
    result = analyze(records, NOW)
    assert [cand for cand in result.candidates if cand.kind == "ngram"] == []


def test_contains_and_drop_subgrams() -> None:
    """Containment handles equal-length and absent cases."""
    assert detectors._contains(("a", "b", "c"), ("b", "c"))
    assert not detectors._contains(("a", "b"), ("a", "b"))
    assert not detectors._contains(("a", "b", "c"), ("c", "d"))
    kept = detectors._drop_subgrams({("a", "b", "c"): 9, ("a", "b"): 9, ("z", "q"): 9})
    assert kept == {("a", "b", "c"), ("z", "q")}


def test_trend_weekly_slope_and_offenders() -> None:
    """Weekly averages, slope, and cache-heaviest sessions."""
    records = [
        record(
            "w1",
            tokens=TokenTotals(cache_read=1000000),
            meta=session_meta(started_at="2026-07-06T10:00:00Z"),
            counts=ActivityCounts(assistant_msgs=10),
        ),
        record(
            "w2",
            tokens=TokenTotals(cache_read=2000000),
            meta=session_meta(started_at="2026-07-13T10:00:00Z"),
            counts=ActivityCounts(assistant_msgs=10),
        ),
        record(
            "w3",
            tokens=TokenTotals(cache_read=3000000),
            meta=session_meta(started_at="2026-07-20T10:00:00Z", slug="heavy-slug"),
            counts=ActivityCounts(assistant_msgs=10),
        ),
        record(
            "w4",
            tokens=TokenTotals(cache_read=100),
            meta=session_meta(started_at="2026-07-21T10:00:00Z"),
            counts=ActivityCounts(assistant_msgs=10),
        ),
        record("undated", meta=session_meta(started_at=None)),
        record("silent", counts=ActivityCounts(assistant_msgs=0)),
    ]
    result = analyze(records, NOW)
    trend = result.trend
    assert len(trend.weekly) == 3
    assert trend.slope_per_week > 0
    assert trend.top_sessions[0] == ("heavy-slug", 300_000)
    assert len(trend.top_sessions) == 3


def test_slope_needs_two_points() -> None:
    """A single weekly sample yields slope 0."""
    assert stats.slope([5.0]) == 0.0


def test_candidate_to_dict() -> None:
    """Serialization exposes rounded score and per_week."""
    records = [record("a", obs=Observations(skill_invocations=[invocation()] * 3))]
    cand = analyze(records, NOW).candidates[0]
    data = cand.to_dict()
    assert data["id"] == "skill-finish"
    assert isinstance(data["per_week"], float)
    assert isinstance(data["session_ids"], list)


def test_recent_session_ids_cap_and_order() -> None:
    """Evidence session ids are newest-first, capped at 10, deduplicated."""
    many = [
        record(
            f"s{index:02d}",
            meta=session_meta(started_at=f"2026-07-{index + 1:02d}T00:00:00Z"),
            obs=Observations(skill_invocations=[invocation()]),
        )
        for index in range(12)
    ]
    many.append(
        record(
            "s00",
            meta=session_meta(started_at="2026-07-01T00:00:00Z"),
            obs=Observations(skill_invocations=[invocation()]),
        )
    )
    cand = analyze(many, NOW).candidates[0]
    assert len(cand.session_ids) == 10
    assert cand.session_ids[0] == "s11"
