"""Tests for the record schema and its defensive JSON coercions."""

from __future__ import annotations

from python_pkg.session_autopsy.records import (
    ActivityCounts,
    Observations,
    SessionMeta,
    SessionRecord,
    SkillInvocation,
    TokenTotals,
    _as_float,
    _as_int,
    _dict_get,
    _list_of_dicts,
    _list_value,
    _opt_str,
    _str_int_map,
    _str_list,
    record_from_dict,
)


def test_as_int_coercions() -> None:
    """bools and junk yield 0; numerics coerce."""
    flag: object = True
    assert _as_int(flag) == 0
    assert _as_int(5) == 5
    assert _as_int(5.9) == 5
    assert _as_int("x") == 0
    assert _as_int(None) == 0


def test_as_float_coercions() -> None:
    """bools and junk yield 0.0; numerics coerce."""
    flag: object = True
    assert _as_float(flag) == 0.0
    assert _as_float(2) == 2.0
    assert _as_float(2.5) == 2.5
    assert _as_float("x") == 0.0


def test_opt_str() -> None:
    """Only non-empty strings survive."""
    assert _opt_str("a") == "a"
    assert _opt_str("") is None
    assert _opt_str(3) is None


def test_dict_get() -> None:
    """Non-dict values index to None."""
    assert _dict_get({"k": 1}, "k") == 1
    assert _dict_get([1], "k") is None


def test_list_helpers() -> None:
    """List filters tolerate wrong types and mixed elements."""
    assert _list_value([1]) == [1]
    assert _list_value("no") == []
    assert _list_of_dicts([{"a": 1}, 2, "x"]) == [{"a": 1}]
    assert _str_list(["a", 1, "b"]) == ["a", "b"]


def test_str_int_map() -> None:
    """Counter maps coerce keys to str and values to int."""
    assert _str_int_map({"a": 2, 3: "junk"}) == {"a": 2, "3": 0}
    assert _str_int_map("no") == {}


def test_token_totals_add_usage() -> None:
    """Usage fields accumulate; missing fields count as 0."""
    totals = TokenTotals()
    totals.add_usage({"output_tokens": 5, "cache_read_input_tokens": 7})
    totals.add_usage(
        {"output_tokens": 2, "input_tokens": 1, "cache_creation_input_tokens": 4}
    )
    assert (totals.output, totals.input, totals.cache_read, totals.cache_write) == (
        7,
        1,
        7,
        4,
    )


def test_round_trip() -> None:
    """to_dict → record_from_dict preserves every field."""
    original = SessionRecord(
        session_id="s",
        project_slug="p",
        transcript_path="/t",
        file_size=9,
        file_mtime=1.5,
        analyzed_at="2026-01-01T00:00:00+00:00",
        meta=SessionMeta(
            started_at="a",
            ended_at="b",
            cwd="/c",
            git_branch="main",
            slug="sl",
            title="ti",
        ),
        counts=ActivityCounts(
            assistant_msgs=3,
            external_prompts=2,
            subagent_count=1,
            compaction_count=1,
            retry_prompt_count=2,
            malformed_lines=3,
        ),
        tokens=TokenTotals(output=1, input=2, cache_read=3, cache_write=4),
        obs=Observations(
            tool_histogram={"Bash": 2},
            bash_first_lines={"git status": 1},
            bash_sequences=[["git", "pytest", "git"]],
            skill_invocations=[
                SkillInvocation(
                    name="finish",
                    tool_only_turns=1,
                    text_turns=2,
                    tokens_output=3,
                    tokens_cache_write=4,
                    bash_sig=["git"],
                )
            ],
            error_signatures={"rm: x": 2},
            typed_prompt_signatures={"do the thing": 1},
        ),
    )
    assert record_from_dict(original.to_dict()) == original


def test_from_dict_tolerates_junk() -> None:
    """Missing keys, wrong types, and junk entries never raise."""
    rebuilt = record_from_dict(
        {
            "tokens": "junk",
            "counts": "junk",
            "obs": {
                "skill_invocations": [{"name": None}, "junk"],
                "bash_sequences": ["junk", ["a", 1]],
            },
            "meta": {"started_at": 5},
        },
    )
    assert rebuilt.session_id == ""
    assert rebuilt.tokens == TokenTotals()
    assert rebuilt.counts == ActivityCounts()
    assert rebuilt.obs.skill_invocations[0].name == "?"
    assert rebuilt.obs.bash_sequences == [[], ["a"]]
    assert rebuilt.meta.started_at is None
