"""Tests for the transcript accumulator: assistant lines, tools, spans."""

from __future__ import annotations

from session_autopsy import parse_constants
from session_autopsy.accumulator import Accumulator
from session_autopsy.tests.conftest import (
    assistant_line,
    bash_block,
    feed,
    prompt_line,
    skill_block,
    text_block,
    tool_block,
)


def test_feed_line_malformed() -> None:
    """Blank, invalid-JSON, and non-dict lines are tolerated and counted."""
    acc = Accumulator()
    acc.feed_line("")
    acc.feed_line("   ")
    acc.feed_line("{not json")
    acc.feed_line("[1, 2]")
    assert acc.counts.malformed_lines == 2


def test_dispatch_types() -> None:
    """Titles, compact boundaries, and unknown types route correctly."""
    acc = Accumulator()
    feed(
        acc,
        {"type": "ai-title", "title": "My Session"},
        {"type": "ai-title", "aiTitle": "Renamed", "title": ""},
        {"type": "ai-title"},
        {"type": "system", "subtype": "compact_boundary"},
        {"type": "system", "subtype": "other"},
        {"type": "attachment"},
    )
    assert acc.meta.title == "Renamed"
    assert acc.counts.compaction_count == 1


def test_assistant_usage_and_metadata() -> None:
    """Usage sums; first/last timestamps and first cwd/branch/slug stick."""
    acc = Accumulator()
    first = assistant_line(text_block("hello"), timestamp="2026-07-01T00:00:00Z")
    first["cwd"] = "/home/kuhy"
    first["gitBranch"] = "main"
    first["slug"] = "slug-one"
    second = assistant_line(timestamp="2026-07-02T00:00:00Z", with_usage=False)
    second["cwd"] = "/elsewhere"
    second["timestamp"] = "2026-07-02T00:00:00Z"
    feed(acc, first, second, {"type": "assistant", "message": "not-a-dict"})
    assert acc.counts.assistant_msgs == 2
    assert acc.tokens.output == 10
    assert acc.meta.started_at == "2026-07-01T00:00:00Z"
    assert acc.meta.ended_at == "2026-07-02T00:00:00Z"
    assert (acc.meta.cwd, acc.meta.git_branch, acc.meta.slug) == (
        "/home/kuhy",
        "main",
        "slug-one",
    )


def test_scan_blocks_edge_shapes() -> None:
    """Non-list content, non-dict blocks, and non-str text are tolerated."""
    acc = Accumulator()
    feed(
        acc,
        {"type": "assistant", "message": {"content": "not-a-list", "usage": None}},
        {
            "type": "assistant",
            "message": {
                "content": ["junk", {"type": "text", "text": 5}, {"type": "other"}]
            },
        },
    )
    assert acc.counts.assistant_msgs == 2
    assert acc.tokens.output == 0


def test_skill_span_via_tool_and_slash() -> None:
    """Skill spans open via Skill tool_use AND via slash-command prompts."""
    acc = Accumulator()
    feed(
        acc,
        assistant_line(skill_block("finish")),
        assistant_line(bash_block("git status")),
        assistant_line(text_block("x" * 500)),
        prompt_line("<command-name>/phone-deploy</command-name> args"),
        assistant_line(bash_block("adb devices")),
        prompt_line("<command-name>/model</command-name>"),
        prompt_line("looks good, thanks"),
    )
    acc.finish()
    names = [inv.name for inv in acc.skill_invocations]
    assert names == ["finish", "phone-deploy"]
    finish_inv = acc.skill_invocations[0]
    assert finish_inv.tool_only_turns == 2
    assert finish_inv.text_turns == 1
    assert finish_inv.bash_sig == ["git"]
    assert acc.counts.external_prompts == 1
    assert acc.typed_prompt_signatures == {"looks good, thanks": 1}


def test_generated_chars_counts_write_inputs() -> None:
    """Big Write/Edit inputs make a turn text-bearing despite tool_use."""
    acc = Accumulator()
    feed(
        acc,
        assistant_line(skill_block("explain-diff")),
        assistant_line(tool_block("Write", content="y" * 1000)),
        assistant_line(tool_block("Edit", new_string="z" * 1000)),
        assistant_line(tool_block("Write", junk=3)),
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Write", "input": "not-a-dict"}
                ]
            },
        },
    )
    acc.finish()
    span = acc.skill_invocations[0]
    assert span.text_turns == 2
    assert span.tool_only_turns == 3


def test_skill_bash_sig_cap() -> None:
    """A skill span's bash signature list stops at the cap."""
    acc = Accumulator()
    feed(acc, assistant_line(skill_block("finish")))
    feed(
        acc,
        {
            "type": "assistant",
            "message": {"content": [], "usage": {"output_tokens": True}},
        },
    )
    for _ in range(parse_constants.MAX_SKILL_BASH_SIG + 5):
        feed(acc, assistant_line(bash_block("git status")))
    acc.finish()
    span = acc.skill_invocations[0]
    assert len(span.bash_sig) == parse_constants.MAX_SKILL_BASH_SIG
    assert span.tokens_output == 10 * (parse_constants.MAX_SKILL_BASH_SIG + 6)


def test_bash_runs_and_caps() -> None:
    """Contiguous Bash runs are kept at >=3, broken by other tools, capped."""
    acc = Accumulator()
    feed(acc, assistant_line(bash_block("git status"), bash_block("pytest -q")))
    feed(acc, assistant_line(tool_block("Read", file_path="/x")))
    for _ in range(parse_constants.MAX_BASH_RUN + 5):
        feed(acc, assistant_line(bash_block("ls")))
    acc.finish()
    assert acc.bash_sequences == [["ls"] * parse_constants.MAX_BASH_RUN]
    assert acc.tool_histogram["Bash"] == 2 + parse_constants.MAX_BASH_RUN + 5
    assert acc.tool_histogram["Read"] == 1


def test_bash_edge_inputs() -> None:
    """Non-string and empty commands are ignored; unnamed tools count as ?."""
    acc = Accumulator()
    feed(
        acc,
        assistant_line(tool_block("Bash", command=5)),
        assistant_line(bash_block("   ")),
        assistant_line({"type": "tool_use", "name": None, "input": {}}),
    )
    acc.finish()
    assert not acc.bash_first_lines
    assert acc.tool_histogram["?"] == 1
