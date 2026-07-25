"""Tests for the streaming transcript parser."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from python_pkg.session_autopsy import parse
from python_pkg.session_autopsy.parse import (
    _Accumulator,
    parse_session,
    subagent_files,
)
from python_pkg.session_autopsy.tests.conftest import (
    assistant_line,
    bash_block,
    prompt_line,
    skill_block,
    text_block,
    tool_block,
    tool_result_line,
    write_transcript,
)

if TYPE_CHECKING:
    from pathlib import Path


def _feed(acc: _Accumulator, *lines: dict[str, object]) -> None:
    """Feed dict lines through the accumulator as JSON."""
    for line in lines:
        acc.feed_line(json.dumps(line))


def test_feed_line_malformed() -> None:
    """Blank, invalid-JSON, and non-dict lines are tolerated and counted."""
    acc = _Accumulator()
    acc.feed_line("")
    acc.feed_line("   ")
    acc.feed_line("{not json")
    acc.feed_line("[1, 2]")
    assert acc.counts.malformed_lines == 2


def test_dispatch_types() -> None:
    """Titles, compact boundaries, and unknown types route correctly."""
    acc = _Accumulator()
    _feed(
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
    acc = _Accumulator()
    first = assistant_line(text_block("hello"), timestamp="2026-07-01T00:00:00Z")
    first["cwd"] = "/home/kuhy"
    first["gitBranch"] = "main"
    first["slug"] = "slug-one"
    second = assistant_line(timestamp="2026-07-02T00:00:00Z", with_usage=False)
    second["cwd"] = "/elsewhere"
    second["timestamp"] = "2026-07-02T00:00:00Z"
    _feed(acc, first, second, {"type": "assistant", "message": "not-a-dict"})
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
    acc = _Accumulator()
    _feed(
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
    acc = _Accumulator()
    _feed(
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
    acc = _Accumulator()
    _feed(
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
    acc = _Accumulator()
    _feed(acc, assistant_line(skill_block("finish")))
    _feed(
        acc,
        {
            "type": "assistant",
            "message": {"content": [], "usage": {"output_tokens": True}},
        },
    )
    for _ in range(parse.MAX_SKILL_BASH_SIG + 5):
        _feed(acc, assistant_line(bash_block("git status")))
    acc.finish()
    span = acc.skill_invocations[0]
    assert len(span.bash_sig) == parse.MAX_SKILL_BASH_SIG
    assert span.tokens_output == 10 * (parse.MAX_SKILL_BASH_SIG + 6)


def test_bash_runs_and_caps() -> None:
    """Contiguous Bash runs are kept at >=3, broken by other tools, capped."""
    acc = _Accumulator()
    _feed(acc, assistant_line(bash_block("git status"), bash_block("pytest -q")))
    _feed(acc, assistant_line(tool_block("Read", file_path="/x")))
    for _ in range(parse.MAX_BASH_RUN + 5):
        _feed(acc, assistant_line(bash_block("ls")))
    acc.finish()
    assert acc.bash_sequences == [["ls"] * parse.MAX_BASH_RUN]
    assert acc.tool_histogram["Bash"] == 2 + parse.MAX_BASH_RUN + 5
    assert acc.tool_histogram["Read"] == 1


def test_bash_edge_inputs() -> None:
    """Non-string and empty commands are ignored; unnamed tools count as ?."""
    acc = _Accumulator()
    _feed(
        acc,
        assistant_line(tool_block("Bash", command=5)),
        assistant_line(bash_block("   ")),
        assistant_line({"type": "tool_use", "name": None, "input": {}}),
    )
    acc.finish()
    assert not acc.bash_first_lines
    assert acc.tool_histogram["?"] == 1


def test_user_line_filters() -> None:
    """Sidechain, non-external, and non-string content never count."""
    acc = _Accumulator()
    _feed(
        acc,
        {"type": "user", "message": "not-a-dict"},
        prompt_line("real prompt here"),
        prompt_line("agent prompt", user_type="agent"),
        prompt_line("sidechain prompt", sidechain=True),
        {
            "type": "user",
            "userType": "external",
            "message": {"content": [{"type": "tool_result"}]},
        },
    )
    assert acc.counts.external_prompts == 1
    assert list(acc.typed_prompt_signatures) == ["real prompt here"]


def test_compact_retry_and_skipped_prompts() -> None:
    """Compact continuations, retries, and pasted/wrapped prompts."""
    acc = _Accumulator()
    _feed(
        acc,
        prompt_line("This session is being continued from a previous conversation."),
        prompt_line("limit got reset, continue"),
        prompt_line("continue where you left off, the limit has reset ok" + "x" * 40),
        prompt_line("[Image: original 1080x2460]"),
        prompt_line("<local-command-caveat>stuff</local-command-caveat>"),
        prompt_line("   "),
    )
    assert acc.counts.compaction_count == 1
    assert acc.counts.retry_prompt_count == 1
    assert acc.counts.external_prompts == 5
    assert "limit got reset, continue" in acc.typed_prompt_signatures
    assert len(acc.typed_prompt_signatures) == 2


def test_error_signatures() -> None:
    """Error lines are mined from str and dict results; generics dropped."""
    acc = _Accumulator()
    long_tail = (
        "\n".join(["noise"] * 20)
        + "\nrm: cannot remove '/home/kuhy/.cache/x': Permission denied"
    )
    _feed(
        acc,
        tool_result_line("Error: Exit code 5\nfatal: not a git repository"),
        tool_result_line(
            {
                "stderr": "Traceback (most recent call last):\nValueError: boom",
                "stdout": "ok",
                "junk": 3,
            }
        ),
        tool_result_line({"stdout": long_tail}),
        tool_result_line(None),
    )
    sigs = acc.error_signatures
    assert sigs["fatal: not a git repository"] == 1
    assert sigs["ValueError: boom"] == 1
    assert sigs["rm: cannot remove '<PATH>': Permission denied"] == 1
    assert "Error: Exit code <N>" not in sigs
    assert "Traceback (most recent call last):" not in sigs


def test_flush_skill_drops_empty_spans() -> None:
    """A span with zero turns (e.g. a bare slash command) is dropped."""
    acc = _Accumulator()
    _feed(acc, prompt_line("<command-name>/finish</command-name>"))
    acc.finish()
    acc.finish()
    assert acc.skill_invocations == []


def test_top_truncation_and_ordering() -> None:
    """bash_first_lines keeps the most frequent entries, ties alphabetical."""
    acc = _Accumulator()
    for command, count in [("ls", 3), ("git status", 3), ("pwd", 1)]:
        for _ in range(count):
            _feed(acc, assistant_line(bash_block(command)))
    top = parse._top(acc.bash_first_lines, 2)
    assert list(top) == ["git status", "ls"]


def test_parse_session_with_subagents(tmp_path: Path) -> None:
    """Subagent transcripts fold into the parent record."""
    projects = tmp_path / "projects"
    main = write_transcript(
        projects,
        "sess-1",
        [assistant_line(text_block("hi")), "not json"],
    )
    sub_dir = main.parent / "sess-1" / "subagents"
    sub_dir.mkdir(parents=True)
    (sub_dir / "agent-a.jsonl").write_text(
        json.dumps(assistant_line(bash_block("git status"))) + "\n",
        encoding="utf-8",
    )
    rec = parse_session(main)
    assert rec.session_id == "sess-1"
    assert rec.counts.subagent_count == 1
    assert rec.counts.assistant_msgs == 2
    assert rec.counts.malformed_lines == 1
    assert rec.obs.tool_histogram == {"Bash": 1}


def test_parse_session_without_subagents(tmp_path: Path) -> None:
    """A lone transcript parses with zero subagents."""
    projects = tmp_path / "projects"
    main = write_transcript(projects, "sess-2", [assistant_line(text_block("hi"))])
    rec = parse_session(main)
    assert rec.counts.subagent_count == 0
    assert subagent_files(main) == []
