"""Tests for session parsing and turn-efficiency accounting."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from session_autopsy.accumulator import Accumulator
from session_autopsy.parse import parse_session, subagent_files
from session_autopsy.tests.conftest import (
    assistant_line,
    bash_block,
    feed,
    text_block,
    write_transcript,
)

if TYPE_CHECKING:
    from pathlib import Path


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


def test_turn_efficiency_groups_by_request_id() -> None:
    """One API response can span several transcript lines.

    Counting per line would report every turn as single-tool and hide the real
    batching rate, so batching is grouped by requestId.
    """
    acc = Accumulator()
    first = assistant_line(bash_block("wc -l < /tmp/scrape.log"))
    first["requestId"] = "req-1"
    second = assistant_line(bash_block("tail -3 /tmp/scrape.log"))
    second["requestId"] = "req-1"  # same API response, two tool calls
    third = assistant_line(bash_block("pytest -q"))
    third["requestId"] = "req-2"
    feed(acc, first, second, third)
    acc.finish()

    assert acc.turns.api_turns == 2
    assert acc.turns.tool_calls == 3
    assert acc.turns.batched_turns == 1  # req-1 carried two calls
    assert acc.turns.batching_rate == 0.5
    assert acc.turns.poll_turns == 1  # req-1 was pure log watching
    assert acc.turns.lint_test_turns == 1  # req-2 was the test gate
    assert acc.turns.mechanical_turns == 2


def test_turn_efficiency_ignores_turns_without_tools() -> None:
    acc = Accumulator()
    prose = assistant_line({"type": "text", "text": "just talking"})
    prose["requestId"] = "req-9"
    feed(acc, prose)
    acc.finish()
    assert acc.turns.api_turns == 0
    assert acc.turns.batching_rate == 0.0
    assert acc.turns.mechanical_rate == 0.0


def test_turn_efficiency_counts_vcs_checks_and_real_work_apart() -> None:
    acc = Accumulator()
    check = assistant_line(bash_block("git ls-remote origin refs/heads/main"))
    check["requestId"] = "req-a"
    work = assistant_line(bash_block("./run.sh --profile kcd index"))
    work["requestId"] = "req-b"
    feed(acc, check, work)
    acc.finish()
    assert acc.turns.vcs_check_turns == 1
    assert acc.turns.mechanical_turns == 1  # the index build is real work
    assert acc.turns.api_turns == 2
