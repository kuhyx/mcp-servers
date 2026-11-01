"""Tests for the accumulator's user-line handling and span flushing."""

from __future__ import annotations

from session_autopsy.accumulator import Accumulator
from session_autopsy.parse_helpers import top as top_entries
from session_autopsy.tests.conftest import (
    assistant_line,
    bash_block,
    feed,
    prompt_line,
    tool_result_line,
)


def test_user_line_filters() -> None:
    """Sidechain, non-external, and non-string content never count."""
    acc = Accumulator()
    feed(
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
    acc = Accumulator()
    feed(
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
    acc = Accumulator()
    long_tail = (
        "\n".join(["noise"] * 20)
        + "\nrm: cannot remove '/home/kuhy/.cache/x': Permission denied"
    )
    feed(
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
    acc = Accumulator()
    feed(acc, prompt_line("<command-name>/finish</command-name>"))
    acc.finish()
    acc.finish()
    assert acc.skill_invocations == []


def test_top_truncation_and_ordering() -> None:
    """bash_first_lines keeps the most frequent entries, ties alphabetical."""
    acc = Accumulator()
    for command, count in [("ls", 3), ("git status", 3), ("pwd", 1)]:
        for _ in range(count):
            feed(acc, assistant_line(bash_block(command)))
    top = top_entries(acc.bash_first_lines, 2)
    assert list(top) == ["git status", "ls"]
