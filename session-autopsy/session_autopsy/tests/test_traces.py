"""Tests for compact trace extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from session_autopsy import traces
from session_autopsy.tests.conftest import (
    FileFacts,
    assistant_line,
    bash_block,
    prompt_line,
    record,
    skill_block,
    text_block,
    tool_block,
    tool_result_line,
)
from session_autopsy.tests.trace_helpers import (
    DENIED,
    trace_candidate,
    trace_session,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_render_traces_missing_and_cap(tmp_path: Path) -> None:
    """Unknown ids are skipped, capture stops at max, empty → message."""
    lines: list[dict[str, object] | str] = [
        assistant_line(skill_block("finish")),
        assistant_line(bash_block("x")),
    ]
    rec_a = trace_session(tmp_path, "a", lines)
    rec_b = trace_session(tmp_path, "b", lines)
    gone = record("gone", file=FileFacts(path=str(tmp_path / "nope.jsonl")))
    by_id = {"a": rec_a, "b": rec_b, "gone": gone}
    capped = traces.render_traces(
        trace_candidate("skill", "finish", ["missing", "a", "b"]), by_id, 1
    )
    assert "## session a" in capped
    assert "## session b" not in capped
    empty = traces.render_traces(trace_candidate("skill", "finish", ["gone"]), by_id, 5)
    assert "no matching spans found" in empty


def test_error_traces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Error hits pair the causing command with the line, capped."""
    monkeypatch.setattr(traces, "MAX_ERR_HITS_PER_SESSION", 1)
    lines: list[dict[str, object] | str] = [
        assistant_line(tool_block("Bash", command=5)),
        assistant_line(tool_block("Read", file_path="/x")),
        assistant_line(bash_block("rm -rf /home/kuhy/.cache/yay/pkg")),
        tool_result_line(f"unrelated noise line\n{DENIED}"),
        tool_result_line({"stderr": DENIED}),
    ]
    rec = trace_session(tmp_path, "err", lines)
    sig = "rm: cannot remove '<PATH>': Permission denied"
    text = traces.render_traces(trace_candidate("err", sig, ["err"]), {"err": rec}, 5)
    assert text.count("CMD | rm -rf") == 1
    assert text.count("ERR |") == 1


def test_ngram_traces(tmp_path: Path) -> None:
    """The first window matching the gram renders its raw commands."""
    lines: list[dict[str, object] | str] = [
        assistant_line(text_block("hi")),
        assistant_line(tool_block("Bash", command=7)),
        assistant_line(
            bash_block("git status"), bash_block("pytest -q"), bash_block("git push")
        ),
    ]
    rec = trace_session(tmp_path, "ng", lines)
    hit = traces.render_traces(
        trace_candidate("ngram", "git → pytest → git", ["ng"]), {"ng": rec}, 5
    )
    assert "CMD | git status" in hit
    assert "CMD | git push" in hit
    miss = traces.render_traces(
        trace_candidate("ngram", "adb → adb", ["ng"]), {"ng": rec}, 5
    )
    assert "no matching spans found" in miss


def test_prompt_traces(tmp_path: Path) -> None:
    """Matching prompts render with the tools that followed."""
    lines: list[dict[str, object] | str] = [
        prompt_line("fix the tests please"),
        assistant_line(
            text_block("narration"),
            bash_block("pytest -q"),
            tool_block("Read", file_path="/x"),
        ),
        prompt_line("unrelated message here"),
        assistant_line(bash_block("ls")),
        prompt_line("fix the tests please"),
    ]
    rec = trace_session(tmp_path, "pr", lines)
    text = traces.render_traces(
        trace_candidate("prompt", "fix the tests please", ["pr"]), {"pr": rec}, 5
    )
    assert text.count("PROMPT | fix the tests please") == 2
    assert "THEN | Bash" in text
    assert "THEN | Read" in text
    assert text.count("THEN |") == 2
