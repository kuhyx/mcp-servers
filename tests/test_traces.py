"""Tests for compact trace extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from python_pkg.session_autopsy import traces
from python_pkg.session_autopsy.detectors import Candidate
from python_pkg.session_autopsy.tests.conftest import (
    FileFacts,
    assistant_line,
    bash_block,
    prompt_line,
    record,
    skill_block,
    text_block,
    tool_block,
    tool_result_line,
    write_transcript,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from python_pkg.session_autopsy.records import SessionRecord


DENIED = "rm: cannot remove '/home/kuhy/.cache/yay/pkg': Permission denied"


def _candidate(kind: str, sig: str, session_ids: list[str]) -> Candidate:
    """A candidate pointing at the given evidence sessions."""
    return Candidate(
        id=f"{kind}-x",
        kind=kind,
        title="t",
        sig=sig,
        sessions=len(session_ids),
        occurrences=1,
        avg_tokens=1,
        per_week=0.0,
        est_weekly_savings=0,
        score=1.0,
        action="a",
        session_ids=session_ids,
    )


def _session(
    tmp_path: Path, session_id: str, lines: list[dict[str, object] | str]
) -> SessionRecord:
    """Write a transcript and a record whose path points at it."""
    path = write_transcript(tmp_path / "projects", session_id, lines)
    return record(session_id, file=FileFacts(path=str(path)))


def test_skill_traces_full_span(tmp_path: Path) -> None:
    """Skill spans render TOOL/TEXT/RESULT lines and close on prompts."""
    lines: list[dict[str, object] | str] = [
        {"type": "attachment"},
        {"type": "assistant", "message": "junk"},
        {"type": "assistant", "message": {"content": "not-a-list"}},
        prompt_line("start", user_type="agent"),
        prompt_line("side", sidechain=True),
        {"type": "user", "message": "junk"},
        assistant_line(skill_block("finish")),
        assistant_line(
            bash_block("git status"), text_block("Running checks now on this repo.")
        ),
        assistant_line(text_block("   ")),
        tool_result_line("line one\n\nline three\nline four"),
        tool_result_line(None),
        prompt_line("done, thanks"),
        assistant_line(skill_block("other")),
        assistant_line(bash_block("should not appear")),
    ]
    rec = _session(tmp_path, "sess", lines)
    text = traces.render_traces(
        _candidate("skill", "finish", ["sess"]), {"sess": rec}, 5
    )
    assert "TOOL Skill | finish" in text
    assert "TOOL Bash" in text
    assert "TEXT | Running checks now" in text
    assert "RESULT | line one" in text
    assert "should not appear" not in text


def test_skill_traces_ignores_other_block_types(tmp_path: Path) -> None:
    """A block that is neither tool_use nor text contributes nothing."""
    lines: list[dict[str, object] | str] = [
        assistant_line(skill_block("finish")),
        assistant_line({"type": "thinking", "thinking": "quietly"}),
        assistant_line(bash_block("git status")),
    ]
    rec = _session(tmp_path, "sess", lines)
    text = traces.render_traces(
        _candidate("skill", "finish", ["sess"]), {"sess": rec}, 5
    )
    assert "TOOL Skill | finish" in text
    assert "TOOL Bash" in text
    assert "quietly" not in text


def test_capture_block_ignores_unrecognised_blocks() -> None:
    """Only tool_use and text blocks can contribute a line."""
    span = traces._SpanBuffer()
    span.start(matched=True, header="TOOL Skill | finish")
    traces._capture_block(span, {"type": "thinking", "thinking": "quietly"}, "finish")
    traces._capture_block(span, {"type": "tool_result"}, "finish")
    traces._capture_block(span, {}, "finish")
    # A text block still needs actual text: neither a non-string nor whitespace
    # counts.
    traces._capture_block(span, {"type": "text", "text": 5}, "finish")
    traces._capture_block(span, {"type": "text", "text": "   "}, "finish")
    assert span.lines == ["TOOL Skill | finish"]


def test_span_buffer_only_captures_inside_a_matching_span() -> None:
    """Lines outside a span, or inside another skill's span, are dropped."""
    span = traces._SpanBuffer()
    span.add("TEXT | before any span")
    span.start(matched=False, header="TOOL Skill | other")
    span.add("TEXT | wrong skill")
    span.start(matched=True, header="TOOL Skill | finish")
    span.add("TEXT | inside")
    span.close()
    span.add("TEXT | after the span closed")
    assert span.lines == ["TOOL Skill | finish", "TEXT | inside"]


def test_span_buffer_honours_the_per_span_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The header is free; captured lines stop at the cap."""
    monkeypatch.setattr(traces, "MAX_LINES_PER_SPAN", 2)
    span = traces._SpanBuffer()
    span.start(matched=True, header="TOOL Skill | finish")
    for index in range(5):
        span.add(f"TEXT | line {index}")
    assert span.lines == ["TOOL Skill | finish", "TEXT | line 0", "TEXT | line 1"]
    # A fresh span gets a fresh budget.
    span.start(matched=True, header="TOOL Skill | finish")
    span.add("TEXT | second span")
    assert span.lines[-2:] == ["TOOL Skill | finish", "TEXT | second span"]


def test_skill_traces_span_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-span line cap stops TOOL/TEXT/RESULT accumulation."""
    monkeypatch.setattr(traces, "MAX_LINES_PER_SPAN", 1)
    lines: list[dict[str, object] | str] = [
        assistant_line(skill_block("finish")),
        assistant_line(bash_block("one"), bash_block("two")),
        assistant_line(text_block("long narration " * 20)),
        tool_result_line("r1\nr2"),
    ]
    rec = _session(tmp_path, "cap", lines)
    text = traces.render_traces(_candidate("skill", "finish", ["cap"]), {"cap": rec}, 5)
    assert text.count("TOOL Bash") == 1
    assert "TEXT |" not in text


def test_render_traces_missing_and_cap(tmp_path: Path) -> None:
    """Unknown ids are skipped, capture stops at max, empty → message."""
    lines: list[dict[str, object] | str] = [
        assistant_line(skill_block("finish")),
        assistant_line(bash_block("x")),
    ]
    rec_a = _session(tmp_path, "a", lines)
    rec_b = _session(tmp_path, "b", lines)
    gone = record("gone", file=FileFacts(path=str(tmp_path / "nope.jsonl")))
    by_id = {"a": rec_a, "b": rec_b, "gone": gone}
    capped = traces.render_traces(
        _candidate("skill", "finish", ["missing", "a", "b"]), by_id, 1
    )
    assert "## session a" in capped
    assert "## session b" not in capped
    empty = traces.render_traces(_candidate("skill", "finish", ["gone"]), by_id, 5)
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
    rec = _session(tmp_path, "err", lines)
    sig = "rm: cannot remove '<PATH>': Permission denied"
    text = traces.render_traces(_candidate("err", sig, ["err"]), {"err": rec}, 5)
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
    rec = _session(tmp_path, "ng", lines)
    hit = traces.render_traces(
        _candidate("ngram", "git → pytest → git", ["ng"]), {"ng": rec}, 5
    )
    assert "CMD | git status" in hit
    assert "CMD | git push" in hit
    miss = traces.render_traces(
        _candidate("ngram", "adb → adb", ["ng"]), {"ng": rec}, 5
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
    rec = _session(tmp_path, "pr", lines)
    text = traces.render_traces(
        _candidate("prompt", "fix the tests please", ["pr"]), {"pr": rec}, 5
    )
    assert text.count("PROMPT | fix the tests please") == 2
    assert "THEN | Bash" in text
    assert "THEN | Read" in text
    assert text.count("THEN |") == 2


def test_iter_lines_tolerates_junk(tmp_path: Path) -> None:
    """Blank, malformed, and non-dict transcript lines are skipped."""
    lines: list[dict[str, object] | str] = [
        "",
        "{bad",
        "[1]",
        assistant_line(bash_block("ls")),
    ]
    rec = _session(tmp_path, "junk", lines)
    text = traces.render_traces(_candidate("ngram", "ls", ["junk"]), {"junk": rec}, 5)
    assert "CMD | ls" in text


def test_result_texts_and_clip() -> None:
    """toolUseResult text extraction and clipping edge cases."""
    assert traces._result_texts({"toolUseResult": {"stderr": "", "stdout": 3}}) == []
    assert traces._result_texts({"toolUseResult": 42}) == []
    assert traces._result_texts({"toolUseResult": {"stderr": "e", "stdout": "o"}}) == [
        "e",
        "o",
    ]
    assert traces._clip("") == ""
    assert traces._clip("x" * 500) == "x" * traces.MAX_LINE_CHARS
