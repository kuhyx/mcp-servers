"""Tests for skill-span excerpting and the transcript reading helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from session_autopsy import trace_io, trace_skill, traces
from session_autopsy.tests.conftest import (
    assistant_line,
    bash_block,
    prompt_line,
    skill_block,
    text_block,
    tool_result_line,
)
from session_autopsy.tests.trace_helpers import (
    trace_candidate,
    trace_session,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


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
    rec = trace_session(tmp_path, "sess", lines)
    text = traces.render_traces(
        trace_candidate("skill", "finish", ["sess"]), {"sess": rec}, 5
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
    rec = trace_session(tmp_path, "sess", lines)
    text = traces.render_traces(
        trace_candidate("skill", "finish", ["sess"]), {"sess": rec}, 5
    )
    assert "TOOL Skill | finish" in text
    assert "TOOL Bash" in text
    assert "quietly" not in text


def test_capture_block_ignores_unrecognised_blocks() -> None:
    """Only tool_use and text blocks can contribute a line."""
    span = trace_skill.SpanBuffer()
    span.start(matched=True, header="TOOL Skill | finish")
    trace_skill.capture_block(
        span, {"type": "thinking", "thinking": "quietly"}, "finish"
    )
    trace_skill.capture_block(span, {"type": "tool_result"}, "finish")
    trace_skill.capture_block(span, {}, "finish")
    # A text block still needs actual text: neither a non-string nor whitespace
    # counts.
    trace_skill.capture_block(span, {"type": "text", "text": 5}, "finish")
    trace_skill.capture_block(span, {"type": "text", "text": "   "}, "finish")
    assert span.lines == ["TOOL Skill | finish"]


def test_span_buffer_only_captures_inside_a_matching_span() -> None:
    """Lines outside a span, or inside another skill's span, are dropped."""
    span = trace_skill.SpanBuffer()
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
    monkeypatch.setattr(trace_skill, "MAX_LINES_PER_SPAN", 2)
    span = trace_skill.SpanBuffer()
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
    monkeypatch.setattr(trace_skill, "MAX_LINES_PER_SPAN", 1)
    lines: list[dict[str, object] | str] = [
        assistant_line(skill_block("finish")),
        assistant_line(bash_block("one"), bash_block("two")),
        assistant_line(text_block("long narration " * 20)),
        tool_result_line("r1\nr2"),
    ]
    rec = trace_session(tmp_path, "cap", lines)
    text = traces.render_traces(
        trace_candidate("skill", "finish", ["cap"]), {"cap": rec}, 5
    )
    assert text.count("TOOL Bash") == 1
    assert "TEXT |" not in text


def test_iter_lines_tolerates_junk(tmp_path: Path) -> None:
    """Blank, malformed, and non-dict transcript lines are skipped."""
    lines: list[dict[str, object] | str] = [
        "",
        "{bad",
        "[1]",
        assistant_line(bash_block("ls")),
    ]
    rec = trace_session(tmp_path, "junk", lines)
    text = traces.render_traces(
        trace_candidate("ngram", "ls", ["junk"]), {"junk": rec}, 5
    )
    assert "CMD | ls" in text


def test_result_texts_and_clip() -> None:
    """toolUseResult text extraction and clipping edge cases."""
    assert trace_io.result_texts({"toolUseResult": {"stderr": "", "stdout": 3}}) == []
    assert trace_io.result_texts({"toolUseResult": 42}) == []
    assert trace_io.result_texts({"toolUseResult": {"stderr": "e", "stdout": "o"}}) == [
        "e",
        "o",
    ]
    assert trace_io.clip("") == ""
    assert trace_io.clip("x" * 500) == "x" * trace_io.MAX_LINE_CHARS
