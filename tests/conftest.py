"""Shared fixtures and builders for the session-autopsy tests.

Fixtures isolate the store and transcript root per test; the builders below make
synthetic transcript lines, transcript files, and records.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING

import pytest

from python_pkg.session_autopsy.records import (
    ActivityCounts,
    Observations,
    SessionMeta,
    SessionRecord,
    SkillInvocation,
    TokenTotals,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class AutopsyEnv:
    """Paths of the per-test isolated store and transcript root."""

    home: Path
    projects: Path


@pytest.fixture
def autopsy_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AutopsyEnv:
    """Point the analyzer at temp dirs so tests never touch ~/.claude."""
    home = tmp_path / "autopsy"
    projects = tmp_path / "projects"
    projects.mkdir()
    monkeypatch.setenv("CLAUDE_AUTOPSY_HOME", str(home))
    monkeypatch.setenv("CLAUDE_PROJECTS_DIR", str(projects))
    return AutopsyEnv(home=home, projects=projects)


Line = dict[str, object]

STAMP = "2026-07-20T10:00:00.000Z"


def usage_block(output: int = 10, cache_write: int = 20) -> dict[str, int]:
    """A minimal assistant usage block."""
    return {
        "output_tokens": output,
        "input_tokens": 1,
        "cache_read_input_tokens": 100,
        "cache_creation_input_tokens": cache_write,
    }


def text_block(text: str) -> Line:
    """A text content block."""
    return {"type": "text", "text": text}


def tool_block(name: str, **tool_input: object) -> Line:
    """A tool_use content block."""
    return {"type": "tool_use", "name": name, "input": dict(tool_input)}


def bash_block(command: str) -> Line:
    """A Bash tool_use block."""
    return tool_block("Bash", command=command)


def skill_block(skill: str) -> Line:
    """A Skill tool_use block."""
    return tool_block("Skill", skill=skill)


def assistant_line(
    *blocks: Line, timestamp: str = STAMP, with_usage: bool = True
) -> Line:
    """An assistant transcript line with the given content blocks."""
    message: dict[str, object] = {"role": "assistant", "content": list(blocks)}
    if with_usage:
        message["usage"] = usage_block()
    return {"type": "assistant", "timestamp": timestamp, "message": message}


def prompt_line(
    content: str,
    *,
    user_type: str = "external",
    sidechain: bool = False,
    timestamp: str = STAMP,
) -> Line:
    """A user transcript line with string content (a typed prompt)."""
    return {
        "type": "user",
        "userType": user_type,
        "isSidechain": sidechain,
        "timestamp": timestamp,
        "message": {"role": "user", "content": content},
    }


def tool_result_line(result: object) -> Line:
    """A user transcript line carrying a toolUseResult."""
    return {
        "type": "user",
        "userType": "external",
        "message": {"role": "user", "content": [{"type": "tool_result"}]},
        "toolUseResult": result,
    }


def write_transcript(
    projects: Path,
    session_id: str,
    lines: list[Line | str],
    project: str = "-home-kuhy",
) -> Path:
    """Write transcript lines (dicts, or raw strings verbatim) as JSONL."""
    project_dir = projects / project
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / f"{session_id}.jsonl"
    rendered = [line if isinstance(line, str) else json.dumps(line) for line in lines]
    path.write_text("\n".join(rendered) + "\n", encoding="utf-8")
    return path


def invocation(
    name: str = "finish",
    tool_only: int = 6,
    text: int = 4,
    output: int = 50_000,
    cache_write: int = 100_000,
    bash_sig: list[str] | None = None,
) -> SkillInvocation:
    """A skill invocation span with sensible defaults."""
    return SkillInvocation(
        name=name,
        tool_only_turns=tool_only,
        text_turns=text,
        tokens_output=output,
        tokens_cache_write=cache_write,
        bash_sig=bash_sig if bash_sig is not None else ["pre-commit", "pytest", "git"],
    )


@dataclass(frozen=True)
class FileFacts:
    """The transcript-file scalars a record carries, defaulted for tests."""

    path: str | None = None
    size: int = 100
    mtime: float = 1.0


def session_meta(
    started_at: str | None = STAMP, slug: str | None = None
) -> SessionMeta:
    """SessionMeta carrying the standard test timestamp unless overridden.

    Args:
        started_at: Session start stamp; defaults to :data:`STAMP`.
        slug: Optional project slug.

    Returns:
        The metadata block.
    """
    return SessionMeta(started_at=started_at, slug=slug)


def record(
    session_id: str = "s1",
    *,
    file: FileFacts | None = None,
    meta: SessionMeta | None = None,
    counts: ActivityCounts | None = None,
    tokens: TokenTotals | None = None,
    obs: Observations | None = None,
) -> SessionRecord:
    """A SessionRecord with sensible defaults; override a whole block at a time.

    Grouping the overrides by the record's own blocks is what keeps this builder
    to six parameters; the flat kwargs it used to take were one per field.

    Args:
        session_id: The session id, also used for the default transcript path.
        file: Transcript-file scalars.
        meta: Session metadata; defaults to :func:`session_meta`.
        counts: Activity counts; defaults to ten assistant messages.
        tokens: Token totals; defaults to a typical spend.
        obs: Observations; defaults to empty.

    Returns:
        The record.
    """
    facts = file or FileFacts()
    return SessionRecord(
        session_id=session_id,
        project_slug="-home-kuhy",
        transcript_path=facts.path or f"/nonexistent/{session_id}.jsonl",
        file_size=facts.size,
        file_mtime=facts.mtime,
        analyzed_at="2026-07-20T10:00:00+00:00",
        meta=meta if meta is not None else session_meta(),
        counts=counts if counts is not None else ActivityCounts(assistant_msgs=10),
        tokens=tokens
        if tokens is not None
        else TokenTotals(
            output=10_000, input=100, cache_read=500_000, cache_write=20_000
        ),
        obs=obs if obs is not None else Observations(),
    )
