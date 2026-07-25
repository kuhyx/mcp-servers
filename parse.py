"""Streaming parser turning one transcript (plus subagents) into a record.

Transcripts can reach 12 MB, so every file is read line-by-line; no line's
parse result outlives the accumulator fields it updates.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import re
from typing import TYPE_CHECKING

from python_pkg.session_autopsy.records import (
    ActivityCounts,
    Observations,
    SessionMeta,
    SessionRecord,
    SkillInvocation,
    TokenTotals,
)
from python_pkg.session_autopsy.signatures import command_signature, normalize_signature

if TYPE_CHECKING:
    from pathlib import Path

GEN_TURN_MIN_CHARS = 400
MAX_BASH_RUN = 30
MIN_BASH_RUN = 3
MAX_ERROR_SIGS = 20
MAX_PROMPT_SIGS = 40
MAX_BASH_FIRST_LINES = 40
MAX_SKILL_BASH_SIG = 60
RESULT_SCAN_LINES = 10
RESULT_TAIL_LINES = 5
RETRY_PROMPT_MAX_LEN = 70

_ERROR_LINE_RE = re.compile(r"^(?:rm: |fatal: |Traceback)|\w*error\b", re.IGNORECASE)
_COMPACT_PREFIX = "This session is being continued from a previous conversation"
_SLASH_COMMAND_RE = re.compile(r"<command-name>/?([\w:-]+)</command-name>")
# Harness built-ins that appear as <command-name> but are not skills.
_BUILTIN_COMMANDS = frozenset(
    {
        "clear",
        "compact",
        "config",
        "context",
        "cost",
        "doctor",
        "effort",
        "exit",
        "fast",
        "help",
        "hooks",
        "ide",
        "login",
        "logout",
        "loop",
        "mcp",
        "memory",
        "model",
        "permissions",
        "plan",
        "plugin",
        "reload-plugins",
        "remote-control",
        "rename",
        "resume",
        "skills",
        "status",
        "usage",
    },
)
# Tool inputs that carry generated content (prose/code the model wrote).
_GENERATIVE_INPUT_KEYS = ("content", "new_string")
# Error lines too generic to act on; the informative line is elsewhere.
_GENERIC_ERROR_SIGS = frozenset(
    {
        "Error: Exit code <N>",
        "Traceback (most recent call last):",
    },
)


class _Accumulator:
    """Mutable per-session state fed one transcript line at a time."""

    def __init__(self) -> None:
        """Initialize empty counters and open-span state."""
        self.tokens = TokenTotals()
        self.meta = SessionMeta()
        self.counts = ActivityCounts()
        self.tool_histogram: Counter[str] = Counter()
        self.bash_first_lines: Counter[str] = Counter()
        self.error_signatures: Counter[str] = Counter()
        self.typed_prompt_signatures: Counter[str] = Counter()
        self.bash_sequences: list[list[str]] = []
        self.skill_invocations: list[SkillInvocation] = []
        self._current_run: list[str] = []
        self._current_skill: SkillInvocation | None = None

    def feed_line(self, raw_line: str) -> None:
        """Parse and account one transcript line.

        Args:
            raw_line: One line of the JSONL transcript.
        """
        stripped = raw_line.strip()
        if not stripped:
            return
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            self.counts.malformed_lines += 1
            return
        if not isinstance(obj, dict):
            self.counts.malformed_lines += 1
            return
        self._dispatch(obj)

    def finish(self) -> None:
        """Flush any still-open Bash run and skill span."""
        self._flush_run()
        self._flush_skill()

    def _dispatch(self, obj: dict[str, object]) -> None:
        """Route a parsed line by its ``type``.

        Args:
            obj: The parsed transcript line.
        """
        line_type = obj.get("type")
        if line_type == "assistant":
            self._on_assistant(obj)
        elif line_type == "user":
            self._on_user(obj)
        elif line_type == "ai-title":
            self._on_title(obj)
        elif line_type == "system" and obj.get("subtype") == "compact_boundary":
            self.counts.compaction_count += 1

    def _on_title(self, obj: dict[str, object]) -> None:
        """Record the session's AI-generated title (last one wins).

        Args:
            obj: An ``ai-title`` transcript line.
        """
        for key in ("title", "aiTitle"):
            value = obj.get(key)
            if isinstance(value, str) and value:
                self.meta.title = value

    def _note_metadata(self, obj: dict[str, object]) -> None:
        """Capture session metadata from the first message line carrying it.

        Args:
            obj: A message-bearing transcript line.
        """
        timestamp = obj.get("timestamp")
        if isinstance(timestamp, str):
            if self.meta.started_at is None:
                self.meta.started_at = timestamp
            self.meta.ended_at = timestamp
        if self.meta.cwd is None and isinstance(obj.get("cwd"), str):
            self.meta.cwd = str(obj["cwd"])
        if self.meta.git_branch is None and isinstance(obj.get("gitBranch"), str):
            self.meta.git_branch = str(obj["gitBranch"])
        if self.meta.slug is None and isinstance(obj.get("slug"), str):
            self.meta.slug = str(obj["slug"])

    def _on_assistant(self, obj: dict[str, object]) -> None:
        """Account one assistant message: usage, tools, turn class.

        Args:
            obj: An ``assistant`` transcript line.
        """
        message = obj.get("message")
        if not isinstance(message, dict):
            return
        self._note_metadata(obj)
        self.counts.assistant_msgs += 1
        usage = message.get("usage")
        if isinstance(usage, dict):
            self.tokens.add_usage(usage)
        text_chars, had_tool_use = self._scan_blocks(message)
        if self._current_skill is not None:
            skill = self._current_skill
            skill.tokens_output += _usage_int(usage, "output_tokens")
            skill.tokens_cache_write += _usage_int(usage, "cache_creation_input_tokens")
            if had_tool_use and text_chars <= GEN_TURN_MIN_CHARS:
                skill.tool_only_turns += 1
            else:
                skill.text_turns += 1

    def _scan_blocks(self, message: dict[str, object]) -> tuple[int, bool]:
        """Walk an assistant message's content blocks.

        Args:
            message: The ``message`` object of an assistant line.

        Returns:
            Total text characters and whether any tool_use block was present.
        """
        text_chars = 0
        had_tool_use = False
        content = message.get("content")
        if not isinstance(content, list):
            return (0, False)
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if isinstance(text, str):
                    text_chars += len(text.strip())
            elif block_type == "tool_use":
                had_tool_use = True
                text_chars += _generated_chars(block)
                self._on_tool_use(block)
        return (text_chars, had_tool_use)

    def _on_tool_use(self, block: dict[str, object]) -> None:
        """Account one tool_use block (histogram, Bash runs, skill spans).

        Args:
            block: A ``tool_use`` content block.
        """
        name = str(block.get("name") or "?")
        self.tool_histogram[name] += 1
        tool_input = block.get("input")
        if not isinstance(tool_input, dict):
            tool_input = {}
        if name == "Bash":
            self._on_bash(tool_input)
            return
        self._flush_run()
        if name == "Skill":
            self._flush_skill()
            skill_name = tool_input.get("skill")
            self._current_skill = SkillInvocation(name=str(skill_name or "?"))

    def _on_bash(self, tool_input: dict[str, object]) -> None:
        """Account one Bash tool_use: first-line counter, run, skill sig.

        Args:
            tool_input: The Bash tool's ``input`` object.
        """
        command = tool_input.get("command")
        if not isinstance(command, str):
            return
        first_line = normalize_signature(command)
        if first_line:
            self.bash_first_lines[first_line] += 1
        sig = command_signature(command)
        if not sig:
            return
        if self._current_skill is not None:
            if len(self._current_skill.bash_sig) < MAX_SKILL_BASH_SIG:
                self._current_skill.bash_sig.append(sig)
        elif len(self._current_run) < MAX_BASH_RUN:
            self._current_run.append(sig)

    def _on_user(self, obj: dict[str, object]) -> None:
        """Account one user line: typed prompts, retries, tool errors.

        Args:
            obj: A ``user`` transcript line.
        """
        message = obj.get("message")
        if not isinstance(message, dict):
            return
        self._note_metadata(obj)
        self._scan_tool_result(obj.get("toolUseResult"))
        content = message.get("content")
        is_external = (
            obj.get("userType") == "external" and obj.get("isSidechain") is not True
        )
        if not (is_external and isinstance(content, str)):
            return
        if content.startswith(_COMPACT_PREFIX):
            self.counts.compaction_count += 1
            return
        slash = _SLASH_COMMAND_RE.search(content)
        if slash is not None:
            self._flush_run()
            self._flush_skill()
            if slash.group(1) not in _BUILTIN_COMMANDS:
                self._current_skill = SkillInvocation(name=slash.group(1))
            return
        self.counts.external_prompts += 1
        self._flush_run()
        self._flush_skill()
        lowered = content.lower()
        if (
            len(content) <= RETRY_PROMPT_MAX_LEN
            and "limit" in lowered
            and ("reset" in lowered or "continue" in lowered)
        ):
            self.counts.retry_prompt_count += 1
        if content.startswith(("<", "[")):
            return
        sig = normalize_signature(content)
        if sig:
            self.typed_prompt_signatures[sig] += 1

    def _scan_tool_result(self, result: object) -> None:
        """Mine a ``toolUseResult`` for repeated error lines.

        Args:
            result: The top-level ``toolUseResult`` value (str, dict, or absent).
        """
        texts: list[str] = []
        if isinstance(result, str):
            texts.append(result)
        elif isinstance(result, dict):
            for key in ("stderr", "stdout"):
                value = result.get(key)
                if isinstance(value, str) and value:
                    texts.append(value)
        for text in texts:
            all_lines = text.splitlines()
            scan = all_lines[:RESULT_SCAN_LINES]
            if len(all_lines) > RESULT_SCAN_LINES:
                scan += all_lines[-RESULT_TAIL_LINES:]
            for line in scan:
                if _ERROR_LINE_RE.search(line):
                    sig = normalize_signature(line)
                    if sig and sig not in _GENERIC_ERROR_SIGS:
                        self.error_signatures[sig] += 1

    def _flush_run(self) -> None:
        """Close the current Bash run, keeping it only if long enough."""
        if len(self._current_run) >= MIN_BASH_RUN:
            self.bash_sequences.append(self._current_run)
        self._current_run = []

    def _flush_skill(self) -> None:
        """Close the current skill span, dropping spans with no turns."""
        skill = self._current_skill
        if skill is not None:
            if skill.tool_only_turns + skill.text_turns > 0:
                self.skill_invocations.append(skill)
            self._current_skill = None


def _generated_chars(block: dict[str, object]) -> int:
    """Characters of model-generated content hidden in a tool_use input.

    Write/Edit-style inputs carry the prose or code the model wrote; counting
    them stops generation-heavy skills from masquerading as tool-only.

    Args:
        block: A ``tool_use`` content block.

    Returns:
        Character count of generative input fields.
    """
    tool_input = block.get("input")
    if not isinstance(tool_input, dict):
        return 0
    return sum(
        len(value)
        for key in _GENERATIVE_INPUT_KEYS
        if isinstance(value := tool_input.get(key), str)
    )


def _usage_int(usage: object, key: str) -> int:
    """Read one int field from a possibly-absent usage dict.

    Args:
        usage: The ``message.usage`` value.
        key: Field name to read.

    Returns:
        The int value, or 0.
    """
    if isinstance(usage, dict):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


def _top(counter: Counter[str], limit: int) -> dict[str, int]:
    """Keep only the most common counter entries, deterministically ordered.

    Args:
        counter: Any string counter.
        limit: Maximum number of entries to keep.

    Returns:
        A dict of the top entries, ties broken alphabetically.
    """
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return dict(ranked[:limit])


def subagent_files(transcript_path: Path) -> list[Path]:
    """Locate the nested subagent transcripts of a session.

    Args:
        transcript_path: Path to the main ``<uuid>.jsonl`` transcript.

    Returns:
        Sorted ``agent-*.jsonl`` paths under ``<uuid>/subagents/``.
    """
    nested = transcript_path.parent / transcript_path.stem / "subagents"
    if not nested.is_dir():
        return []
    return sorted(nested.glob("agent-*.jsonl"))


def parse_session(transcript_path: Path) -> SessionRecord:
    """Parse one session transcript (and its subagents) into a record.

    Subagent token usage and tool calls fold into the parent session's
    totals; subagent prompts never count as external (they are sidechains).

    Args:
        transcript_path: Path to the main ``<uuid>.jsonl`` transcript.

    Returns:
        The populated :class:`SessionRecord`.
    """
    acc = _Accumulator()
    subagents = subagent_files(transcript_path)
    for path in [transcript_path, *subagents]:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                acc.feed_line(raw_line)
        acc.finish()
    stat = transcript_path.stat()
    # The accumulator owns the metadata and counts records outright, so they are
    # handed over rather than copied field by field. Only the subagent count is
    # not something a transcript line can carry: it comes from the file listing.
    acc.counts.subagent_count = len(subagents)
    return SessionRecord(
        session_id=transcript_path.stem,
        project_slug=transcript_path.parent.name,
        transcript_path=str(transcript_path),
        file_size=stat.st_size,
        file_mtime=stat.st_mtime,
        analyzed_at=datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        meta=acc.meta,
        counts=acc.counts,
        tokens=acc.tokens,
        obs=Observations(
            tool_histogram=dict(acc.tool_histogram),
            bash_first_lines=_top(acc.bash_first_lines, MAX_BASH_FIRST_LINES),
            bash_sequences=acc.bash_sequences,
            skill_invocations=acc.skill_invocations,
            error_signatures=_top(acc.error_signatures, MAX_ERROR_SIGS),
            typed_prompt_signatures=_top(acc.typed_prompt_signatures, MAX_PROMPT_SIGS),
        ),
    )
