"""Per-session accumulator fed one transcript line at a time."""

from __future__ import annotations

import json
from collections import Counter

from session_autopsy.parse_constants import (
    BUILTIN_COMMANDS,
    COMPACT_PREFIX,
    GEN_TURN_MIN_CHARS,
    SLASH_COMMAND_RE,
)
from session_autopsy.parse_helpers import (
    error_signatures,
    generated_chars,
    is_retry_prompt,
    message_id,
    note_metadata,
    note_title,
    usage_int,
)
from session_autopsy.parse_spans import SpanTracker
from session_autopsy.records import (
    ActivityCounts,
    SessionMeta,
    SkillInvocation,
    TokenTotals,
    TurnEfficiency,
)
from session_autopsy.signatures import command_signature, normalize_signature
from session_autopsy.turn_tracker import TurnTracker


class Accumulator:
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
        self._spans = SpanTracker()
        # One API response can span several transcript lines, so batching has to be
        # grouped by requestId — counting per line would report every turn as
        # single-tool and hide the real (much lower) batching rate.
        self._turns = TurnTracker()

    @property
    def bash_sequences(self) -> list[list[str]]:
        """Closed Bash runs kept so far."""
        return self._spans.bash_sequences

    @property
    def skill_invocations(self) -> list[SkillInvocation]:
        """Closed skill spans kept so far."""
        return self._spans.skill_invocations

    @property
    def turns(self) -> TurnEfficiency:
        """Turn-efficiency totals accumulated so far."""
        return self._turns.totals

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
        """Flush any still-open Bash run, skill span and API turn."""
        self._spans.flush_all()
        self._turns.flush()

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
            note_title(self.meta, obj)
        elif line_type == "system" and obj.get("subtype") == "compact_boundary":
            self.counts.compaction_count += 1

    def _on_assistant(self, obj: dict[str, object]) -> None:
        """Account one assistant message: usage, tools, turn class.

        Args:
            obj: An ``assistant`` transcript line.
        """
        message = obj.get("message")
        if not isinstance(message, dict):
            return
        note_metadata(self.meta, obj)
        self.counts.assistant_msgs += 1
        self._turns.begin(obj.get("requestId") or message_id(message))
        usage = message.get("usage")
        if isinstance(usage, dict):
            self.tokens.add_usage(usage)
        text_chars, had_tool_use = self._scan_blocks(message)
        skill = self._spans.current_skill
        if skill is not None:
            skill.tokens_output += usage_int(usage, "output_tokens")
            skill.tokens_cache_write += usage_int(usage, "cache_creation_input_tokens")
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
                self._turns.add_tool()
                text_chars += generated_chars(block)
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
        self._spans.flush_run()
        if name == "Skill":
            self._spans.flush_skill()
            skill_name = tool_input.get("skill")
            self._spans.current_skill = SkillInvocation(name=str(skill_name or "?"))

    def _on_bash(self, tool_input: dict[str, object]) -> None:
        """Account one Bash tool_use: first-line counter, run, skill sig.

        Args:
            tool_input: The Bash tool's ``input`` object.
        """
        command = tool_input.get("command")
        if not isinstance(command, str):
            return
        self._turns.add_command(command)
        first_line = normalize_signature(command)
        if first_line:
            self.bash_first_lines[first_line] += 1
        sig = command_signature(command)
        if not sig:
            return
        self._spans.add_signature(sig)

    def _on_user(self, obj: dict[str, object]) -> None:
        """Account one user line: typed prompts, retries, tool errors.

        Args:
            obj: A ``user`` transcript line.
        """
        message = obj.get("message")
        if not isinstance(message, dict):
            return
        note_metadata(self.meta, obj)
        self._scan_tool_result(obj.get("toolUseResult"))
        content = message.get("content")
        is_external = (
            obj.get("userType") == "external" and obj.get("isSidechain") is not True
        )
        if not (is_external and isinstance(content, str)):
            return
        if content.startswith(COMPACT_PREFIX):
            self.counts.compaction_count += 1
            return
        slash = SLASH_COMMAND_RE.search(content)
        if slash is not None:
            self._spans.flush_all()
            if slash.group(1) not in BUILTIN_COMMANDS:
                self._spans.current_skill = SkillInvocation(name=slash.group(1))
            return
        self.counts.external_prompts += 1
        self._spans.flush_all()
        if is_retry_prompt(content):
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
        for sig in error_signatures(result):
            self.error_signatures[sig] += 1
