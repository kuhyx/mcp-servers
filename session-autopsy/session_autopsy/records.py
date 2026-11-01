"""Dataclass schema for per-session autopsy records (schema v2)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from session_autopsy.record_values import as_int

SCHEMA_VERSION = 3


@dataclass
class TokenTotals:
    """Token usage summed over every assistant message in a session."""

    output: int = 0
    input: int = 0
    cache_read: int = 0
    cache_write: int = 0

    def add_usage(self, usage: dict[str, object]) -> None:
        """Accumulate one assistant message's ``usage`` block.

        Args:
            usage: The ``message.usage`` object from a transcript line.
        """
        self.output += as_int(usage.get("output_tokens"))
        self.input += as_int(usage.get("input_tokens"))
        self.cache_read += as_int(usage.get("cache_read_input_tokens"))
        self.cache_write += as_int(usage.get("cache_creation_input_tokens"))


@dataclass
class TurnEfficiency:
    """API round-trips taken, and how many of them a script could have taken.

    Cache-read tokens dominate a long session (one measured session: 235M read
    against 800k output), and cache read is charged per round-trip on the whole
    conversation so far. So the cost lever is the NUMBER of turns, not the length
    of any message — and turns where the model decided nothing are pure waste.
    """

    api_turns: int = 0
    tool_calls: int = 0
    batched_turns: int = 0
    poll_turns: int = 0
    lint_test_turns: int = 0
    vcs_check_turns: int = 0

    @property
    def mechanical_turns(self) -> int:
        """Turns whose every action was observation a script could have done."""
        return self.poll_turns + self.lint_test_turns + self.vcs_check_turns

    @property
    def batching_rate(self) -> float:
        """Share of tool-carrying turns that issued more than one call (0..1)."""
        return self.batched_turns / self.api_turns if self.api_turns else 0.0

    @property
    def mechanical_rate(self) -> float:
        """Share of tool-carrying turns that were purely mechanical (0..1)."""
        return self.mechanical_turns / self.api_turns if self.api_turns else 0.0


@dataclass
class SessionMeta:
    """Where and when the session happened."""

    started_at: str | None = None
    ended_at: str | None = None
    cwd: str | None = None
    git_branch: str | None = None
    slug: str | None = None
    title: str | None = None


@dataclass
class ActivityCounts:
    """How much happened, by category."""

    assistant_msgs: int = 0
    external_prompts: int = 0
    subagent_count: int = 0
    compaction_count: int = 0
    retry_prompt_count: int = 0
    malformed_lines: int = 0


@dataclass
class SkillInvocation:
    """One skill span (Skill tool_use or slash command) and its turns."""

    name: str
    tool_only_turns: int = 0
    text_turns: int = 0
    tokens_output: int = 0
    tokens_cache_write: int = 0
    bash_sig: list[str] = field(default_factory=list)


@dataclass
class Observations:
    """Signals mined from the transcript for the detectors."""

    tool_histogram: dict[str, int] = field(default_factory=dict)
    bash_first_lines: dict[str, int] = field(default_factory=dict)
    bash_sequences: list[list[str]] = field(default_factory=list)
    skill_invocations: list[SkillInvocation] = field(default_factory=list)
    error_signatures: dict[str, int] = field(default_factory=dict)
    typed_prompt_signatures: dict[str, int] = field(default_factory=dict)


@dataclass
class SessionRecord:
    """Compact, deterministic summary of one Claude Code session transcript."""

    session_id: str
    project_slug: str
    transcript_path: str
    file_size: int
    file_mtime: float
    analyzed_at: str
    schema_version: int = SCHEMA_VERSION
    meta: SessionMeta = field(default_factory=SessionMeta)
    counts: ActivityCounts = field(default_factory=ActivityCounts)
    tokens: TokenTotals = field(default_factory=TokenTotals)
    turns: TurnEfficiency = field(default_factory=TurnEfficiency)
    obs: Observations = field(default_factory=Observations)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dict.

        Returns:
            A plain dict with nested dataclasses expanded.
        """
        data = asdict(self)
        # asdict() only walks fields, so the derived rates are added by hand —
        # readers of sessions.jsonl should not have to recompute them.
        data["turns"] |= {
            "mechanical_turns": self.turns.mechanical_turns,
            "batching_rate": round(self.turns.batching_rate, 4),
            "mechanical_rate": round(self.turns.mechanical_rate, 4),
        }
        return data
