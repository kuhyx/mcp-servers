"""Dataclass schema for per-session autopsy records (schema v2)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

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
        self.output += _as_int(usage.get("output_tokens"))
        self.input += _as_int(usage.get("input_tokens"))
        self.cache_read += _as_int(usage.get("cache_read_input_tokens"))
        self.cache_write += _as_int(usage.get("cache_creation_input_tokens"))


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


def _meta_from_dict(data: object) -> SessionMeta:
    """Rebuild the meta block from its serialized form.

    Args:
        data: The ``meta`` value of a stored record.

    Returns:
        The reconstructed meta (empty on junk).
    """
    return SessionMeta(
        started_at=_opt_str(_dict_get(data, "started_at")),
        ended_at=_opt_str(_dict_get(data, "ended_at")),
        cwd=_opt_str(_dict_get(data, "cwd")),
        git_branch=_opt_str(_dict_get(data, "git_branch")),
        slug=_opt_str(_dict_get(data, "slug")),
        title=_opt_str(_dict_get(data, "title")),
    )


def _counts_from_dict(data: object) -> ActivityCounts:
    """Rebuild the counts block from its serialized form.

    Args:
        data: The ``counts`` value of a stored record.

    Returns:
        The reconstructed counts (zeros on junk).
    """
    return ActivityCounts(
        assistant_msgs=_as_int(_dict_get(data, "assistant_msgs")),
        external_prompts=_as_int(_dict_get(data, "external_prompts")),
        subagent_count=_as_int(_dict_get(data, "subagent_count")),
        compaction_count=_as_int(_dict_get(data, "compaction_count")),
        retry_prompt_count=_as_int(_dict_get(data, "retry_prompt_count")),
        malformed_lines=_as_int(_dict_get(data, "malformed_lines")),
    )


def _tokens_from_dict(data: object) -> TokenTotals:
    """Rebuild token totals from their serialized form.

    Args:
        data: The ``tokens`` value of a stored record.

    Returns:
        The reconstructed totals (zeros on junk).
    """
    return TokenTotals(
        output=_as_int(_dict_get(data, "output")),
        input=_as_int(_dict_get(data, "input")),
        cache_read=_as_int(_dict_get(data, "cache_read")),
        cache_write=_as_int(_dict_get(data, "cache_write")),
    )


def _turns_from_dict(data: object) -> TurnEfficiency:
    """Rebuild turn-efficiency counts from their serialized form.

    Args:
        data: The ``turns`` value of a stored record.

    Returns:
        The reconstructed counts (zeros on junk or on pre-v3 records).
    """
    return TurnEfficiency(
        api_turns=_as_int(_dict_get(data, "api_turns")),
        tool_calls=_as_int(_dict_get(data, "tool_calls")),
        batched_turns=_as_int(_dict_get(data, "batched_turns")),
        poll_turns=_as_int(_dict_get(data, "poll_turns")),
        lint_test_turns=_as_int(_dict_get(data, "lint_test_turns")),
        vcs_check_turns=_as_int(_dict_get(data, "vcs_check_turns")),
    )


def _obs_from_dict(data: object) -> Observations:
    """Rebuild the observations block from its serialized form.

    Args:
        data: The ``obs`` value of a stored record.

    Returns:
        The reconstructed observations (empty on junk).
    """
    invocations = [
        SkillInvocation(
            name=str(_dict_get(raw, "name") or "?"),
            tool_only_turns=_as_int(_dict_get(raw, "tool_only_turns")),
            text_turns=_as_int(_dict_get(raw, "text_turns")),
            tokens_output=_as_int(_dict_get(raw, "tokens_output")),
            tokens_cache_write=_as_int(_dict_get(raw, "tokens_cache_write")),
            bash_sig=_str_list(_dict_get(raw, "bash_sig")),
        )
        for raw in _list_of_dicts(_dict_get(data, "skill_invocations"))
    ]
    return Observations(
        tool_histogram=_str_int_map(_dict_get(data, "tool_histogram")),
        bash_first_lines=_str_int_map(_dict_get(data, "bash_first_lines")),
        bash_sequences=[
            _str_list(seq) for seq in _list_value(_dict_get(data, "bash_sequences"))
        ],
        skill_invocations=invocations,
        error_signatures=_str_int_map(_dict_get(data, "error_signatures")),
        typed_prompt_signatures=_str_int_map(
            _dict_get(data, "typed_prompt_signatures")
        ),
    )


def record_from_dict(data: dict[str, object]) -> SessionRecord:
    """Rebuild a :class:`SessionRecord` from its serialized form.

    Args:
        data: A dict previously produced by :meth:`SessionRecord.to_dict`.

    Returns:
        The reconstructed record; unknown keys are ignored so old readers
        survive future additive schema changes.
    """
    return SessionRecord(
        session_id=str(data.get("session_id") or ""),
        project_slug=str(data.get("project_slug") or ""),
        transcript_path=str(data.get("transcript_path") or ""),
        file_size=_as_int(data.get("file_size")),
        file_mtime=_as_float(data.get("file_mtime")),
        analyzed_at=str(data.get("analyzed_at") or ""),
        schema_version=_as_int(data.get("schema_version")),
        meta=_meta_from_dict(data.get("meta")),
        counts=_counts_from_dict(data.get("counts")),
        tokens=_tokens_from_dict(data.get("tokens")),
        turns=_turns_from_dict(data.get("turns")),
        obs=_obs_from_dict(data.get("obs")),
    )


def _as_int(value: object) -> int:
    """Coerce a JSON value to int, treating bools and junk as 0.

    Args:
        value: Any value read from a transcript or store line.

    Returns:
        The int value, or 0 when absent/bool/non-numeric.
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _as_float(value: object) -> float:
    """Coerce a JSON value to float, treating bools and junk as 0.0.

    Args:
        value: Any value read from a transcript or store line.

    Returns:
        The float value, or 0.0 when absent/bool/non-numeric.
    """
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _opt_str(value: object) -> str | None:
    """Return the value if it is a non-empty string, else None.

    Args:
        value: Any JSON value.

    Returns:
        The string, or None.
    """
    if isinstance(value, str) and value:
        return value
    return None


def _dict_get(value: object, key: str) -> object:
    """Safely index into a value that may not be a dict.

    Args:
        value: Any JSON value.
        key: Key to read when the value is a dict.

    Returns:
        The mapped value, or None.
    """
    if isinstance(value, dict):
        return value.get(key)
    return None


def _list_value(value: object) -> list[object]:
    """Return the value if it is a list, else an empty list.

    Args:
        value: Any JSON value.

    Returns:
        A list (possibly empty).
    """
    if isinstance(value, list):
        return value
    return []


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    """Filter a JSON value down to its dict elements.

    Args:
        value: Any JSON value.

    Returns:
        The dict elements of the list, or an empty list.
    """
    return [item for item in _list_value(value) if isinstance(item, dict)]


def _str_list(value: object) -> list[str]:
    """Filter a JSON value down to its string elements.

    Args:
        value: Any JSON value.

    Returns:
        The string elements of the list, or an empty list.
    """
    return [item for item in _list_value(value) if isinstance(item, str)]


def _str_int_map(value: object) -> dict[str, int]:
    """Coerce a JSON value to a ``str -> int`` counter mapping.

    Args:
        value: Any JSON value.

    Returns:
        A dict of the string keys to int counts, or an empty dict.
    """
    if not isinstance(value, dict):
        return {}
    return {str(key): _as_int(val) for key, val in value.items()}
