"""Rebuild session records from their stored JSON form."""

from __future__ import annotations

from session_autopsy.record_values import (
    as_float,
    as_int,
    dict_get,
    list_of_dicts,
    list_value,
    opt_str,
    str_int_map,
    str_list,
)
from session_autopsy.records import (
    ActivityCounts,
    Observations,
    SessionMeta,
    SessionRecord,
    SkillInvocation,
    TokenTotals,
    TurnEfficiency,
)


def _meta_from_dict(data: object) -> SessionMeta:
    """Rebuild the meta block from its serialized form.

    Args:
        data: The ``meta`` value of a stored record.

    Returns:
        The reconstructed meta (empty on junk).
    """
    return SessionMeta(
        started_at=opt_str(dict_get(data, "started_at")),
        ended_at=opt_str(dict_get(data, "ended_at")),
        cwd=opt_str(dict_get(data, "cwd")),
        git_branch=opt_str(dict_get(data, "git_branch")),
        slug=opt_str(dict_get(data, "slug")),
        title=opt_str(dict_get(data, "title")),
    )


def _counts_from_dict(data: object) -> ActivityCounts:
    """Rebuild the counts block from its serialized form.

    Args:
        data: The ``counts`` value of a stored record.

    Returns:
        The reconstructed counts (zeros on junk).
    """
    return ActivityCounts(
        assistant_msgs=as_int(dict_get(data, "assistant_msgs")),
        external_prompts=as_int(dict_get(data, "external_prompts")),
        subagent_count=as_int(dict_get(data, "subagent_count")),
        compaction_count=as_int(dict_get(data, "compaction_count")),
        retry_prompt_count=as_int(dict_get(data, "retry_prompt_count")),
        malformed_lines=as_int(dict_get(data, "malformed_lines")),
    )


def _tokens_from_dict(data: object) -> TokenTotals:
    """Rebuild token totals from their serialized form.

    Args:
        data: The ``tokens`` value of a stored record.

    Returns:
        The reconstructed totals (zeros on junk).
    """
    return TokenTotals(
        output=as_int(dict_get(data, "output")),
        input=as_int(dict_get(data, "input")),
        cache_read=as_int(dict_get(data, "cache_read")),
        cache_write=as_int(dict_get(data, "cache_write")),
    )


def _turns_from_dict(data: object) -> TurnEfficiency:
    """Rebuild turn-efficiency counts from their serialized form.

    Args:
        data: The ``turns`` value of a stored record.

    Returns:
        The reconstructed counts (zeros on junk or on pre-v3 records).
    """
    return TurnEfficiency(
        api_turns=as_int(dict_get(data, "api_turns")),
        tool_calls=as_int(dict_get(data, "tool_calls")),
        batched_turns=as_int(dict_get(data, "batched_turns")),
        poll_turns=as_int(dict_get(data, "poll_turns")),
        lint_test_turns=as_int(dict_get(data, "lint_test_turns")),
        vcs_check_turns=as_int(dict_get(data, "vcs_check_turns")),
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
            name=str(dict_get(raw, "name") or "?"),
            tool_only_turns=as_int(dict_get(raw, "tool_only_turns")),
            text_turns=as_int(dict_get(raw, "text_turns")),
            tokens_output=as_int(dict_get(raw, "tokens_output")),
            tokens_cache_write=as_int(dict_get(raw, "tokens_cache_write")),
            bash_sig=str_list(dict_get(raw, "bash_sig")),
        )
        for raw in list_of_dicts(dict_get(data, "skill_invocations"))
    ]
    return Observations(
        tool_histogram=str_int_map(dict_get(data, "tool_histogram")),
        bash_first_lines=str_int_map(dict_get(data, "bash_first_lines")),
        bash_sequences=[
            str_list(seq) for seq in list_value(dict_get(data, "bash_sequences"))
        ],
        skill_invocations=invocations,
        error_signatures=str_int_map(dict_get(data, "error_signatures")),
        typed_prompt_signatures=str_int_map(dict_get(data, "typed_prompt_signatures")),
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
        file_size=as_int(data.get("file_size")),
        file_mtime=as_float(data.get("file_mtime")),
        analyzed_at=str(data.get("analyzed_at") or ""),
        schema_version=as_int(data.get("schema_version")),
        meta=_meta_from_dict(data.get("meta")),
        counts=_counts_from_dict(data.get("counts")),
        tokens=_tokens_from_dict(data.get("tokens")),
        turns=_turns_from_dict(data.get("turns")),
        obs=_obs_from_dict(data.get("obs")),
    )
