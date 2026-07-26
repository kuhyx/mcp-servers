"""Tests for classifying turns a script could have taken instead of the model."""

from __future__ import annotations

import pytest

from python_pkg.session_autopsy.turns import (
    LINT_TEST,
    POLL,
    SUBSTANTIVE,
    VCS_CHECK,
    classify_command,
    classify_turn,
)


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # Waiting on something.
        ("kill -0 262023 2>/dev/null && echo ALIVE", POLL),
        ("pgrep -af claude | head", POLL),
        ("ps -eo pid,pgid,stat,args | grep python", POLL),
        ("wc -l < /tmp/scrape.log", POLL),
        ("tail -5 /tmp/scrape.log", POLL),
        ("nvidia-smi --query-gpu=utilization.gpu --format=csv", POLL),
        ("sleep 30", POLL),
        ("until grep -q DONE log; do sleep 5; done", POLL),
        ("date '+%H:%M'", POLL),
        # Adjudicated by an exit code.
        ("pytest -q --cov=wikikb", LINT_TEST),
        (".venv/bin/python -m ruff check wikikb", LINT_TEST),
        ("pre-commit run --all-files", LINT_TEST),
        ("shellcheck scripts/safepush.sh", LINT_TEST),
        # Read-only VCS inspection.
        ("git status --short", VCS_CHECK),
        ("git ls-remote origin refs/heads/main", VCS_CHECK),
        ("git rev-parse HEAD", VCS_CHECK),
        ("gh api repos/kuhyx/wiki-kb --jq .visibility", VCS_CHECK),
        # Real work: anything that changes state, fetches, or builds.
        ("./run.sh --profile kcd index", SUBSTANTIVE),
        ("git commit -q -m 'thing'", SUBSTANTIVE),
        ("git push origin main", SUBSTANTIVE),
        ("curl -sS https://example.com/api.php", SUBSTANTIVE),
        ("", SUBSTANTIVE),
        ("   ", SUBSTANTIVE),
    ],
)
def test_classify_command(command: str, expected: str) -> None:
    assert classify_command(command) == expected


def test_commit_and_push_are_substantive_even_though_scriptable() -> None:
    # They change state, so a turn carrying them was not pure observation. Counting
    # them as waste would overstate the finding, and an overstated report is worse
    # than none.
    assert classify_command("git add -A && git commit -m x") == SUBSTANTIVE


def test_classify_turn_requires_every_command_to_be_mechanical() -> None:
    assert classify_turn(["git status", "git rev-parse HEAD"]) == VCS_CHECK
    assert classify_turn(["pytest -q"]) == LINT_TEST
    # Mixed kinds: still mechanical individually, but the turn is not one thing.
    assert classify_turn(["git status", "pytest -q"]) == SUBSTANTIVE
    # One real action makes the round-trip necessary regardless of what else it did.
    assert classify_turn(["wc -l < log", "./run.sh index"]) == SUBSTANTIVE


def test_classify_turn_with_no_bash_is_substantive() -> None:
    # A turn that ran Edit/Write/Read rather than Bash is not a polling turn.
    assert classify_turn([]) == SUBSTANTIVE
