"""Classify assistant turns by whether a model was needed to take them.

Cache-read tokens dominate the cost of a long session — one measured session spent
235M cache-read against 800k output, a ratio of roughly 290:1. Cache read is paid per
API round-trip on the whole conversation so far, so the lever is the NUMBER OF TURNS,
not the length of any message. A short turn taken late is as expensive as a long one.

That makes it worth counting the turns where the model decided nothing: polling a
counter, running a linter, checking whether a push landed. Those are pass/fail
mechanisms that code can adjudicate, which is exactly what belongs in a script rather
than in a round-trip (see ``rules/token-spend.instructions.md``).

Classification is deliberately conservative: a command is only called mechanical when
its whole purpose is observation. Anything that edits, fetches, builds or decides
counts as substantive, so the reported waste is a floor rather than a guess.
"""

from __future__ import annotations

import re

POLL = "poll"
LINT_TEST = "lint_test"
VCS_CHECK = "vcs_check"
SUBSTANTIVE = "substantive"

MECHANICAL = frozenset({POLL, LINT_TEST, VCS_CHECK})

# Ordered: the first pattern to match wins, so put the narrow ones first.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Waiting on something: process liveness, log tails, progress counters, clocks.
    (
        POLL,
        re.compile(
            r"""(?x)
            \bkill\s+-0\b            # is that pid still alive
          | \bpgrep\b | \bpidof\b
          | \bps\s+-[eo]            # ps -eo / ps -o : process listing
          | \bnvidia-smi\b
          | \bwc\s+-l\s*<           # counting lines in a log
          | \btail\s+(-[nf]?\d*\s+)?\S*\.log
          | ^\s*sleep\s
          | \buntil\s+.*\bdo\b.*\bdone\b
          | \bdate\s+['\"]?\+
            """
        ),
    ),
    # Adjudicated by an exit code, never by a model.
    (
        LINT_TEST,
        re.compile(
            r"\b(pytest|ruff|mypy|shellcheck|pre-commit\s+run|npm\s+test|cargo\s+test)\b"
        ),
    ),
    # Read-only VCS inspection. `git commit`/`push`/`add` are excluded on purpose:
    # they change state, so they are substantive even though they are scriptable.
    (
        VCS_CHECK,
        re.compile(
            r"""(?x)
            \bgit\s+(status|log|diff|ls-remote|rev-parse|remote\s+-v|branch|show)\b
          | \bgh\s+(api|repo\s+view|pr\s+view)\b
            """
        ),
    ),
)


def classify_command(command: str) -> str:
    """Classify one Bash command.

    Args:
        command: The raw Bash tool input (may be multi-line).

    Returns:
        One of :data:`POLL`, :data:`LINT_TEST`, :data:`VCS_CHECK` or
        :data:`SUBSTANTIVE`.
    """
    text = command.strip()
    if not text:
        return SUBSTANTIVE
    for label, pattern in _PATTERNS:
        if pattern.search(text):
            return label
    return SUBSTANTIVE


def classify_turn(commands: list[str]) -> str:
    """Classify a whole turn from the Bash commands it issued.

    A turn only counts as mechanical when *every* command in it is — one real piece
    of work makes the round-trip necessary regardless of what else it carried, and
    over-counting waste would make the report untrustworthy.

    Args:
        commands: Every Bash command issued in this turn, in order.

    Returns:
        The shared mechanical label, or :data:`SUBSTANTIVE` if the turn mixed kinds,
        carried real work, or ran no Bash at all.
    """
    if not commands:
        return SUBSTANTIVE
    labels = {classify_command(c) for c in commands}
    if len(labels) == 1:
        only = labels.pop()
        if only in MECHANICAL:
            return only
    return SUBSTANTIVE
