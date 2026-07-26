"""Turn-efficiency accounting for a single transcript.

Split out of ``parse.py``: the tracker is self-contained — it knows only about
turn totals and the classification of the commands in a turn — and lifting it
out keeps the parser inside the repo's 500-line-per-file limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from python_pkg.session_autopsy.records import TurnEfficiency
from python_pkg.session_autopsy.turns import LINT_TEST, POLL, VCS_CHECK, classify_turn


@dataclass
class OpenTurn:
    """The API turn currently being accumulated, keyed by requestId."""

    request_id: str | None = None
    tools: int = 0
    commands: list[str] = field(default_factory=list)


class TurnTracker:
    """Running turn-efficiency totals plus the turn currently open.

    The two belong together: the open turn only exists to be folded into the
    totals, and keeping them as one object is also what keeps the parser's
    accumulator inside its instance-attribute budget.
    """

    def __init__(self) -> None:
        """Start with empty totals and no open turn."""
        self.totals = TurnEfficiency()
        self._open = OpenTurn()

    def begin(self, request_id: object) -> None:
        """Open a new turn if this line belongs to a different API response.

        Args:
            request_id: The line's ``requestId`` (or message-id fallback).
        """
        if request_id != self._open.request_id:
            self.flush()
            self._open.request_id = request_id if isinstance(request_id, str) else None

    def add_tool(self) -> None:
        """Count one tool_use block in the open turn."""
        self._open.tools += 1

    def add_command(self, command: str) -> None:
        """Record one Bash command issued in the open turn.

        Args:
            command: The raw Bash tool input.
        """
        self._open.commands.append(command)

    def flush(self) -> None:
        """Fold the open turn into the totals and start a fresh one."""
        open_turn = self._open
        self._open = OpenTurn()
        if open_turn.tools <= 0:
            return
        self.totals.api_turns += 1
        self.totals.tool_calls += open_turn.tools
        if open_turn.tools > 1:
            self.totals.batched_turns += 1
        label = classify_turn(open_turn.commands)
        if label == POLL:
            self.totals.poll_turns += 1
        elif label == LINT_TEST:
            self.totals.lint_test_turns += 1
        elif label == VCS_CHECK:
            self.totals.vcs_check_turns += 1
