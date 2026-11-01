"""Open Bash-run and skill-span state for the transcript parser."""

from __future__ import annotations

from session_autopsy.parse_constants import (
    MAX_BASH_RUN,
    MAX_SKILL_BASH_SIG,
    MIN_BASH_RUN,
)
from session_autopsy.records import SkillInvocation


class SpanTracker:
    """Tracks the current Bash run and skill span, and what they closed into."""

    def __init__(self) -> None:
        """Initialize with nothing open and nothing collected."""
        self.bash_sequences: list[list[str]] = []
        self.skill_invocations: list[SkillInvocation] = []
        self.current_run: list[str] = []
        self.current_skill: SkillInvocation | None = None

    def add_signature(self, sig: str) -> None:
        """Attach a Bash command signature to the open skill, else the open run.

        Args:
            sig: The command's normalized signature.
        """
        if self.current_skill is not None:
            if len(self.current_skill.bash_sig) < MAX_SKILL_BASH_SIG:
                self.current_skill.bash_sig.append(sig)
        elif len(self.current_run) < MAX_BASH_RUN:
            self.current_run.append(sig)

    def flush_run(self) -> None:
        """Close the current Bash run, keeping it only if long enough."""
        if len(self.current_run) >= MIN_BASH_RUN:
            self.bash_sequences.append(self.current_run)
        self.current_run = []

    def flush_skill(self) -> None:
        """Close the current skill span, dropping spans with no turns."""
        skill = self.current_skill
        if skill is not None:
            if skill.tool_only_turns + skill.text_turns > 0:
                self.skill_invocations.append(skill)
            self.current_skill = None

    def flush_all(self) -> None:
        """Close both the current Bash run and the current skill span."""
        self.flush_run()
        self.flush_skill()
