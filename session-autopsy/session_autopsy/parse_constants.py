"""Tunables and lookup tables for the transcript parser."""

from __future__ import annotations

import re

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

ERROR_LINE_RE = re.compile(r"^(?:rm: |fatal: |Traceback)|\w*error\b", re.IGNORECASE)
COMPACT_PREFIX = "This session is being continued from a previous conversation"
SLASH_COMMAND_RE = re.compile(r"<command-name>/?([\w:-]+)</command-name>")
# Harness built-ins that appear as <command-name> but are not skills.
BUILTIN_COMMANDS = frozenset(
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
GENERATIVE_INPUT_KEYS = ("content", "new_string")
# Error lines too generic to act on; the informative line is elsewhere.
GENERIC_ERROR_SIGS = frozenset(
    {
        "Error: Exit code <N>",
        "Traceback (most recent call last):",
    },
)
