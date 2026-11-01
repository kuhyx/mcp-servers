"""Output and clock seams shared by the CLI commands."""

from __future__ import annotations

import sys
from datetime import UTC, datetime


def emit(message: str) -> None:
    """Write a line to stdout.

    Args:
        message: Line to write, without a trailing newline.
    """
    sys.stdout.write(f"{message}\n")


def emit_err(message: str) -> None:
    """Write a line to stderr.

    Args:
        message: Line to write, without a trailing newline.
    """
    sys.stderr.write(f"{message}\n")


def now() -> datetime:
    """Current UTC time (single seam for tests).

    Returns:
        Timezone-aware now.
    """
    return datetime.now(tz=UTC)
