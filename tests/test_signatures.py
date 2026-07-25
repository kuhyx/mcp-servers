"""Tests for the cross-session text signatures."""

from __future__ import annotations

from python_pkg.session_autopsy import signatures
from python_pkg.session_autopsy.signatures import command_signature, normalize_signature


def test_normalize_signature() -> None:
    """Paths, hex runs, and numbers are masked; first line only."""
    assert normalize_signature("") == ""
    assert normalize_signature("   \n\n") == ""
    sig = normalize_signature("rm -rf /home/kuhy/.cache/yay/python-jaxlib\nsecond line")
    assert sig == "rm -rf <PATH>"
    assert normalize_signature("deadbeefcafe1234") == "<HEX>"
    assert normalize_signature("retry 12 times") == "retry <N> times"
    assert len(normalize_signature("x" * 500)) == signatures.MAX_SIG_LEN


def test_command_signature() -> None:
    """Wrappers keep their first non-flag, non-numeric argument."""
    assert command_signature("") == ""
    assert command_signature("git status") == "git"
    assert command_signature("cd /home/kuhy/testsAndMisc && ls") == "cd <PATH>"
    assert command_signature("timeout 120 python3 -m x") == "timeout python3"
    assert command_signature("sudo -n true") == "sudo true"
    assert command_signature("sudo -n") == "sudo"
    assert command_signature("cd") == "cd"
