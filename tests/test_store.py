"""Tests for the flock-guarded sessions.jsonl store."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from python_pkg.session_autopsy.records import ActivityCounts
from python_pkg.session_autopsy.store import (
    SESSIONS_FILE,
    StoreLock,
    load_file_index,
    load_records,
    upsert_records,
)
from python_pkg.session_autopsy.tests.conftest import (
    FileFacts,
    record,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_lock_exit_without_enter(tmp_path: Path) -> None:
    """__exit__ with no held handle is a safe no-op."""
    StoreLock(tmp_path).__exit__(None, None, None)


def test_load_records_missing_store(tmp_path: Path) -> None:
    """An absent sessions.jsonl reads as empty."""
    assert load_records(tmp_path / "home") == []


def test_iter_skips_bad_lines(tmp_path: Path) -> None:
    """Blank, malformed, and non-dict store lines are skipped."""
    home = tmp_path
    good = record("keep")
    (home / SESSIONS_FILE).write_text(
        "\n".join(["", "{bad json", json.dumps([1, 2]), json.dumps(good.to_dict())])
        + "\n",
        encoding="utf-8",
    )
    records = load_records(home)
    assert [rec.session_id for rec in records] == ["keep"]


def test_upsert_inserts_and_replaces(tmp_path: Path) -> None:
    """Upsert keys by session id: new ids append, same ids replace."""
    home = tmp_path / "home"
    assert upsert_records(home, [record("a"), record("b")]) == 2
    replacement = record("a", counts=ActivityCounts(assistant_msgs=99))
    assert upsert_records(home, [replacement]) == 2
    by_id = {rec.session_id: rec for rec in load_records(home)}
    assert by_id["a"].counts.assistant_msgs == 99
    assert by_id["b"].counts.assistant_msgs == 10


def test_load_file_index(tmp_path: Path) -> None:
    """The index maps transcript paths to (size, mtime)."""
    home = tmp_path / "home"
    upsert_records(
        home, [record("a", file=FileFacts(path="/t/a.jsonl", size=5, mtime=2.5))]
    )
    assert load_file_index(home) == {"/t/a.jsonl": (5, 2.5)}
