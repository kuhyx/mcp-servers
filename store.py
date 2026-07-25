"""Flock-guarded persistence for sessions.jsonl.

The store is small (~1 record per session, hundreds total), so every write
is a full read-modify-replace under an exclusive lock; this keeps concurrent
SessionEnd hook firings and a parallel ``scan`` safe without a database.
"""

from __future__ import annotations

import fcntl
import json
from typing import TYPE_CHECKING

from typing_extensions import Self

from python_pkg.session_autopsy.records import SessionRecord, record_from_dict

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

SESSIONS_FILE = "sessions.jsonl"
LOCK_FILE = ".lock"


class StoreLock:
    """Context manager holding the store's exclusive flock."""

    def __init__(self, home: Path) -> None:
        """Remember the store directory.

        Args:
            home: The autopsy home directory.
        """
        self._lock_path = home / LOCK_FILE
        self._handle = None

    def __enter__(self) -> Self:
        """Create the home dir if needed and take the exclusive lock.

        Returns:
            Self, with the lock held.
        """
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._lock_path.open("a")
        fcntl.flock(self._handle, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Release the lock and close the handle.

        Args:
            exc_info: Ignored exception triple.
        """
        if self._handle is not None:
            fcntl.flock(self._handle, fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


def _iter_stored(home: Path) -> Iterator[SessionRecord]:
    """Yield every record currently in sessions.jsonl.

    Args:
        home: The autopsy home directory.

    Yields:
        Stored records; malformed lines are skipped silently.
    """
    sessions_path = home / SESSIONS_FILE
    if not sessions_path.is_file():
        return
    with sessions_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                yield record_from_dict(data)


def load_records(home: Path) -> list[SessionRecord]:
    """Load all stored records under the lock.

    Args:
        home: The autopsy home directory.

    Returns:
        All stored records, in file order.
    """
    with StoreLock(home):
        return list(_iter_stored(home))


def load_file_index(home: Path) -> dict[str, tuple[int, float]]:
    """Map transcript paths to their (size, mtime) at last analysis.

    Used by ``scan`` to skip transcripts that have not changed.

    Args:
        home: The autopsy home directory.

    Returns:
        ``{transcript_path: (file_size, file_mtime)}`` for every record.
    """
    with StoreLock(home):
        return {
            record.transcript_path: (record.file_size, record.file_mtime)
            for record in _iter_stored(home)
        }


def upsert_records(home: Path, new_records: list[SessionRecord]) -> int:
    """Insert or replace records by session id, atomically.

    Args:
        home: The autopsy home directory.
        new_records: Records to insert (replacing same-session entries).

    Returns:
        The total number of stored records after the write.
    """
    with StoreLock(home):
        merged = {record.session_id: record for record in _iter_stored(home)}
        for record in new_records:
            merged[record.session_id] = record
        tmp_path = home / (SESSIONS_FILE + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for record in merged.values():
                handle.write(json.dumps(record.to_dict(), separators=(",", ":")) + "\n")
        tmp_path.replace(home / SESSIONS_FILE)
        return len(merged)
