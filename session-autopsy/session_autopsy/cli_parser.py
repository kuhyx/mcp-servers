"""Argument parser for the session-autopsy command line."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_SCAN_JOBS = 4
DEFAULT_TRACE_INVOCATIONS = 5


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        Parser with all subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="python -m session_autopsy",
        description="Deterministic Claude Code transcript analyzer.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="analyze one session transcript")
    ingest.add_argument(
        "transcript", type=Path, help="path to the <uuid>.jsonl transcript"
    )
    ingest.add_argument(
        "--quiet", action="store_true", help="suppress the summary line"
    )
    ingest.add_argument(
        "--no-report", action="store_true", help="skip REPORT.md regeneration"
    )

    scan = subparsers.add_parser(
        "scan", help="analyze every transcript under the projects dir"
    )
    scan.add_argument(
        "--all",
        action="store_true",
        help="accepted for symmetry; scan always walks everything",
    )
    scan.add_argument(
        "--jobs", type=int, default=DEFAULT_SCAN_JOBS, help="parallel worker processes"
    )
    scan.add_argument(
        "--force", action="store_true", help="reanalyze transcripts even if unchanged"
    )
    scan.add_argument(
        "--no-report", action="store_true", help="skip REPORT.md regeneration"
    )

    report_cmd = subparsers.add_parser(
        "report", help="regenerate REPORT.md from stored records"
    )
    report_cmd.add_argument(
        "--mark-reviewed",
        action="store_true",
        help="mark all current candidates reviewed",
    )

    candidates = subparsers.add_parser("candidates", help="list ranked candidates")
    candidates.add_argument(
        "--json", action="store_true", help="machine-readable output"
    )

    trace_cmd = subparsers.add_parser(
        "traces", help="export compact excerpts for a candidate"
    )
    trace_cmd.add_argument(
        "candidate_id", help="candidate id from REPORT.md (e.g. skill-finish)"
    )
    trace_cmd.add_argument(
        "--max-invocations", type=int, default=DEFAULT_TRACE_INVOCATIONS
    )
    trace_cmd.add_argument(
        "--out", type=Path, default=None, help="write to a file instead of stdout"
    )

    measure = subparsers.add_parser(
        "measure", help="before/after tokens for compiled skills"
    )
    measure.add_argument(
        "skill", nargs="?", default=None, help="restrict to one skill name"
    )
    return parser
