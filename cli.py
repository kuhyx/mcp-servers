"""Command-line entry point for the session autopsy analyzer.

Usage, with ``PYTHONPATH=~/testsAndMisc`` and ``python3 -m
python_pkg.session_autopsy``:

    ingest <transcript.jsonl>
    scan --jobs 8
    report [--mark-reviewed]
    traces skill-finish
"""

from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING

from python_pkg.session_autopsy import config, detectors, report, store, traces
from python_pkg.session_autopsy.parse import parse_session

if TYPE_CHECKING:
    from python_pkg.session_autopsy.records import SessionRecord

DEFAULT_SCAN_JOBS = 4
DEFAULT_TRACE_INVOCATIONS = 5
EXIT_NO_FILE = 2
EXIT_UNKNOWN_ID = 3


def _emit(message: str) -> None:
    """Write a line to stdout.

    Args:
        message: Line to write, without a trailing newline.
    """
    sys.stdout.write(f"{message}\n")


def _emit_err(message: str) -> None:
    """Write a line to stderr.

    Args:
        message: Line to write, without a trailing newline.
    """
    sys.stderr.write(f"{message}\n")


def _now() -> datetime:
    """Current UTC time (single seam for tests).

    Returns:
        Timezone-aware now.
    """
    return datetime.now(tz=timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        Parser with all subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="python -m python_pkg.session_autopsy",
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


def _regenerate_report(records: list[SessionRecord] | None = None) -> int:
    """Re-run detectors and rewrite REPORT.md + state.json.

    Args:
        records: Preloaded records, or None to load from the store.

    Returns:
        The unreviewed-candidate count.
    """
    home = config.autopsy_home()
    if records is None:
        records = store.load_records(home)
    result = detectors.analyze(records, _now())
    return report.write_report(home, records, result, _now())


def _summary_line(record: SessionRecord) -> str:
    """Render the one-line ingest summary for a record.

    Args:
        record: The freshly parsed record.

    Returns:
        A compact human-readable summary.
    """
    return (
        f"{record.session_id}: {record.counts.assistant_msgs} assistant msgs, "
        f"{record.tokens.output} out / {record.tokens.cache_write} cache-write tokens, "
        f"{len(record.obs.skill_invocations)} skill spans, "
        f"{record.counts.subagent_count} subagents, "
        f"{record.counts.malformed_lines} malformed lines"
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    """Analyze one transcript, upsert its record, refresh the report.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code (0 on success, 2 on missing file).
    """
    transcript: Path = args.transcript.expanduser()
    if not transcript.is_file():
        _emit_err(f"ingest: no such transcript: {transcript}")
        return EXIT_NO_FILE
    record = parse_session(transcript)
    total = store.upsert_records(config.autopsy_home(), [record])
    if not args.no_report:
        _regenerate_report()
    if not args.quiet:
        _emit(_summary_line(record))
        _emit(f"store: {total} sessions")
    return 0


def _find_transcripts(root: Path) -> list[Path]:
    """List every top-level session transcript under the projects dir.

    Subagent transcripts (``**/subagents/agent-*.jsonl``) are excluded;
    they are folded into their parent session by the parser.

    Args:
        root: The projects directory.

    Returns:
        Sorted transcript paths.
    """
    return sorted(path for path in root.glob("*/*.jsonl") if path.is_file())


def _needs_analysis(path: Path, index: dict[str, tuple[int, float]]) -> bool:
    """Decide whether a transcript changed since its stored record.

    Args:
        path: Transcript path.
        index: Stored ``{path: (size, mtime)}`` map.

    Returns:
        True when the transcript is new or its size/mtime moved.
    """
    stored = index.get(str(path))
    if stored is None:
        return True
    stat = path.stat()
    return (stat.st_size, stat.st_mtime) != stored


def _parse_one(path: Path) -> SessionRecord | None:
    """Worker-side parse wrapper that never raises across the pool.

    Args:
        path: Transcript path to parse.

    Returns:
        The record, or None when the file vanished or was unreadable.
    """
    try:
        return parse_session(path)
    except OSError as error:
        _emit_err(f"scan: skipping {path}: {error}")
        return None


def cmd_scan(args: argparse.Namespace) -> int:
    """Analyze every changed transcript in parallel and upsert results.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code (always 0; unreadable files are reported and skipped).
    """
    started = time.monotonic()
    home = config.autopsy_home()
    root = config.projects_dir()
    transcripts = _find_transcripts(root)
    index = {} if args.force else store.load_file_index(home)
    pending = [path for path in transcripts if _needs_analysis(path, index)]
    records: list[SessionRecord] = []
    if pending:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, args.jobs)
        ) as pool:
            records = [
                record for record in pool.map(_parse_one, pending) if record is not None
            ]
        store.upsert_records(home, records)
    if not args.no_report:
        _regenerate_report()
    elapsed = time.monotonic() - started
    _emit(
        f"scan: {len(transcripts)} transcripts, {len(pending)} analyzed, "
        f"{len(transcripts) - len(pending)} unchanged, {elapsed:.1f}s",
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Regenerate REPORT.md (optionally marking candidates reviewed).

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code (always 0).
    """
    home = config.autopsy_home()
    records = store.load_records(home)
    result = detectors.analyze(records, _now())
    report.write_report(home, records, result, _now())
    if args.mark_reviewed:
        report.write_state(
            home, [cand.id for cand in result.candidates], _now(), mark_reviewed=True
        )
        _emit("report: all candidates marked reviewed")
    _emit(f"report: {len(result.candidates)} candidates → {home / report.REPORT_FILE}")
    return 0


def cmd_candidates(args: argparse.Namespace) -> int:
    """List ranked candidates.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code (always 0).
    """
    records = store.load_records(config.autopsy_home())
    result = detectors.analyze(records, _now())
    if args.json:
        _emit(json.dumps([cand.to_dict() for cand in result.candidates], indent=2))
        return 0
    for cand in result.candidates:
        savings = report.fmt_tokens(cand.est_weekly_savings)
        _emit(
            f"{cand.id}  {cand.kind:6}  {cand.occurrences:4}x  "
            f"~{savings}/wk  {cand.action}"
        )
    return 0


def cmd_traces(args: argparse.Namespace) -> int:
    """Export compact excerpts for one candidate.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code (0, or 3 for an unknown candidate id).
    """
    records = store.load_records(config.autopsy_home())
    result = detectors.analyze(records, _now())
    match = next(
        (cand for cand in result.candidates if cand.id == args.candidate_id), None
    )
    if match is None:
        _emit_err(f"traces: unknown candidate id: {args.candidate_id}")
        return EXIT_UNKNOWN_ID
    records_by_id = {record.session_id: record for record in records}
    text = traces.render_traces(
        match, records_by_id, max_invocations=args.max_invocations
    )
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
        _emit(f"traces: wrote {args.out}")
        return 0
    _emit(text)
    return 0


def cmd_measure(args: argparse.Namespace) -> int:
    """Report before/after tokens per invocation for compiled skills.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code (always 0).
    """
    home = config.autopsy_home()
    records = store.load_records(home)
    for line in report.measure_lines(records, report.load_compiled(home), args.skill):
        _emit(line)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument vector (None means ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    handlers = {
        "ingest": cmd_ingest,
        "scan": cmd_scan,
        "report": cmd_report,
        "candidates": cmd_candidates,
        "traces": cmd_traces,
        "measure": cmd_measure,
    }
    return handlers[args.command](args)
