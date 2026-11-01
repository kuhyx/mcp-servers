"""Command-line entry point for the session autopsy analyzer.

Usage, from ``~/src/mcp-servers/session-autopsy`` with ``PYTHONPATH=.`` and
``python3 -m session_autopsy``:

    ingest <transcript.jsonl>
    scan --jobs 8
    report [--mark-reviewed]
    traces skill-finish
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from session_autopsy import (
    cli_scan,
    config,
    detectors,
    report,
    report_measure,
    report_state,
    store,
    traces,
)
from session_autopsy.cli_io import emit, emit_err, now
from session_autopsy.cli_parser import build_parser
from session_autopsy.parse import parse_session

if TYPE_CHECKING:
    from session_autopsy.records import SessionRecord

EXIT_NO_FILE = 2
EXIT_UNKNOWN_ID = 3


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
    result = detectors.analyze(records, now())
    return report.write_report(home, records, result, now())


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
        emit_err(f"ingest: no such transcript: {transcript}")
        return EXIT_NO_FILE
    record = parse_session(transcript)
    total = store.upsert_records(config.autopsy_home(), [record])
    if not args.no_report:
        _regenerate_report()
    if not args.quiet:
        emit(_summary_line(record))
        emit(f"store: {total} sessions")
    return 0


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
    transcripts = cli_scan.find_transcripts(root)
    index = {} if args.force else store.load_file_index(home)
    pending = [path for path in transcripts if cli_scan.needs_analysis(path, index)]
    records: list[SessionRecord] = []
    if pending:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, args.jobs)
        ) as pool:
            records = [
                record
                for record in pool.map(cli_scan.parse_one, pending)
                if record is not None
            ]
        store.upsert_records(home, records)
    if not args.no_report:
        _regenerate_report()
    elapsed = time.monotonic() - started
    emit(
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
    result = detectors.analyze(records, now())
    report.write_report(home, records, result, now())
    if args.mark_reviewed:
        handled = report_state.handled_ids(report_state.load_compiled(home))
        pending = [
            cid
            for cid in report_state.significant_ids(result.candidates)
            if cid not in handled
        ]
        report_state.write_state(home, pending, now(), mark_reviewed=True)
        emit("report: all candidates marked reviewed")
    emit(f"report: {len(result.candidates)} candidates → {home / report.REPORT_FILE}")
    return 0


def cmd_candidates(args: argparse.Namespace) -> int:
    """List ranked candidates.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code (always 0).
    """
    records = store.load_records(config.autopsy_home())
    result = detectors.analyze(records, now())
    if args.json:
        emit(json.dumps([cand.to_dict() for cand in result.candidates], indent=2))
        return 0
    for cand in result.candidates:
        savings = report_measure.fmt_tokens(cand.est_weekly_savings)
        emit(
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
    result = detectors.analyze(records, now())
    match = next(
        (cand for cand in result.candidates if cand.id == args.candidate_id), None
    )
    if match is None:
        emit_err(f"traces: unknown candidate id: {args.candidate_id}")
        return EXIT_UNKNOWN_ID
    records_by_id = {record.session_id: record for record in records}
    text = traces.render_traces(
        match, records_by_id, max_invocations=args.max_invocations
    )
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
        emit(f"traces: wrote {args.out}")
        return 0
    emit(text)
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
    for line in report_measure.measure_lines(
        records, report_state.load_compiled(home), args.skill
    ):
        emit(line)
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
