"""Tests for the CLI, config env overrides, and the module entry point."""

from __future__ import annotations

import json
import runpy
import sys
from typing import TYPE_CHECKING

import pytest

from python_pkg.session_autopsy import cli, config, report, store
from python_pkg.session_autopsy.records import Observations
from python_pkg.session_autopsy.tests.conftest import (
    FileFacts,
    assistant_line,
    bash_block,
    invocation,
    prompt_line,
    record,
    skill_block,
    text_block,
    write_transcript,
)

if TYPE_CHECKING:
    from pathlib import Path

    from python_pkg.session_autopsy.tests.conftest import AutopsyEnv


def test_config_env_overrides(autopsy_env: AutopsyEnv) -> None:
    """Both locations follow their environment variables."""
    assert config.autopsy_home() == autopsy_env.home
    assert config.projects_dir() == autopsy_env.projects


def test_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without overrides, both locations live under ~/.claude."""
    monkeypatch.delenv("CLAUDE_AUTOPSY_HOME", raising=False)
    monkeypatch.delenv("CLAUDE_PROJECTS_DIR", raising=False)
    assert config.autopsy_home().name == "autopsy"
    assert config.projects_dir().name == "projects"


def _seed_transcript(env: AutopsyEnv, session_id: str = "sess-1") -> Path:
    """Write a small but representative transcript into the projects dir."""
    return write_transcript(
        env.projects,
        session_id,
        [
            assistant_line(skill_block("finish")),
            assistant_line(bash_block("git status"), text_block("checking")),
            prompt_line("wrap it up please"),
        ],
    )


def test_ingest_missing_file(
    autopsy_env: AutopsyEnv, capsys: pytest.CaptureFixture[str]
) -> None:
    """A nonexistent transcript exits 2 with a stderr message."""
    assert (
        cli.main(["ingest", str(autopsy_env.projects / "nope.jsonl")])
        == cli.EXIT_NO_FILE
    )
    assert "no such transcript" in capsys.readouterr().err


def test_ingest_and_report_regeneration(
    autopsy_env: AutopsyEnv, capsys: pytest.CaptureFixture[str]
) -> None:
    """ingest upserts, prints a summary, and regenerates REPORT.md."""
    transcript = _seed_transcript(autopsy_env)
    assert cli.main(["ingest", str(transcript), "--no-report"]) == 0
    out = capsys.readouterr().out
    assert "sess-1" in out
    assert not (autopsy_env.home / report.REPORT_FILE).exists()
    assert cli.main(["ingest", str(transcript), "--quiet"]) == 0
    assert capsys.readouterr().out == ""
    assert (autopsy_env.home / report.REPORT_FILE).is_file()


def test_scan_skip_force_and_nonfile(
    autopsy_env: AutopsyEnv, capsys: pytest.CaptureFixture[str]
) -> None:
    """scan analyzes changed transcripts, skips unchanged, honors --force."""
    _seed_transcript(autopsy_env, "s1")
    _seed_transcript(autopsy_env, "s2")
    (autopsy_env.projects / "-home-kuhy" / "dir.jsonl").mkdir()
    assert cli.main(["scan", "--all", "--jobs", "0", "--no-report"]) == 0
    assert "2 analyzed, 0 unchanged" in capsys.readouterr().out
    assert cli.main(["scan", "--no-report"]) == 0
    assert "0 analyzed, 2 unchanged" in capsys.readouterr().out
    assert cli.main(["scan", "--force"]) == 0
    assert "2 analyzed" in capsys.readouterr().out


def test_needs_analysis_on_size_change(autopsy_env: AutopsyEnv) -> None:
    """A stored (size, mtime) mismatch marks the transcript changed."""
    transcript = _seed_transcript(autopsy_env)
    index = {str(transcript): (transcript.stat().st_size, transcript.stat().st_mtime)}
    assert not cli._needs_analysis(transcript, index)
    index[str(transcript)] = (1, 1.0)
    assert cli._needs_analysis(transcript, index)


def test_parse_one_swallows_oserror(
    autopsy_env: AutopsyEnv, capsys: pytest.CaptureFixture[str]
) -> None:
    """A vanished transcript logs to stderr and yields None."""
    assert cli._parse_one(autopsy_env.projects / "gone.jsonl") is None
    assert "skipping" in capsys.readouterr().err


def test_regenerate_report_with_preloaded_records(autopsy_env: AutopsyEnv) -> None:
    """The records argument skips the store read."""
    assert cli._regenerate_report([]) == 0


def _seed_store(env: AutopsyEnv) -> None:
    """Store records that produce one deterministic skill candidate."""
    transcript = _seed_transcript(env, "t1")
    store.upsert_records(
        env.home,
        [
            record(
                "t1",
                file=FileFacts(path=str(transcript)),
                obs=Observations(skill_invocations=[invocation()]),
            ),
            record("t2", obs=Observations(skill_invocations=[invocation()])),
            record("t3", obs=Observations(skill_invocations=[invocation()])),
        ],
    )


def test_report_and_mark_reviewed(
    autopsy_env: AutopsyEnv, capsys: pytest.CaptureFixture[str]
) -> None:
    """report writes the file; --mark-reviewed zeroes the badge count."""
    _seed_store(autopsy_env)
    assert cli.main(["report"]) == 0
    assert "candidates" in capsys.readouterr().out
    assert cli.main(["report", "--mark-reviewed"]) == 0
    assert "marked reviewed" in capsys.readouterr().out
    state = json.loads(
        (autopsy_env.home / report.STATE_FILE).read_text(encoding="utf-8")
    )
    assert state["unreviewed_count"] == 0


def test_candidates_json_and_plain(
    autopsy_env: AutopsyEnv, capsys: pytest.CaptureFixture[str]
) -> None:
    """candidates renders both machine and human output."""
    _seed_store(autopsy_env)
    assert cli.main(["candidates", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["id"] == "skill-finish"
    assert cli.main(["candidates"]) == 0
    assert "skill-finish" in capsys.readouterr().out


def test_traces_unknown_and_output_modes(
    autopsy_env: AutopsyEnv, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """traces exits 3 on unknown ids and writes --out files."""
    _seed_store(autopsy_env)
    assert cli.main(["traces", "nope-123"]) == cli.EXIT_UNKNOWN_ID
    assert "unknown candidate id" in capsys.readouterr().err
    out_file = tmp_path / "traces.txt"
    assert cli.main(["traces", "skill-finish", "--out", str(out_file)]) == 0
    assert "TOOL Skill | finish" in out_file.read_text(encoding="utf-8")
    assert cli.main(["traces", "skill-finish", "--max-invocations", "1"]) == 0
    assert "traces skill-finish" in capsys.readouterr().out


def test_measure_command(
    autopsy_env: AutopsyEnv, capsys: pytest.CaptureFixture[str]
) -> None:
    """measure prints the placeholder when nothing is compiled."""
    assert cli.main(["measure"]) == 0
    assert "nothing compiled yet" in capsys.readouterr().out


def test_main_module_entry(
    autopsy_env: AutopsyEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """python -m python_pkg.session_autopsy dispatches and exits."""
    monkeypatch.setattr(sys, "argv", ["session_autopsy", "measure"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("python_pkg.session_autopsy", run_name="__main__")
    assert excinfo.value.code == 0
