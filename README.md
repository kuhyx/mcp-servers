# session-autopsy

Reads Claude Code session transcripts and tells you where the tokens went — with no
model in the loop. Pure Python, stdlib only, runs as a `SessionEnd` hook.

Two things it answers:

**What should be a script?** It mines repeated Bash sequences, error signatures and
prompt shapes across every stored session and ranks automation candidates by how much
they'd save.

**Which turns did nothing?** Cache-read tokens dominate a long session's cost — one
measured session spent **267M cache-read against 856k output**, a 290:1 ratio. Cache
read is charged per API round-trip on the whole conversation so far, so the lever is
the *number of turns*, not message length. `turns.py` classifies each turn as
`poll` / `lint_test` / `vcs_check` / `substantive` and prices the ones a script could
have taken.

Measured across 263 stored transcripts: **33,508 tool turns at an 8% batching rate, of
which 4,437 (13%) decided nothing — roughly 2.7B cache-read tokens.**

## Use

```bash
pip install -e .
session-autopsy scan          # ingest new transcripts
session-autopsy report        # write REPORT.md
```

As a hook (`~/.claude/settings.json`, `SessionEnd`):

```bash
session-autopsy ingest "$transcript" --quiet
```

## Design

Classification is deliberately conservative. `git commit` and `git push` count as
*substantive* even though they are scriptable, because they change state; and a turn
is only mechanical when **every** command in it is. The reported waste is a floor, not
a guess — an overstated report is worse than none.

Batching is grouped by `requestId`, not by transcript line: one API response can span
several lines, and counting per line reports every turn as single-tool (0% batching
where the real rate is 12%).

Extracted from the `testsAndMisc` monorepo with history intact. 100% branch coverage.
