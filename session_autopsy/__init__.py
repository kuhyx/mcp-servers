"""Deterministic Claude Code transcript analyzer (zero LLM tokens).

Parses ``~/.claude/projects/**/*.jsonl`` session transcripts into compact
per-session records, then mines them for automation candidates: skills worth
"compiling" into deterministic scripts, repeated command sequences, repeated
errors worth a fix-once script, and repeated typed prompts.
"""
