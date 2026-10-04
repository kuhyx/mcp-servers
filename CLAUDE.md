## Commands

- run: n/a: manifest + setup script (`./setup.sh`) and a hook-driven analyzer; no single entry point
- test: `PYTHONPATH=session-autopsy python3 -m pytest -q -c session-autopsy/pyproject.toml session-autopsy/session_autopsy/tests`
- test-changed: `scripts/test_changed.sh`
- lint: `ruff check session-autopsy && shellcheck setup.sh scripts/*.sh`
- coverage: `PYTHONPATH=session-autopsy python3 -m pytest -q -c session-autopsy/pyproject.toml --cov=session_autopsy --cov-branch --cov-report=xml:coverage.xml session-autopsy/session_autopsy/tests`
- coverage-gaps: `coverage-gaps coverage.xml`

Only `servers.json`, `setup.sh`, `scripts/`, `session-autopsy/` are tracked; `servers/` is untracked upstream clones (not ours, no tests here).
