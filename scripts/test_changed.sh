#!/bin/bash

# ============================================================================
# Run only the session-autopsy tests related to files changed vs HEAD (staged,
# unstaged, untracked). Quiet: failures plus a one-line summary. Falls back to
# the full suite when a change cannot be mapped. servers/ is untracked third-
# party code and has no tests owned here.
# ============================================================================

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
readonly PKG=session-autopsy
export PYTHONPATH="$PKG"
pytest_run() { python3 -m pytest -q --tb=short -c "$PKG/pyproject.toml" "$@"; }

CHANGED=()
while IFS= read -r f; do
    [[ -n "$f" && -e "$f" ]] && CHANGED+=("$f")
done < <({ git diff --name-only HEAD 2>/dev/null || true; git ls-files --others --exclude-standard; } | sort -u)

full=0
tests=()
for f in "${CHANGED[@]}"; do
    case "$f" in
        "$PKG"/pyproject.toml | */conftest.py) full=1 ;;
        "$PKG"/*.py)
            base="$(basename "$f" .py)"
            if [[ "$base" == test_* ]]; then tests+=("$f")
            else
                while IFS= read -r t; do tests+=("$t"); done < <(git ls-files "$PKG/*test_${base}.py")
            fi
            ;;
    esac
done

if [[ ${#CHANGED[@]} -eq 0 ]]; then echo "no changes vs HEAD: nothing to test"; exit 0; fi
if [[ $full -eq 0 && ${#tests[@]} -gt 0 ]]; then pytest_run "${tests[@]}"; exit $?; fi
if [[ $full -eq 0 ]] && ! printf '%s\n' "${CHANGED[@]}" | grep -q "^$PKG/.*\.py$"; then
    echo "no session-autopsy python changes: nothing to test"; exit 0
fi
echo "no mapped tests: running full suite"
pytest_run "$PKG/session_autopsy/tests"
