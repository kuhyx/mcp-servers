#!/bin/bash

# ============================================================================
# setup.sh — reconstruct the MCP server fleet from servers.json.
#
# Clones every server at its pinned commit into servers/. Existing clones are
# fetched and checked out to the pin, so re-running is safe and idempotent.
#
# The clones are deliberately NOT tracked by this repo: 24 upstream repos with
# their own .git plus ~10GB of regenerable build output. servers.json is the
# artifact worth keeping — ~50KB that restores 12GB reproducibly.
#
#   ./setup.sh              # all servers
#   ./setup.sh vestige      # one server
#   ./setup.sh --pin        # rewrite servers.json from the working clones
# ============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly MANIFEST="${SCRIPT_DIR}/servers.json"
readonly SERVERS="${SCRIPT_DIR}/servers"

need() { command -v "$1" >/dev/null || { echo "Error: $1 is required" >&2; exit 1; }; }

# Rewrite the manifest from whatever the clones currently point at.
repin() {
    python3 - "$SERVERS" "$MANIFEST" <<'PY'
import json, pathlib, subprocess, sys
root, manifest = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
out = {}
for d in sorted(p for p in root.iterdir() if (p / ".git").exists()):
    g = lambda *a: subprocess.run(["git", "-C", str(d), *a], capture_output=True,
                                  text=True, timeout=20).stdout.strip()
    out[d.name] = {"upstream": g("remote", "get-url", "origin"),
                   "commit": g("rev-parse", "HEAD"),
                   "branch": g("rev-parse", "--abbrev-ref", "HEAD")}
manifest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(f"re-pinned {len(out)} servers")
PY
}

clone_one() {
    local name="$1" upstream="$2" commit="$3" dest="${SERVERS}/$1"
    if [[ -d "$dest/.git" ]]; then
        git -C "$dest" fetch --quiet origin 2>/dev/null
    else
        echo "  cloning $name"
        git clone --quiet "$upstream" "$dest" || { echo "  FAILED: $name" >&2; return 1; }
    fi
    git -C "$dest" checkout --quiet "$commit" 2>/dev/null \
        || { echo "  WARN: $name has no commit $commit (upstream rewrote history?)" >&2; return 1; }
    echo "  ok $name @ ${commit:0:8}"
}

main() {
    need git; need python3
    [[ -f "$MANIFEST" ]] || { echo "Error: $MANIFEST missing" >&2; exit 1; }

    if [[ "${1:-}" == "--pin" ]]; then repin; exit $?; fi

    mkdir -p "$SERVERS"
    local only="${1:-}" failed=0
    while IFS=$'\t' read -r name upstream commit; do
        [[ -n "$only" && "$name" != "$only" ]] && continue
        clone_one "$name" "$upstream" "$commit" || failed=$((failed + 1))
    done < <(python3 -c "
import json,sys
for n, v in sorted(json.load(open('$MANIFEST')).items()):
    print(n, v['upstream'], v['commit'], sep='\t')
")
    echo "done ($failed failed)"
    # Build steps are per-server and deliberately not automated here: they need
    # npm/cargo/uv and differ per project. See each server's own README.
    return $((failed > 0))
}

main "$@"
