# mcp-servers

The MCP server fleet: 24 third-party servers, moved out of the `testsAndMisc`
monorepo where 12 GB of clones and build output did not belong.

## What is tracked, and what is not

**Tracked (~50 KB):** `servers.json` — every server's upstream URL and the exact
commit in use — plus `setup.sh`, which reconstructs the fleet from it.

**Not tracked:** `servers/` itself. Those are 24 upstream repositories with their own
`.git` directories plus ~10 GB of regenerable build output (`node_modules`, Rust
`target/`, venvs). Committing them would produce broken gitlinks and vendor artifacts
that rebuild anyway — and it would not make anything more recoverable. A pinned
manifest restores the fleet; a stale 12 GB fork does not.

## Use

```bash
./setup.sh              # clone/checkout every server at its pinned commit
./setup.sh vestige      # just one
./setup.sh --pin        # rewrite servers.json from the working clones
```

`setup.sh` is idempotent: existing clones are fetched and checked out to the pin.
Build steps are per-server (npm / cargo / uv) and deliberately not automated — see
each server's own README.

## Registration

Servers are registered in `~/.claude.json` (always-on core) and
`~/.claude/mcp-optional/*.json` (on-demand, loaded via `claude --mcp-config`).
Those configs reference absolute paths under `servers/`, so **re-run `./setup.sh`
before expecting a fresh checkout to work**, and rebuild any server whose binary
lives under `target/`.
