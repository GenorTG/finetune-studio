<!-- Base branch: `dev` for work (<type>/<slug>), `main` only for release/vX.Y.Z or hotfix/vX.Y.Z. CONTRIBUTING.md -->

## What and why

## How it was verified
<!-- the test files run, ruff, and for GPU paths the real-hardware run -->

## Checklist
- [ ] Branch cut from `dev` and named `<type>/<slug>` (feat fix chore docs refactor test ci perf build)
- [ ] feat/fix/perf: a line under `## [Unreleased]` in CHANGELOG.md
- [ ] New API route / CLI command / MCP tool: the web UI that drives it is in this PR (+ Guide registry and KB entry)
- [ ] `VERSION` untouched (only `scripts/release.py` changes it)
- [ ] Tests for the changed module + `.venv/bin/ruff check src/ scripts/` pass
