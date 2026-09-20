#!/usr/bin/env bash
# Point this repo's git hooks at the versioned scripts/git-hooks directory.
# One commit = one new build version (VERSION file, BUILD segment auto-bump).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath scripts/git-hooks
chmod +x scripts/git-hooks/commit-msg
echo "hooks installed: core.hooksPath=scripts/git-hooks (commit-msg bumps VERSION)"
