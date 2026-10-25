#!/usr/bin/env bash
# Point this clone's git hooks at scripts/git-hooks (CONTRIBUTING.md "Branches"):
#   pre-commit  no commits on main/dev; VERSION changes only on release/hotfix branches
#   pre-push    no direct pushes to main/dev, never the history archive, branch + tag naming
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath scripts/git-hooks
chmod +x scripts/git-hooks/pre-commit scripts/git-hooks/pre-push
echo "hooks installed: core.hooksPath=scripts/git-hooks (pre-commit, pre-push)"
