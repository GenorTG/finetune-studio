#!/usr/bin/env bash
#
# Finetune Studio — safe update script
#
# Checks GitHub for a newer release/tag and updates the running instance
# WITHOUT touching user data (projects, files, DB, RAG corpora, models).
#
# Usage:
#   bash scripts/update.sh              # auto-detect current version, pull, restart
#   bash scripts/update.sh --check      # only check, don't update
#   bash scripts/update.sh --force      # force reinstall of deps
#   bash scripts/update.sh v2026.09.12   # checkout specific tag/commit
#
# Data safety: only the source tree under $FTS_HOME is touched.
# All user data lives under ~/.finetune-studio/ (see docs/DEPENDENCIES.md).

set -euo pipefail

FTS_HOME="${FTS_HOME:-/home/genortg/finetune-studio}"
REMOTE="${REMOTE:-github.com:GenorTG/finetune-studio.git}"
SERVICE="${SERVICE:-finetune-studio.service}"
CHECK_ONLY=0
FORCE_DEPS=0
TARGET_REF=""

# ── Arg parsing ────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)    CHECK_ONLY=1; shift ;;
        --force)    FORCE_DEPS=1; shift ;;
        -h|--help)  head -18 "$0"; exit 0 ;;
        *)          TARGET_REF="$1"; shift ;;
    esac
done

cd "$FTS_HOME"

# ── Current version ────────────────────────────────────────────────
CURRENT_REF=""
if git rev-parse HEAD &>/dev/null 2>&1; then
    CURRENT_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
fi
CURRENT_TAG="$(git describe --tags --exact-match 2>/dev/null || echo '(no tag)')"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           Finetune Studio — Update Checker                  ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Installed: ${CURRENT_REF}  tag: ${CURRENT_TAG}"
echo "║  Source:    ${FTS_HOME}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# ── Fetch latest ──────────────────────────────────────────────────
echo "→ Fetching latest from ${REMOTE}..."
if ! git fetch --tags --force origin main 2>/dev/null; then
    echo "✗ Cannot reach ${REMOTE}."
    echo "  Check network / SSH keys. Data untouched."
    exit 1
fi

# ── Resolve target ref ────────────────────────────────────────────
if [[ -n "$TARGET_REF" ]]; then
    REF="$TARGET_REF"
    echo "→ Target: user-specified ref ${REF}"
else
    # Prefer latest tag on main, else latest main
    REF="$(git rev-parse origin/main 2>/dev/null || echo '')"
    LATEST_TAG="$(git describe --tags --abbrev=0 origin/main 2>/dev/null || echo '')"
    if [[ -n "$LATEST_TAG" ]]; then
        REF="$(git rev-list -n1 "$LATEST_TAG" 2>/dev/null || echo "$REF")"
        echo "→ Latest release: ${LATEST_TAG} (${REF:0:8})"
    else
        echo "→ Latest main: ${REF:0:8}"
    fi
fi

# ── Compare ───────────────────────────────────────────────────────
LOCAL_REF="$(git rev-parse HEAD 2>/dev/null || echo '')"
REMOTE_REF="$(git rev-parse "$REF" 2>/dev/null || echo '')"

if [[ "$LOCAL_REF" == "$REMOTE_REF" ]] && [[ "$CHECK_ONLY" -eq 0 ]]; then
    echo "✓ Already up-to-date."
    exit 0
fi

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    if [[ "$LOCAL_REF" != "$REMOTE_REF" ]]; then
        echo "✓ Update available: ${LOCAL_REF:0:8} → ${REMOTE_REF:0:8}"
        echo "  Run without --check to apply."
    else
        echo "✓ Already up-to-date."
    fi
    exit 0
fi

# ── Confirm ───────────────────────────────────────────────────────
echo
echo "┌──────────────────────────────────────────────────────────────┐"
│  ⚠  UPDATE WILL:                                              │
│    1. Stash local changes (preserved, restorable)              │
│    2. Reset source to ${REF:0:8}                               │
│    3. Reinstall Python deps if pyproject.toml changed          │
│    4. Run DB migrations (non-destructive)                      │
│    5. Restart ${SERVICE}                                       │
│                                                               │
│  ✗  WILL NOT TOUCH:                                            │
│    ~/.finetune-studio/  (projects, files, DB, RAG, models)     │
│    Custom config in config.yaml (preserved)                    │
└──────────────────────────────────────────────────────────────┘"
echo
read -p "Proceed? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted. Data untouched."
    exit 0
fi

# ── Stash local changes ──────────────────────────────────────────
echo "→ Stashing local modifications..."
git stash push -m "update-script-$(date +%Y%m%d-%H%M%S)" 2>/dev/null || true

# ── Pull ─────────────────────────────────────────────────────────
echo "→ Checking out ${REF:0:8}..."
if ! git checkout "$REF" 2>/dev/null; then
    echo "✗ Checkout failed. Restoring previous state..."
    git stash pop 2>/dev/null || true
    exit 1
fi

# ── Check if deps changed ────────────────────────────────────────
DEPS_CHANGED=0
if git diff --name-only HEAD@{1} HEAD 2>/dev/null | grep -qE "pyproject.toml|requirements.*\.txt"; then
    DEPS_CHANGED=1
    echo "→ Dependency file changed — will reinstall."
fi

# ── Install deps ──────────────────────────────────────────────────
VENV="${FTS_HOME}/.venv"
if [[ -d "$VENV" ]] && { [[ "$DEPS_CHANGED" -eq 1 ]] || [[ "$FORCE_DEPS" -eq 1 ]]; }; then
    echo "→ Reinstalling Python dependencies..."
    "$VENV/bin/pip" install -e "${FTS_HOME}[all]" --quiet 2>&1 | tail -3
    # Also install optional packages that may be new
    echo "→ Installing optional packages (unsloth, gptq+optimum, numpy, scipy)..."
    "$VENV/bin/pip" install "numpy>=1.24.0" "scipy>=1.10.0" "unsloth>=2024.10.0" --quiet 2>&1 | tail -3 || true
    "$VENV/bin/pip" install -e "${FTS_HOME}[gptq]" --quiet 2>&1 | tail -3 || true
fi

# ── DB migrations (non-destructive) ───────────────────────────────
if [[ -f "${FTS_HOME}/src/finetune_studio/db/migrations.py" ]]; then
    echo "→ Running DB migrations..."
    "$VENV/bin/python" -m finetune_studio.db.migrations 2>&1 | tail -5 || true
fi

# ── Restart service ───────────────────────────────────────────────
echo "→ Restarting ${SERVICE}..."
if systemctl --user is-active "$SERVICE" &>/dev/null; then
    systemctl --user restart "$SERVICE"
    sleep 2
    if systemctl --user is-active "$SERVICE" &>/dev/null; then
        echo "✓ ${SERVICE} restarted successfully."
    else
        echo "✗ ${SERVICE} failed to start. Check: journalctl --user -u ${SERVICE}"
        exit 1
    fi
else
    echo "! ${SERVICE} not managed by systemd. Restart manually:"
    echo "  cd ${FTS_HOME} && .venv/bin/python -m finetune_studio.webui.app"
fi

# ── Done ──────────────────────────────────────────────────────────
NEW_REF="$(git rev-parse --short HEAD)"
echo
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✓ Updated: ${CURRENT_REF:0:8} → ${NEW_REF}                                ║"
echo "║  Data preserved: ~/.finetune-studio/                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
