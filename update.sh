#!/usr/bin/env bash
# Finetune Studio — self-healing update script.
#
# Run on fan-dragon (or any host running finetune-studio) to:
#   1. Pull latest code (fast-forward only)
#   2. Repair venv if broken (recreate via install.sh)
#   3. Sync pip deps (pip install -e .)
#   4. Build llama.cpp CLI tools if missing
#   5. Run DB migrations (init_db)
#   6. Restart finetune-studio.service
#
# Outputs structured log to stdout (each line timestamped) so the
# HTTP wrapper at /api/system/update can stream it to the DB.
#
# Flags:
#   --no-pull     skip git pull
#   --no-llama    skip llama.cpp CLI install
#   --no-restart  skip service restart
#   --check       dry-run (deps + status, no changes)
#   --repair      force venv recreate even if present

set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"

VENV_DIR="${VENV_DIR:-.venv}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
NO_PULL=0
NO_LLAMA=0
NO_RESTART=0
CHECK_MODE=0
REPAIR_MODE=0

log()  { echo "[$(date +%H:%M:%S)] $*"; }
warn() { echo "[$(date +%H:%M:%S)] WARN: $*" >&2; }
die()  { echo "[$(date +%H:%M:%S)] ERROR: $*" >&2; exit 1; }

for arg in "$@"; do
    case "$arg" in
        --no-pull)    NO_PULL=1 ;;
        --no-llama)   NO_LLAMA=1 ;;
        --no-restart) NO_RESTART=1 ;;
        --check)      CHECK_MODE=1 ;;
        --repair)     REPAIR_MODE=1 ;;
        --help|-h)    sed -n '2,22p' "$0" | sed 's/^# *//'; exit 0 ;;
        *) die "Unknown arg: $arg  (--help for usage)" ;;
    esac
done

VENV_PY="$PWD/$VENV_DIR/bin/python"

# ── Step 1: git pull ────────────────────────────────────────────────────
if [ "$NO_PULL" = "0" ]; then
    log "git pull (ff-only)..."
    # Deterministic env for git/ssh: derive HOME from the passwd DB (the
    # service env may lack it). The sandbox (ProtectSystem=full +
    # ProtectHome=read-only) also breaks openssh's ownership check on
    # /etc/ssh/ssh_config.d/* → "Bad owner or permissions" — only inside
    # the service. GIT_SSH_COMMAND with -F skips the system config and
    # reads just the user's, which authenticates fine (verified via
    # systemd-run repro, 2026-09-11).
    _real_home="$(getent passwd "$(id -u)" | cut -d: -f6)"
    _git_ssh="ssh"
    if [ -f "$_real_home/.ssh/config" ]; then
        _git_ssh="ssh -F $_real_home/.ssh/config"
    fi
    if ! env -i HOME="$_real_home" PATH="/usr/local/bin:/usr/bin:/bin" \
            GIT_SSH_COMMAND="$_git_ssh" git pull --ff-only origin main 2>&1; then
        warn "git pull failed (local may be ahead or divergent). Using local code."
    fi
else
    log "skipping git pull (--no-pull)."
fi

# ── Step 2: venv check / repair ─────────────────────────────────────────
DIAGNOSE_PY="$(dirname "$(readlink -f "$0")")/scripts/install_diagnose.py"
run_diagnose() {
    local py
    if [ -x "$VENV_PY" ] && "$VENV_PY" -c "import sys" >/dev/null 2>&1; then
        py="$VENV_PY"
    else
        py="$(command -v python3 || command -v python)"
    fi
    "$py" "$DIAGNOSE_PY" --venv "$VENV_DIR" --llama-cpp "$LLAMA_CPP_DIR" "$@"
}

if [ "$REPAIR_MODE" = "1" ]; then
    log "REPAIR: removing $VENV_DIR and recreating..."
    rm -rf "$VENV_DIR"
fi

if [ ! -x "$VENV_PY" ] || ! "$VENV_PY" -c "import sys" >/dev/null 2>&1; then
    warn "venv missing or broken at $VENV_DIR — running install.sh..."
    bash install.sh 2>&1 | tail -8 || die "install.sh failed"
    VENV_PY="$PWD/$VENV_DIR/bin/python"
    [ -x "$VENV_PY" ] || die "venv still broken after install.sh"
else
    # Existing venv: run deep diagnostic to catch mixed installs,
    # missing deps, and broken torchaudio that the old existence check missed.
    if [ "$CHECK_MODE" = "0" ]; then
        log "Running venv health check (autodetect broken installs)..."
        if ! run_diagnose --no-service-check > /tmp/fts-update-diag.txt 2>&1; then
            warn "Health check found issues — see /tmp/fts-update-diag.txt"
            cat /tmp/fts-update-diag.txt | sed 's/^/  /'
            log "Attempting autofix via scripts/install_diagnose.py --repair..."
            if run_diagnose --repair --no-service-check; then
                log "✓ repair ok, continuing with update."
            else
                warn "Repair couldn't fully resolve. Recreating venv..."
                rm -rf "$VENV_DIR"
                bash install.sh 2>&1 | tail -8 || die "install.sh failed"
                VENV_PY="$PWD/$VENV_DIR/bin/python"
                [ -x "$VENV_PY" ] || die "venv still broken after install.sh"
            fi
        else
            log "✓ venv health OK."
        fi
    fi
fi

# ── Step 3: pip sync ────────────────────────────────────────────────────
if [ "$CHECK_MODE" = "0" ]; then
    log "Syncing deps via 'pip install -e .'..."
    "$VENV_PY" -m pip install --quiet --disable-pip-version-check -e . 2>&1 | tail -5 \
        || warn "pip install -e . failed — deps may be stale"
else
    log "check mode: skipping pip install"
fi

# ── Step 4: llama.cpp CLI ────────────────────────────────────────────────
if [ "$NO_LLAMA" = "0" ] && [ "$CHECK_MODE" = "0" ]; then
    if [ -x "$LLAMA_CPP_DIR/build/bin/llama-quantize" ] && [ -f "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" ]; then
        log "llama.cpp CLI present at $LLAMA_CPP_DIR."
    else
        log "Building llama.cpp CLI at $LLAMA_CPP_DIR..."
        command -v cmake >/dev/null 2>&1 || warn "cmake not found; install build-essential + cmake"
        command -v git   >/dev/null 2>&1 || warn "git not found"
        if [ ! -d "$LLAMA_CPP_DIR" ]; then
            git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_CPP_DIR" \
                || warn "git clone llama.cpp failed"
        fi
        if [ -d "$LLAMA_CPP_DIR" ]; then
            "$VENV_PY" -m pip install --quiet --disable-pip-version-check \
                -r "$LLAMA_CPP_DIR/requirements/requirements-convert_hf_to_gguf.txt" 2>&1 | tail -3 \
                || warn "convert_hf_to_gguf pip deps failed"
            cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" 2>&1 | tail -2 \
                || warn "cmake configure failed"
            cmake --build "$LLAMA_CPP_DIR/build" --config Release -j 2>&1 | tail -3 \
                || warn "cmake build failed"
        fi
    fi
else
    log "skipping llama.cpp install."
fi

# ── Step 5: DB migrations ───────────────────────────────────────────────
log "Running DB migrations (init_db)..."
"$VENV_PY" -c "from finetune_studio.db import init_db; init_db(); print('  schema OK')" 2>&1 \
    || die "init_db failed"

# ── Step 6: restart ─────────────────────────────────────────────────────
if [ "$NO_RESTART" = "0" ] && [ "$CHECK_MODE" = "0" ]; then
    if systemctl --user is-active finetune-studio >/dev/null 2>&1; then
        log "Restarting finetune-studio.service..."
        systemctl --user restart finetune-studio 2>&1 \
            || warn "service restart failed (will keep current code path)"
    else
        warn "finetune-studio.service not active on this host"
    fi
else
    log "skipping service restart."
fi

log "Update complete."
