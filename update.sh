#!/usr/bin/env bash
# Finetune Studio — self-healing update script.
#
# Run on any host running finetune-studio to:
#   1. Pull latest code (fast-forward only)
#   2. Repair venv if broken (recreate via install.sh)
#   3. Sync pip deps: `-e .[parsers]` + every extra install.sh installs
#      (bitsandbytes, numpy/scipy, unsloth when compatible) under the torch pin
#   4. Build (or rebuild for the right GPU backend) the llama.cpp CLI tools
#   5. Run DB migrations (init_db) + the one-time file-source migration (knowledge.migrate)
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
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$PWD/.llama.cpp}"
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

# Steps that failed: any of them stops the update before the restart and makes it exit 1 with the reasons (it used to
# warn, restart and end "Update complete" with stale dependencies).
PROBLEMS=()
problem() { warn "$*"; PROBLEMS+=("$*"); }

# uv: PATH first, then the path install.sh recorded, then where uv's installer and package managers put it. A shell
# started by systemd (Settings -> Apply update) has a short PATH without ~/.local/bin or Linuxbrew, so `uv` alone was
# not found there and every dependency step failed.
find_uv() {
    local home c
    home="${HOME:-$(getent passwd "$(id -u)" | cut -d: -f6)}"
    for c in "$(command -v uv 2>/dev/null || true)" "$(cat "$VENV_DIR/fts-uv-path" 2>/dev/null || true)" \
             "${UV_INSTALL_DIR:+$UV_INSTALL_DIR/uv}" "${XDG_BIN_HOME:+$XDG_BIN_HOME/uv}" "$home/.local/bin/uv" \
             "$home/.cargo/bin/uv" /home/linuxbrew/.linuxbrew/bin/uv "$home/.linuxbrew/bin/uv" /usr/local/bin/uv /usr/bin/uv; do
        if [ -n "$c" ] && [ -x "$c" ]; then echo "$c"; return 0; fi
    done
    return 1
}
UV="$(find_uv || true)"
if [ -n "$UV" ]; then
    case ":$PATH:" in *":$(dirname "$UV"):"*) ;; *) export PATH="$(dirname "$UV"):$PATH" ;; esac   # accel_plan.py calls uv too
fi

# uv-created venvs have NO pip module (hit on the dev box, 2026-09-25) —
# prefer uv, fall back to the venv's own pip. Mirrors install.sh.
pip_install() {
    if [ -n "$UV" ]; then
        "$UV" pip install --python "$VENV_PY" "$@"
    elif "$VENV_PY" -c "import pip" >/dev/null 2>&1; then
        "$VENV_PY" -m pip install "$@"
    else
        echo "uv not found (looked on PATH=$PATH, in $VENV_DIR/fts-uv-path, ~/.local/bin, ~/.cargo/bin, Linuxbrew) and this venv has no pip" >&2
        return 1
    fi
}

ACCEL_PY="$PWD/scripts/accel_plan.py"
accel() {  # accel <subcommand> [args...]: scripts/accel_plan.py owns GPU detection + every wheel/CMake choice
    local sub="$1"; shift
    local py="$VENV_PY"
    [ -x "$py" ] || py="$(command -v python3 || command -v python)"
    "$py" "$ACCEL_PY" "$sub" ${ACCEL_FLAGS[@]+"${ACCEL_FLAGS[@]}"} "$@"
}
# The accelerator forced at install time (install.sh --cpu / --gpu) stays the choice: health check, repair and
# dep sync all honour it instead of "repairing" a deliberate CPU install to GPU wheels.
if [ -z "${FTS_FORCE_VENDOR:-}" ] && [ -s "$VENV_DIR/fts-force-vendor" ]; then
    FTS_FORCE_VENDOR="$(tr -d '[:space:]' < "$VENV_DIR/fts-force-vendor")"
    export FTS_FORCE_VENDOR
    log "accelerator: $FTS_FORCE_VENDOR (chosen at install time)"
fi
ACCEL_FLAGS=()
[ -n "${FTS_FORCE_VENDOR:-}" ] && ACCEL_FLAGS+=(--gpu "$FTS_FORCE_VENDOR")

# Keep the GPU-matched torch family pinned during dep sync: install.sh writes
# .venv/torch-constraints.txt after install_torch. Without it, a later resolve
# (unsloth/`-e .`) silently upgrades torch to the default PyPI build and
# breaks the venv on older drivers (undefined symbol: ncclCommResume, 535).
# Existing installs that predate the file get it generated from the currently
# installed (working) torch family so sync can't drift.
torch_constraint_args() {
    local f="$VENV_DIR/torch-constraints.txt"
    # Written every time: an update can bring new pins (pins/app.txt goes into this file).
    accel constraints --venv "$VENV_DIR" >/dev/null 2>&1 || true
    [ -s "$f" ] && printf -- '-c\n%s\n' "$f"
}

# ── Step 1: git pull ────────────────────────────────────────────────────
# Deterministic env for git/ssh (see the pull below): shared by the check's fetch and the pull.
_real_home="$(getent passwd "$(id -u)" | cut -d: -f6)"
_git_ssh="ssh"
[ -f "$_real_home/.ssh/config" ] && _git_ssh="ssh -F $_real_home/.ssh/config"
_branch="$(git symbolic-ref --short -q HEAD || true)"
git_env() { env -i HOME="$_real_home" PATH="/usr/local/bin:/usr/bin:/bin" GIT_SSH_COMMAND="$_git_ssh" "$@"; }
_before="$(git rev-parse --short HEAD 2>/dev/null || true)"
if [ "$CHECK_MODE" = "1" ]; then
    # A check answers "is there anything to update?": fetch (changes no file) and list what a pull would bring.
    case "$_branch" in
        main|dev)
            if git_env git fetch --quiet origin "$_branch" 2>&1; then
                _new="$(git rev-list --count "HEAD..origin/$_branch" 2>/dev/null || echo 0)"
                if [ "$_new" = "0" ]; then
                    log "Up to date: $_branch is at $_before, the same as GitHub."
                else
                    log "Update available: $_new new change(s) on $_branch ($_before -> $(git rev-parse --short "origin/$_branch")):"
                    git log --format='    - %s' "HEAD..origin/$_branch" | head -15
                fi
            else
                warn "could not reach GitHub to look for updates (offline?)."
            fi ;;
        *) log "on '${_branch:-a detached HEAD}', not main or dev: updates come from your own git commands." ;;
    esac
    log "--check: skipping git pull (dry-run, no changes)."
elif [ "$NO_PULL" = "0" ]; then
    log "git pull (ff-only)..."
    # Deterministic env for git/ssh: derive HOME from the passwd DB (the
    # service env may lack it). The sandbox (ProtectSystem=full +
    # ProtectHome=read-only) also breaks openssh's ownership check on
    # /etc/ssh/ssh_config.d/* → "Bad owner or permissions" — only inside
    # the service. GIT_SSH_COMMAND with -F skips the system config and
    # reads just the user's, which authenticates fine (verified via
    # systemd-run repro, 2026-09-11).
    # Pull the branch this checkout is on: `main` = releases (users), `dev` = development (the dev box).
    # Any other branch (a feature branch) is work in progress and is left alone (CONTRIBUTING.md).
    case "$_branch" in
        main|dev)
            if ! git_env git pull --ff-only origin "$_branch" 2>&1; then
                warn "git pull failed (local may be ahead or divergent). Using local code."
            fi
            _after="$(git rev-parse --short HEAD 2>/dev/null || true)"
            if [ "$_before" = "$_after" ]; then
                log "Code: already up to date at $_after."
            else
                log "Code: updated $_before -> $_after ($(git rev-list --count "$_before..$_after") change(s))."
            fi ;;
        *)
            warn "on '${_branch:-a detached HEAD}', not main or dev: skipping git pull (update a feature branch yourself)." ;;
    esac
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

# Rebuild the venv from scratch, keeping the old one until the new one passes the health check: a rebuild that loses the
# GPU (or anything else) puts the working environment back instead of leaving the app broken.
rebuild_venv() {
    local backup="$VENV_DIR.before-rebuild" rc=0
    rm -rf "$backup"
    [ -d "$VENV_DIR" ] && mv "$VENV_DIR" "$backup"
    bash install.sh 2>&1 | tail -8 || rc=$?
    VENV_PY="$PWD/$VENV_DIR/bin/python"
    if [ "$rc" = "0" ] && [ -x "$VENV_PY" ] && run_diagnose --no-service-check > /tmp/fts-update-diag.txt 2>&1; then
        log "✓ the rebuilt environment passed the health check."
        rm -rf "$backup"
        return 0
    fi
    warn "the rebuilt environment did not pass the health check:"
    sed 's/^/  /' /tmp/fts-update-diag.txt 2>/dev/null | tail -20 || true
    if [ -d "$backup" ]; then
        rm -rf "$VENV_DIR"
        mv "$backup" "$VENV_DIR"
        VENV_PY="$PWD/$VENV_DIR/bin/python"
        die "Repair failed; the previous environment was put back unchanged."
    fi
    die "the environment could not be rebuilt (install.sh exit $rc)."
}

if [ "$REPAIR_MODE" = "1" ]; then
    log "REPAIR: rebuilding $VENV_DIR from scratch (the old one is kept until the new one passes)..."
    rebuild_venv
fi

if [ "$CHECK_MODE" = "1" ] && { [ ! -x "$VENV_PY" ] || ! "$VENV_PY" -c "import sys" >/dev/null 2>&1; }; then
    die "--check: venv missing or broken at $VENV_DIR (run: bash update.sh, or bash install.sh)"
elif [ ! -x "$VENV_PY" ] || ! "$VENV_PY" -c "import sys" >/dev/null 2>&1; then
    warn "venv missing or broken at $VENV_DIR — running install.sh..."
    bash install.sh 2>&1 | tail -8 || die "install.sh failed"
    VENV_PY="$PWD/$VENV_DIR/bin/python"
    [ -x "$VENV_PY" ] || die "venv still broken after install.sh"
else
    # Existing venv: run deep diagnostic to catch mixed installs,
    # missing deps, and broken torchaudio that the old existence check missed.
    if [ "$CHECK_MODE" = "1" ]; then
        log "--check: venv health (read-only)..."
        CHECK_RC=0
        run_diagnose --no-service-check || CHECK_RC=$?
    else
        log "Running venv health check (autodetect broken installs)..."
        if ! run_diagnose --no-service-check > /tmp/fts-update-diag.txt 2>&1; then
            warn "Health check found issues — see /tmp/fts-update-diag.txt"
            cat /tmp/fts-update-diag.txt | sed 's/^/  /'
            log "Attempting autofix via scripts/install_diagnose.py --repair..."
            if run_diagnose --repair --no-service-check; then
                log "✓ repair ok, continuing with update."
            else
                warn "Repair couldn't fully resolve. Recreating venv..."
                rebuild_venv
            fi
        else
            log "✓ venv health OK."
        fi
    fi
fi

# ── Step 3: pip sync ────────────────────────────────────────────────────
if [ "$CHECK_MODE" = "0" ]; then
    log "Syncing deps via 'pip install -e .[parsers]'..."
    # [parsers] kept in sync like install.sh does — without it, updated
    # hosts keep whatever parser-dep set (if any) was resolved at install
    # time, and PDF/DOCX/XLSX/PPTX/HTML/RTF sources silently stop parsing.
    # shellcheck disable=SC2046
    pip_install --quiet $(torch_constraint_args) -e '.[parsers]' 2>&1 | tail -5 \
        || problem "dependency sync (pip install -e .[parsers]) failed$([ -n "$UV" ] || echo ': uv was not found')"
    # Every extra install.sh installs, so updated hosts match fresh ones:
    # bitsandbytes (CUDA/ROCm/XPU backends in one wheel), numpy/scipy (abliteration),
    # unsloth (NVIDIA only; skipped unless a release resolves against the installed stack).
    log "Syncing optional packages (bitsandbytes, numpy, scipy, unsloth, Qwen3.5 kernels)..."
    accel install bitsandbytes --venv "$VENV_DIR" 2>&1 | tail -3 \
        || problem "bitsandbytes sync failed"
    # shellcheck disable=SC2046
    pip_install --quiet $(torch_constraint_args) "numpy>=1.24.0" "scipy>=1.10.0" 2>&1 | tail -2 \
        || problem "numpy/scipy sync failed"
    accel install unsloth --venv "$VENV_DIR" 2>&1 | tail -3 \
        || problem "unsloth sync failed"
    # Qwen3.5 linear-attention kernels: re-probed on every update (a torch change can break a built extension)
    accel install linear-attn --venv "$VENV_DIR" 2>&1 | tail -4 \
        || problem "Qwen3.5 kernel sync failed"
    # PyTorch and llama-cpp-python are only moved by a rebuild (they depend on the GPU, and the second one compiles).
    "$VENV_PY" scripts/pin_versions.py check --venv "$VENV_DIR" 2>&1 | sed 's/^/  /' \
        || warn "this install differs from the tested versions above. Settings → Repair moves PyTorch and llama-cpp-python."
else
    log "check mode: skipping pip install"
fi

# ── Step 4: llama.cpp CLI ────────────────────────────────────────────────
# build-llama-cli is a no-op when the CLI exists AND was built for this host's
# GPU backend; a CPU-only build on a GPU host (or a backend change) is rebuilt.
if [ "$NO_LLAMA" = "0" ] && [ "$CHECK_MODE" = "0" ]; then
    accel build-llama-cli --dir "$LLAMA_CPP_DIR" --venv "$VENV_DIR" 2>&1 | tail -12 \
        || problem "llama.cpp CLI build failed (see output above)"
else
    log "skipping llama.cpp install."
fi

# ── Step 5: DB migrations ───────────────────────────────────────────────
if [ "$CHECK_MODE" = "1" ]; then
    log "--check: skipping DB migrations (dry-run, no changes)."
else
    log "Running DB migrations (init_db + knowledge sources)..."
    "$VENV_PY" -c "from finetune_studio.db import init_db; init_db(); from finetune_studio.knowledge.migrate import migrate_all; print('  schema OK, projects moved:', migrate_all())" 2>&1 \
        || die "init_db failed"
fi

# Heretic's overlay reads this venv's packages: a rebuild or a dependency change can stop it from starting.
if [ "$CHECK_MODE" = "0" ]; then
    healed="$("$VENV_PY" -c "from finetune_studio.training.heretic_tool import heal; print(heal())" 2>/dev/null | tail -1)" \
        || warn "Heretic no longer starts and could not be re-installed: fts heretic install"
    [ -n "$healed" ] && log "Heretic: $healed for the new environment."
fi

# ── Step 6: restart ─────────────────────────────────────────────────────
# The user unit is restarted only when it serves THIS checkout: an update of a second clone (a test install, a
# sandbox) must not bounce the instance another checkout runs.
unit_active=0; unit_dir=""
if systemctl --user is-active finetune-studio >/dev/null 2>&1; then
    unit_active=1
    unit_dir="$(systemctl --user show -p WorkingDirectory --value finetune-studio 2>/dev/null)"
fi
# A restart stops running jobs (training, exports): never do it under them. The code is updated on disk; the new
# version starts when the service restarts after the job (or run: systemctl --user restart finetune-studio).
running_jobs="$("$VENV_PY" -m finetune_studio.cli status --json 2>/dev/null \
    | "$VENV_PY" -c 'import json,sys; print(" ".join(j["id"] for j in json.load(sys.stdin).get("jobs", []) if j.get("running")))' \
    2>/dev/null || true)"
if [ "${#PROBLEMS[@]}" -gt 0 ] && [ "$CHECK_MODE" = "0" ]; then
    echo "[$(date +%H:%M:%S)] ERROR: update not finished, ${#PROBLEMS[@]} step(s) failed: $(IFS=';'; echo "${PROBLEMS[*]}")." >&2
    echo "[$(date +%H:%M:%S)] ERROR: the service was not restarted, so it keeps running what it started with. Fix the step above, then run 'bash update.sh' in a terminal (or Settings -> Repair)." >&2
    exit 1
fi
if [ -n "$running_jobs" ] && [ "$NO_RESTART" = "0" ] && [ "$CHECK_MODE" = "0" ]; then
    warn "not restarting: job(s) running ($running_jobs). The update applies at the next restart after they end."
    NO_RESTART=1
fi
if [ "$NO_RESTART" = "0" ] && [ "$CHECK_MODE" = "0" ]; then
    if [ "$unit_active" = "1" ] && [ "$(readlink -f "$unit_dir" 2>/dev/null)" != "$(readlink -f "$PWD")" ]; then
        warn "finetune-studio.service runs ${unit_dir:-another checkout}, not $PWD: not restarting it"
    elif [ "$unit_active" = "1" ]; then
        unit_exec="$(systemctl --user show -p ExecStart --value finetune-studio 2>/dev/null)"
        if ! grep -q "finetune_studio.supervisor" <<<"$unit_exec"; then
            warn "finetune-studio.service still runs a bare uvicorn (no supervisor): run 'bash install-service.sh' once on this host"
        elif grep -q -- "--host" <<<"$unit_exec"; then
            # Units before 2026-10-15 baked the address into ExecStart, which overrides server.json (Settings ->
            # Access & network). Rewrite the unit; install-service.sh carries the old address into server.json.
            log "Moving the listen address out of the unit into server.json..."
            bash install-service.sh --no-start 2>&1 | tail -4 || warn "unit rewrite failed; run bash install-service.sh"
        fi
        log "Restarting finetune-studio.service..."
        systemctl --user restart finetune-studio 2>&1 \
            || warn "service restart failed (will keep current code path)"
    else
        # No systemd unit: a manually started supervisor (`fts up --manual`) restarts only its web child.
        "$VENV_PY" -m finetune_studio.cli restart web >/dev/null 2>&1
        rc=$?
        if [ "$rc" = "0" ]; then
            log "Restarting finetune-studio.service (web component of the manual supervisor)..."
        elif [ "$rc" = "3" ]; then
            warn "finetune-studio is not running under systemd or a supervisor on this host; start it with 'fts up'"
        else
            warn "supervisor restart of the web component failed (exit $rc)"
        fi
    fi
else
    log "skipping service restart."
fi

if [ "$CHECK_MODE" = "1" ]; then
    log "Check complete (health rc=${CHECK_RC:-0}; nothing was changed)."
    exit "${CHECK_RC:-0}"
fi
log "Update complete."
