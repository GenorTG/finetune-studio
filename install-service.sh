#!/usr/bin/env bash
# Finetune Studio — systemd service installer.
#
# Installs and enables a systemd user unit so the webui auto-starts on
# login (or on boot if loginctl enable-linger is set for the user).
# This replaces the bare nohup/uvicorn pattern that doesn't survive a
# reboot or crash.
#
# Flags:
#   --system     install as a system unit (/etc/systemd/system/...) — REQUIRES sudo
#   --uninstall  remove the unit + stop the service
#   --restart    just restart the existing service (no install)
#   --status     show current unit status
#   --no-start   install + enable, but do not start (CI prep)
#   --help       show usage
#
# Detects:
#   - systemd availability (falls back to a clear error if missing)
#   - venv location (defaults to ./venv relative to the repo)
#   - port (defaults to 7860)
#
# Verifies:
#   - service is active after start (polls /api/providers for up to 20s)
#   - service is enabled (survives logout / reboot with lingering)
#
# Logs:
#   - journalctl --user -u finetune-studio -n 50   (user unit)
#   - journalctl -u finetune-studio -n 50          (system unit)

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# ── Defaults ─────────────────────────────────────────────────────────────
INSTALL_MODE="user"      # "user" or "system"
ACTION="install"
VENV_DIR="${VENV_DIR:-.venv}"
PORT="${PORT:-7860}"
HOST="${HOST:-0.0.0.0}"
FTS_ROOT="${FTS_ROOT:-}"
NO_START=0

# ── Args ────────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --system)     INSTALL_MODE="system" ;;
        --user)       INSTALL_MODE="user" ;;
        --uninstall)  ACTION="uninstall" ;;
        --restart)    ACTION="restart" ;;
        --status)     ACTION="status" ;;
        --no-start)   NO_START=1 ;;
        --help|-h)
            sed -n '2,30p' "$0" | sed 's/^# *//'
            exit 0
            ;;
        *) echo "Unknown arg: $arg  (try --help)" >&2; exit 1 ;;
    esac
done

# ── Paths ───────────────────────────────────────────────────────────────
REPO_DIR="$(pwd)"
VENV_PY="$REPO_DIR/$VENV_DIR/bin/python"

if [ "$INSTALL_MODE" = "user" ]; then
    UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
    UNIT_FILE="$UNIT_DIR/finetune-studio.service"
    UNIT_NAME="finetune-studio"
    SYSTEMCTL=(systemctl --user)
    SUDO=()
else
    UNIT_DIR="/etc/systemd/system"
    UNIT_FILE="$UNIT_DIR/finetune-studio.service"
    UNIT_NAME="finetune-studio"
    SYSTEMCTL=(systemctl)
    SUDO=(sudo)
fi

log()  { echo "[svc] $*"; }
warn() { echo "[svc] WARN: $*" >&2; }
die()  { echo "[svc] ERROR: $*" >&2; exit 1; }

# ── Systemd detection ───────────────────────────────────────────────────
command -v systemctl >/dev/null 2>&1 \
    || die "systemctl not found — this host doesn't use systemd."

# ── Status / restart shortcuts ──────────────────────────────────────────
if [ "$ACTION" = "status" ]; then
    if [ -f "$UNIT_FILE" ]; then
        "${SYSTEMCTL[@]}" status "$UNIT_NAME" --no-pager || true
    else
        echo "Unit not installed at $UNIT_FILE"
        exit 1
    fi
    exit 0
fi

if [ "$ACTION" = "restart" ]; then
    if [ ! -f "$UNIT_FILE" ]; then
        die "Unit not installed — run: bash install-service.sh"
    fi
    "${SYSTEMCTL[@]}" restart "$UNIT_NAME"
    "${SYSTEMCTL[@]}" status "$UNIT_NAME" --no-pager || true
    exit 0
fi

# ── Uninstall ───────────────────────────────────────────────────────────
if [ "$ACTION" = "uninstall" ]; then
    if [ -f "$UNIT_FILE" ]; then
        "${SYSTEMCTL[@]}" disable --now "$UNIT_NAME" 2>/dev/null || true
        "${SUDO[@]}" rm -f "$UNIT_FILE"
        "${SYSTEMCTL[@]}" daemon-reload
        log "Removed $UNIT_FILE"
    fi
    # Best-effort: kill any bare uvicorn process that survived uninstall
    pkill -f "uvicorn finetune_studio.webui.app:app" 2>/dev/null || true
    exit 0
fi

# ── Install ─────────────────────────────────────────────────────────────
log "Installing systemd $INSTALL_MODE unit..."
log "  unit file: $UNIT_FILE"
log "  python:    $VENV_PY"
log "  port:      $PORT"

# Sanity: venv must exist + importable
[ -x "$VENV_PY" ] || die "venv python not found at $VENV_PY — run install.sh first."
"$VENV_PY" -c "import fastapi, jinja2" 2>/dev/null \
    || die "venv python can't import fastapi/jinja2 — run install.sh first."

# Build environment block (overridable via FTS_ROOT etc.)
ENV_BLOCK=""
if [ -n "$FTS_ROOT" ]; then
    ENV_BLOCK="Environment=FTS_ROOT=$FTS_ROOT"
else
    ENV_BLOCK="Environment=FTS_ROOT=%h/.finetune-studio"
fi

mkdir -p "$UNIT_DIR"
cat > "$UNIT_FILE" <<EOF
# Finetune Studio — systemd unit (managed by install-service.sh)
[Unit]
Description=Finetune Studio WebUI (port $PORT)
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
WorkingDirectory=$REPO_DIR
ExecStart=$VENV_PY -m uvicorn finetune_studio.webui.app:app --host $HOST --port $PORT --no-access-log
Restart=on-failure
RestartSec=5
TimeoutStopSec=20
StandardOutput=journal
StandardError=journal
$ENV_BLOCK
# Hardening (relaxed for dev: write to project dir + ~/.finetune-studio)
ProtectSystem=full
ProtectHome=read-only
ReadWritePaths=$REPO_DIR $HOME/.finetune-studio

[Install]
WantedBy=default.target
EOF

log "Wrote $UNIT_FILE"

# Reload + enable
"${SYSTEMCTL[@]}" daemon-reload
"${SYSTEMCTL[@]}" enable "$UNIT_NAME" \
    || die "systemctl enable failed (check sudo / dbus policy)."

# For user units: enable lingering so the service survives logout/reboot
if [ "$INSTALL_MODE" = "user" ]; then
    USER_NAME="${USER:-$(id -un)}"
    if command -v loginctl >/dev/null 2>&1; then
        if ! loginctl show-user "$USER_NAME" -p Linger 2>/dev/null | grep -q "Linger=yes"; then
            log "Enabling lingering for $USER_NAME (service will survive logout/reboot)..."
            # loginctl enable-linger needs no sudo for self
            loginctl enable-linger "$USER_NAME" 2>/dev/null \
                || warn "loginctl enable-linger failed — service will stop on logout."
        else
            log "Linger already enabled for $USER_NAME."
        fi
    fi
fi

if [ "$NO_START" = "1" ]; then
    log "Installed + enabled (--no-start set). Start with: ${SYSTEMCTL[*]} start $UNIT_NAME"
    exit 0
fi

# Start + verify
log "Starting $UNIT_NAME..."
"${SYSTEMCTL[@]}" start "$UNIT_NAME" || die "systemctl start failed."

# Poll the health endpoint for up to 20 seconds
log "Waiting for webui to come up on :$PORT..."
HEALTH_URL="http://127.0.0.1:$PORT/api/providers"
for i in $(seq 1 40); do
    sleep 0.5
    if curl -fsS --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
        log "✓ webui healthy (HTTP 200 from $HEALTH_URL after ~${i}*0.5s)"
        break
    fi
    if ! "${SYSTEMCTL[@]}" is-active --quiet "$UNIT_NAME"; then
        warn "service died during startup. Recent journal:"
        "${SYSTEMCTL[@]}" status "$UNIT_NAME" --no-pager || true
        die "service failed to stay running."
    fi
done

# Final state
"${SYSTEMCTL[@]}" status "$UNIT_NAME" --no-pager || true
echo ""
echo "═══════════════════════════════════════════════════════════════"
log "✓ finetune-studio.service installed and running."
echo ""
echo "Useful commands:"
echo "  status:    ${SYSTEMCTL[*]} status $UNIT_NAME"
echo "  logs:      journalctl --user -u $UNIT_NAME -n 50 -f     # user unit"
echo "  restart:   bash install-service.sh --restart"
echo "  uninstall: bash install-service.sh --uninstall"
echo "═══════════════════════════════════════════════════════════════"
