#!/usr/bin/env bash
# Install + enable the finetune-studio systemd unit on a fan-dragon-style host.
# Assumes:
#   - repo checked out at /home/genortg/finetune-studio (override with REPO_DIR)
#   - .venv already provisioned by `bash install.sh`
#   - root or sudo available for systemctl / log dir
#
# After install: `systemctl status finetune-studio` to verify,
# `journalctl -u finetune-studio -f` to tail.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/genortg/finetune-studio}"
SERVICE_USER="${SERVICE_USER:-genortg}"
SERVICE_FILE="${REPO_DIR}/scripts/finetune-studio.service"

if [[ ! -f "$SERVICE_FILE" ]]; then
  echo "missing $SERVICE_FILE" >&2
  exit 1
fi

sudo install -d -m 0755 /var/log/finetune-studio
sudo chown "$SERVICE_USER":"$SERVICE_USER" /var/log/finetune-studio

sudo cp "$SERVICE_FILE" /etc/systemd/system/finetune-studio.service
sudo systemctl daemon-reload
sudo systemctl enable --now finetune-studio

echo "---"
sudo systemctl --no-pager status finetune-studio | head -15
echo "---"
ss -ltnp | grep 7860 || echo "(nothing on 7860 yet — give it a sec)"
