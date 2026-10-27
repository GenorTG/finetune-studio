#!/usr/bin/env bash
# Finetune Studio — run helper.
# Auto-installs if venv is missing, then starts the webui.

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# Bootstrap .venv if missing
if [ ! -d ".venv" ]; then
    echo "[run] No venv found. Running install.sh first..."
    bash install.sh
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# Address: $FTS_ROOT/server.json (Settings -> Access & network; default 127.0.0.1:7860, this computer only).
# HOST=... / PORT=... override it for this run.
echo "[run] Finetune Studio, address: $(python -m finetune_studio.cli server show)"
exec python -m finetune_studio.cli webui ${HOST:+--host "$HOST"} ${PORT:+--port "$PORT"} "$@"
