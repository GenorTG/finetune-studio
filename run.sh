#!/usr/bin/env bash
# Finetune Studio — run helper.
# Auto-installs if venv is missing, then starts the webui.

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# Bootstrap .venv if missing
if [ ! -d ".venv" ] && [ "${USE_CONDA:-0}" != "1" ]; then
    echo "[run] No venv found. Running install.sh first..."
    bash install.sh
fi

if [ "${USE_CONDA:-0}" = "1" ]; then
    CONDA_ENV="${CONDA_ENV:-chris-ai}"
    for cb in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda" "/usr/local/miniconda3"; do
        [ -x "$cb/bin/conda" ] && CONDA_BASE="$cb" && break
    done
    if [ -z "${CONDA_BASE:-}" ]; then
        echo "[run] ERROR: conda not found. Install miniconda or unset USE_CONDA." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
else
    # shellcheck disable=SC1090
    source .venv/bin/activate
fi

# Address: $FTS_ROOT/server.json (Settings -> Access & network; default 127.0.0.1:7860, this computer only).
# HOST=... / PORT=... override it for this run.
echo "[run] Finetune Studio, address: $(python -m finetune_studio.cli server show)"
exec python -m finetune_studio.cli webui ${HOST:+--host "$HOST"} ${PORT:+--port "$PORT"} "$@"
