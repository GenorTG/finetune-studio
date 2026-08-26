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

PORT="${PORT:-7860}"
HOST="${HOST:-0.0.0.0}"
echo "[run] Finetune Studio → http://localhost:$PORT"
exec python -m uvicorn finetune_studio.webui.app:app --host "$HOST" --port "$PORT" "$@"
