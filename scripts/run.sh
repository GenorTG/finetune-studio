#!/usr/bin/env bash
# Finetune Studio — launcher that reads port from user settings.
# Used by the systemd unit so port changes via the Settings page take effect.

set -euo pipefail

SETTINGS="$HOME/.finetune-studio/settings.json"
PORT=7860
HOST="0.0.0.0"

if [[ -f "$SETTINGS" ]]; then
    # Use python to parse JSON (jq may not be installed)
    read -r PORT HOST < <(python3 -c "
import json
with open('$SETTINGS') as f:
    s = json.load(f)
print(s.get('port', 7860), s.get('host', '0.0.0.0'))
")
fi

exec "$HOME/finetune-studio/.venv/bin/python" -m uvicorn finetune_studio.webui.app:app \
    --host "$HOST" --port "$PORT" --no-access-log
