#!/usr/bin/env bash
# Nightly Finetune Studio UI regression runner.
# Exercises every page + interaction, writes report + screenshots,
# and pushes a Discord notification with the verdict.

set -uo pipefail

if [ "${FTS_ALLOW_LIVE_E2E:-0}" != "1" ]; then
    echo "Refusing live QA; set FTS_ALLOW_LIVE_E2E=1 to opt in." >&2
    exit 2
fi

REPO=/home/genorbox1/work/finetune-studio
SHOTS=/home/genorbox1/.openclaw/workspace/media/qa_nightly
LOG=/home/genorbox1/.openclaw/workspace/media/qa_nightly.log
WEBUI=http://fan-dragon:7860
RUN_LOG="$SHOTS/run-$(date +%Y%m%d-%H%M%S)-$$.log"

mkdir -p "$SHOTS"

echo "=== $(date -Iseconds) starting nightly QA against $WEBUI ===" >> "$LOG"

# Probe: if webui is down, abort (don't spam failures on an outage).
if ! curl -fsS --max-time 5 "$WEBUI/" >/dev/null 2>&1; then
    echo "ABORT: webui not reachable" >> "$LOG"
    exit 2
fi

cd "$REPO"
if FTS_BASE="$WEBUI" .venv/bin/python tests/e2e_ui_qa.py \
    > "$RUN_LOG" 2>&1; then
    QA_STATUS=0
else
    QA_STATUS=$?
fi
cat "$RUN_LOG" >> "$LOG"

# Summarise.
PASS=$(awk '/\[PASS\]/{n++} END{print n+0}' "$RUN_LOG")
FAIL=$(awk '/\[FAIL\]/{n++} END{print n+0}' "$RUN_LOG")
TOTAL=$((PASS + FAIL))
if [ "$TOTAL" -eq 0 ]; then
    QA_STATUS=1
fi

echo "=== $(date -Iseconds) done: $PASS pass / $FAIL fail (exit $QA_STATUS) ===" >> "$LOG"

# Push to Discord via the openclaw message tool if available.
# (Falls back to log-only if the gateway isn't reachable from cron.)
if command -v openclaw >/dev/null 2>&1; then
    MSG="🟢 QA nightly: ${PASS}/${TOTAL} PASS"
    if [ "$FAIL" -gt 0 ] || [ "$QA_STATUS" -ne 0 ]; then
        MSG="🔴 QA nightly: ${FAIL} FAIL / ${PASS} PASS (runner exit ${QA_STATUS}) — see ${LOG}"
    fi
    openclaw message send --channel discord --target "user:1484556791588065330" \
        --message "$MSG" 2>>"$LOG" || true
fi

if [ "$QA_STATUS" -ne 0 ]; then
    exit "$QA_STATUS"
fi
[ "$FAIL" -eq 0 ]
