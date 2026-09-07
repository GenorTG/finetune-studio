#!/usr/bin/env bash
# Nightly Finetune Studio UI regression runner.
# Exercises every page + interaction, writes report + screenshots,
# and pushes a Discord notification with the verdict.

set -uo pipefail

REPO=/home/genorbox1/work/finetune-studio
SHOTS=/home/genorbox1/.openclaw/workspace/media/qa_nightly
REPORT=$SHOTS/report.json
LOG=/home/genorbox1/.openclaw/workspace/media/qa_nightly.log
WEBUI=http://fan-dragon:7860

mkdir -p "$SHOTS"

echo "=== $(date -Iseconds) starting nightly QA against $WEBUI ===" >> "$LOG"

# Probe: if webui is down, abort (don't spam failures on an outage).
if ! curl -fsS --max-time 5 "$WEBUI/" >/dev/null 2>&1; then
    echo "ABORT: webui not reachable" >> "$LOG"
    exit 2
fi

cd "$REPO"
FTS_BASE="$WEBUI" python tests/e2e_ui_qa.py \
    >> "$LOG" 2>&1 \
    || true  # don't crash on non-zero; we report from results

# Summarise.
PASS=$(grep -c '\[PASS\]' "$LOG" 2>/dev/null | tail -1)
FAIL=$(grep -c '\[FAIL\]' "$LOG" 2>/dev/null | tail -1)
TOTAL=$((PASS + FAIL))

echo "=== $(date -Iseconds) done: $PASS pass / $FAIL fail ===" >> "$LOG"

# Push to Discord via the openclaw message tool if available.
# (Falls back to log-only if the gateway isn't reachable from cron.)
if command -v openclaw >/dev/null 2>&1; then
    MSG="🟢 QA nightly: ${PASS}/${TOTAL} PASS"
    [ "$FAIL" -gt 0 ] && MSG="🔴 QA nightly: ${FAIL} FAIL / ${PASS} PASS — see ${LOG}"
    openclaw message send --channel discord --target "user:1484556791588065330" \
        --message "$MSG" 2>>"$LOG" || true
fi

exit 0
