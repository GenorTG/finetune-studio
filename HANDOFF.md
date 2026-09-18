# HANDOFF — finetune-studio

Local fine-tune + data-prep WebUI (`src/finetune_studio/`).
Edit on **genorbox1** → push → **fan-dragon** runs `finetune-studio.service` on `:7860`.

**Read first:** `docs/WORKPLAN.md` (order is law — read FIRST) · `docs/PRODUCT-BRIEF.md` (north star) · this file · `AGENTS.md`.

## Mission

Ship an honest studio: every op visible in Activity, training that actually learns the ingested corpus, Q&A you can audit by reading answers — not fake greens.

## State (verified 2026-09-18 ~15:00 CEST · genorbox1 `463a97e` = fan-dragon `4ed1336`+docs·463a97e pending pull)

| Area | Status |
|------|--------|
| **genorbox1 git** | `main` @ `463a97e` = `origin/main`. Clean except `.tmp/` (untracked, fine). |
| **fan-dragon** | Checkout `4ed1336` (one docs commit behind `463a97e` — docs only, harmless). **Service active**, cgroup verified `finetune-studio.service`, HTTP 200, activity feed live (60 tasks). VRAM: 3.6/24 GiB used. Testing engine has GGUF `quality-v4-source-disjoint/model-q4_k_m.gguf` LOADED — unload before big train. |
| **Judged quality (source-disjoint, 52 cases)** | **Human verdict: ~8% pass strict / ~15% lenient. Auto said 35% — WRONG on 17/52 (33%)**: 13 false positives (wrong dates/owners passed), 2 false negatives, 2 severity disputes. Report: `docs/judging/2026-09-18-source-disjoint-q4.md`. Auto-scoring OVER-scores; bare-answer under-score trap matters less than hallucinated-numbers pass. |
| **HF search** | FIXED live on fan-dragon (`/api/hf/search?q=...` returns real results; token-match + pipeline-tag retry, `4ed1336`). Playwrong-path note: route prefix is `/api/hf/*`, page is `/models/explore` + `/hf-models` (not `/projects/{pid}/hf-models` — that is 404). |
| **Tests (genorbox1)** | activity-classifier 38 pass; full suite historically ~922. |
| **Codemap** | `make codemap` / `--grep NAME` / `make codemap-check` live. Commit regenerated `docs/CODEMAP.md` with code moves. |
| **Suites / RAG UI** | full-corpus + RAG-grounded testing UI landed; NOT yet smoke-verified live. |

## Next steps (do in order — WORKPLAN.md governs)

1. **Pull `463a97e` on fan-dragon** (`bash update.sh`) — docs-only diff, 1 min.
2. **Unload the testing GGUF** after any bench work (leave box clean).
3. **Fix suite bugs found while judging:** regenerate `source-disjoint-held-out.json` — case 050's expected answer is a raw table dump; dedupe overlapping cases (000/046/048, 005/044/049, 006/007/011/051).
4. **Retrain with a real run** (≥200 optimizer steps, headroom loads fine now) then re-run judging session — person/date/number hallucinations are the dominant failure classes; use source-grounded augmentation.
5. **RAG export zip round-trip** via WebUI — untested end-to-end.
6. **Auto-judge wiring** — only after trust gate in `docs/judging/PROTOCOL.md` (≥95% human agreement over 2 runs + Genor 10-case spot check).
7. **Rewrite/HANDOFF** at next milestone; keep ≤120 lines, archive old copies.

## Commands

```bash
# genorbox1
cd ~/work/finetune-studio
make test                       # focused: .venv/bin/python -m pytest tests/test_ACTIVITY.py -v
.venv/bin/python -m ruff check src/   # NEVER `make lint` (swallows failures)
make codemap                    # regenerate docs/CODEMAP.md after code moves; commit it together

# deploy (scripted route — iron rule 2b)
git push
ssh fan-dragon 'bash -c "cd /home/genortg/finetune-studio && bash update.sh 2>&1 | tail -15"'
ssh fan-dragon 'bash -c "cd /home/genortg/finetune-studio && git log --oneline -1; systemctl --user is-active finetune-studio; ss -ltnp | grep 7860"'

# service truth (squat-uvicorn check)
ssh fan-dragon 'bash -c "ss -ltnp | grep 7860 | grep -oP \"pid=\\K[0-9]+\" | head -1 | xargs -I{} sh -c \"grep finetune-studio /proc/{}/cgroup\""'

# API verify (browser flakes once → fall back to curl; never CDP loops)
curl -s http://fan-dragon:7860/api/activity | python3 -m json.tool | head -40
ssh fan-dragon 'bash -c "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; free -h | head -2"'
```

## Blockers

- None hard. Next milestone is quality (training + re-judge), not ops.

## Fresh-session kickoff (paste for MiniMax)

You are on finetune-studio. Read `docs/WORKPLAN.md` FIRST (order is law), then `docs/PRODUCT-BRIEF.md` + this `HANDOFF.md` + `AGENTS.md`. Do **not** compact, do **not** restart OpenClaw gateway, do **not** self-test OpenClaw tools. Deploy ONLY via `update.sh` on fan-dragon; never manual reset chains; never start service on stale checkout. Never kill foreign GPU/RAM pids. Auto-test verdicts are untrustworthy — human judging per `docs/judging/PROTOCOL.md` is mandatory. Evidence format: `DONE <change> / verified: <pasted lines> / gaps: <unchecked>`. `get_goal` before any create. Consult `docs/CODEMAP.md` instead of grepping.
