# HANDOFF — finetune-studio

Local fine-tune + data-prep WebUI (`src/finetune_studio/`).  
Edit on **genorbox1** → push → **fan-dragon** runs `finetune-studio.service` on `:7860`.

**Read first:** `docs/PRODUCT-BRIEF.md` (what “good” means) · this file (ops state) · `AGENTS.md` (how to work).

## Mission

Ship an honest studio: every op visible in Activity, training that actually learns the ingested corpus, Q&A/RAG you can audit by reading answers — not fake greens.

## State (verified 2026-09-18 ~11:00 CEST · genorbox1 `53ddebc`)

| Area | Status |
|------|--------|
| **genorbox1 git** | `main` @ `53ddebc` (= `origin/main`). Dirty (uncommitted): `HANDOFF.md`, `AGENTS.md`, `README.md`, `docs/PRODUCT-BRIEF.md`, `docs/README.md`, `hf_models.py` search WIP, `tests/test_activity_kind_classifier.py` |
| **fan-dragon** | Checkout **`e2e6635`** (behind genorbox1). **Service FAILED** (stop-sigterm timeout). HTTP :7860 down. ~51/62 Gi RAM used, ~16.5/24 Gi VRAM — often other apps/games; do not kill them; pause studio work if headroom is insufficient. Within the studio, load/unload models yourself. |
| **Activity feed** | Classifier covers model_load / inference / rag_build / download / etc. (`e2e6635`); probe saw HF download API 200 but Hub **401** on bad/tokenless pulls |
| **Event loop** | Blocking GPU/infer paths wrapped in `asyncio.to_thread` + locks (ac84053…00d4a70) |
| **Startup reconcile** | Stale `queued`/`running` → `failed` on 5 tables (`cf6e703`) |
| **Quality (last solid numbers)** | Augmented run `8b1dd006`: RAG ~93–100%; held-out ~17–26%; source-disjoint once **0%**. Earlier fresh Q8 held-out ~4.3% with ID/fact gaps |
| **Suites / RAG UI** | full-ingested-corpus discovery + RAG-grounded testing UI landed in tree; verify live after service is up |
| **Tests (genorbox1)** | Historically ~922 pass; re-run after WIP commits. GPU profiler tests skip here |
| **OpenClaw** | MiniMax tools OK if you use `timeoutSeconds`, prefer curl, no tool self-tests. Old dashboard chat `…eec0a8da…` is toxic — **use a fresh session** |

## Next steps (do in order)

1. **Bring fan-dragon WebUI back** — `systemctl --user start finetune-studio` only (do **not** kill unrelated GPU/RAM users). If start fails or VRAM/RAM is too tight for studio work: pause and report. Confirm cgroup + `curl -sI http://fan-dragon:7860/`.
2. **Align revisions** — `git push` any finished WIP; fan-dragon `fetch` + `reset --hard origin/main` + restart; note both SHAs.
3. **Smoke product path (API + 1–2 WebUI screenshots)** — OCR/upload → parse → use as source; `/api/activity` kinds; HF search page (`/hf-models`) shows results; fix empty search if still broken.
4. **RAG export round-trip** — build → export zip from WebUI (add download if missing) → re-import → query; fail loudly if pack is useless.
5. **Training quality** — only when headroom allows (else pause). Unload studio helpers/models you loaded before a big train. Longer run (≥200 optimizer steps) on merged augmented data; **held-out** (not leakage); download transcript and **read** answers; augment for dispatch/return/C-17/INC-1842 gaps if still weak.
6. **Helper GGUF** — HF download a Qwen GGUF **>4B and <27B** for data-prep/judge helper; wire/load via inference paths; leave 27B off the default helper seat.
7. **Rewrite this HANDOFF** when a milestone is verified (archive old copy under `docs/archive/`).

## Commands

```bash
# genorbox1
cd ~/work/finetune-studio
make test
.venv/bin/python -m ruff check src/

# deploy
git push
ssh fan-dragon 'bash -c "cd /home/genortg/finetune-studio && git fetch origin -q && git reset --hard origin/main && systemctl --user restart finetune-studio && sleep 5 && systemctl --user is-active finetune-studio && ss -ltnp | grep 7860"'

# truth: unit cgroup (not a squat uvicorn)
ssh fan-dragon 'bash -c "ss -ltnp | grep 7860 | grep -oP \"pid=\\K[0-9]+\" | head -1 | xargs -I{} cat /proc/{}/cgroup | grep finetune-studio"'

# activity + resource peek (observe only — do not kill foreign PIDs)
curl -s http://fan-dragon:7860/api/activity | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d.get(\"tasks\",[])),\"tasks\")"
ssh fan-dragon 'bash -c "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv; free -h | head -2"'
```

## Blockers

- **fan-dragon WebUI down** (`finetune-studio.service` failed) — restart the **service** only; do not clear RAM/VRAM by killing games/other services. If resources stay insufficient for studio ops → pause.
- Uncommitted HF search / AGENTS / activity-classifier test on genorbox1 — finish or stash before hard reset on fan-dragon.

## Fresh-session kickoff (paste for MiniMax)

You are on finetune-studio. Read `docs/PRODUCT-BRIEF.md` + this `HANDOFF.md` + `AGENTS.md`. Do **not** compact, do **not** restart OpenClaw gateway, do **not** self-test OpenClaw tools. Prefer `curl` to fan-dragon; use `timeoutSeconds` on exec. `get_goal` before `create_goal`. Status ≤8 lines then execute Next steps from #1 without stopping for another essay. Work in this session (edits + checks); use Cursor ACP only if Genor asks or a change is large — if ACP handshake fails once, hand-edit or report once. On fan-dragon: never kill foreign GPU/RAM users (games/other services); pause if the box is too full. Manage Finetune Studio load/unload yourself.
