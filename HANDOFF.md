# HANDOFF — finetune-studio

Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## Mission
Make the studio show all real activity and keep training/RAG/testing quality
honest — real root-cause fixes, real fan-dragon verification, no fake greens.

## State (verified 2026-09-17 20:30 CEST)
| Area | Status |
|------|--------|
| **Activity feed** | `b573758` deployed — `collect_activity()` merges live in-memory progress with persisted history from every durable table (training, benchmark, data_prep, rag, export, hf download, system_update). Was 1 ephemeral row → now 15 real tasks on fan-dragon |
| **Operation log** | Durable `activity_events` plus HTTP middleware records every mutating API operation (upload/OCR, training, RAG, testing, benchmark, export, model/provider, settings/update); feed renders operation kind, status, project, and route |
| **Training E2E** | Fan Dragon run `43cb2fbf` via `/api/training/start`: 246 rows, 6 epochs, Unsloth, rank 64/alpha 128, LR 2e-4, batch 2 × accumulation 4, seq 2048, 168/168 steps, final_loss 0.1319; merged + Q8 artifacts; same 10-case training probe 10/10. A 2-epoch standard run `34a071ff` only scored 5/10 and final_loss 1.1286, so it is not an acceptable quality preset for this corpus. |
| **Secondary judge** | `secondary_local` loads a separate local model only after transcripts exist, judges each transcript, unloads it, and stores verdict/reasoning/confidence under `judge_input.secondary_judge`; source-grounded verdicts and audit scores remain unchanged. Live proof: benchmark `9050d067`, 1/1 secondary pass, primary audit still passed. |
| Activity dedup | Live rows win by `run_id`/`id`; live training uses the bare db run id (not the `{pid}-{id}` composite) so it collapses with its persisted row |
| Activity cap | 60-row window, **live rows never evicted** (fixed a regression where stale persisted active rows could hide a running task) |
| Activity badge | Persisted running/queued counted only if recent (<2h) or live — no forever-spinning badge from interrupted runs |
| New feed kinds | `benchmark` (✚), `export` (⇪), `system_update` (⟳) in `activity.js`; type-filter values fixed in `base.html` (`rag`→`rag_build`/`rag_ready`, added export/system_update) |
| **Test isolation** | conftest `temp_db` now patches `db.connection.settings` (not just `config.settings`) — tests were writing to the real dev DB (6.7k junk projects accumulated). Full suite **917 passed, 0 failed** |
| WebUI verified | Screenshots: benchmark + training kinds render with badges, type-filter works, expand panel shows loss/project/progress/GO-TO |
| Recovered CSS | fan-dragon-only commit `12fde89` (mid-desktop nav padding) re-applied on genorbox1 (`b3090c6`) so deploy fast-forwards clean |

## Next steps
1. WebUI quality sweep of training surface — inspect run `43cb2fbf` in `/projects/<pid>/training`, including the saved settings, loss, artifacts, and activity row.
2. RAG build/chat surface — trigger a build, confirm `rag_build`→`rag_ready` transition shows in the feed with doc/chunk counts.
3. Testing/benchmarks surface — run a suite, confirm per-case table (not raw JSON) and that the run appears as a `benchmark` activity row.
4. Prune the dev DB junk on genorbox1 if desired (6.7k test-`P`/`E`/`R` projects) — `data/finetune_studio.db` is gitignored runtime.
5. Consider reconciling stale persisted `queued`/`running` rows for exports/data_prep/rag on startup (training already does via `reconcile_stale_runs`).
6. Use the Benchmarks “Secondary transcript judge” action with a local judge path when an independent semantic review is needed; it annotates cases without replacing the source-grounded score.

## Commands
```
# activity + isolation tests
.venv/bin/python -m pytest tests/test_activity_feed.py tests/test_live_updates.py -v --tb=short
# full suite (excl GPU-only vram file)
.venv/bin/python -m pytest tests/ -q --ignore=tests/test_vram_profiler.py
# ruff (run directly; Makefile hides failures)
.venv/bin/python -m ruff check src/finetune_studio/webui/routes/activity.py
# live feed smoke
.venv/bin/python -c "from finetune_studio.webui.routes.activity import collect_activity; import collections; p=collect_activity(); print(dict(collections.Counter(t['kind'] for t in p['tasks'])))"
# deploy: push here, then on fan-dragon
ssh fan-dragon 'bash -c "cd /home/genortg/finetune-studio && git fetch origin -q && git reset --hard origin/main && systemctl --user restart finetune-studio"'
```

## Blockers
- Cursor ACP helper handshake times out ("Opening handshake has timed out") — implemented this work by hand instead. Re-provision before delegating larger changes.
- fan-dragon had drifted (detached HEAD, `main` ahead 1/behind 71). Resolved by recovering the local commit + `reset --hard origin/main`; watch for future drift.
