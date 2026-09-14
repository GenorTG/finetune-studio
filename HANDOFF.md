# HANDOFF — finetune-studio

## Mission
Ships a local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`) with a data-prep tool-calling chat, training/merge/GGUF export, and an E2E browser test suite.
Genor edits on genorbox1 and pushes to GitHub; fan-dragon pulls and runs it as a bare `uvicorn` on port 7860 (RTX 3090). No systemd by default.

## State (verified 2026-09-14 18:20)
| Area | Status |
|------|--------|
| Git | `main` — Phase 2 item 7 (export expand-row) landing this commit |
| Export expand-row (`/projects/{pid}/export`) | ✅ Done: 7-col trained-exports table; click name → source run + training settings + dir contents via existing `/contents` API; `flOpenInInference` → POST `/api/models/load` → `/inference`. Radio/format/Export selected unchanged |
| RAG docs-indexed | ✅ Done (`7fcf40c` / verified `2871767`) |
| Models expand-row | ✅ Done (`6418257`) |
| Benchmarks / Testing / Training / Data-prep / Phase 1 | ✅ Done (prior commits) |
| Pytest (genorbox1) | ✅ 43 passed (38 prior + 5 `test_project_export`) |
| Ruff (touched Python) | ✅ `--select F821,F401,E,W` clean |
| Verification logs | 📝 `.tmp/audit-phase1-item{1,2,3}/REPORT.md` + `.tmp/audit-phase2-item{1,2,3,4,5,6}/REPORT.md` |
| E2E suite | ⚠️ Stopped at Phase 3 (project `b08426e3`); Phases 3–8 pending |
| Data-prep PARSED tab | ⚠️ No `GET /files/{fid}/parsed` — heuristics + banner only |

## Next steps
1. **fan-dragon verify item 7** — `git pull --ff-only`; restart per `RESTART.md`; open `http://fan-dragon:7860/projects/b08426e3/export`; expand a row
2. **UI audit Phase 2 remaining** — settings log tail panel (new `/api/logs` endpoint reading `/tmp/uvicorn.log`)
3. Optional: `GET /api/projects/{pid}/files/{fid}/parsed` for data-prep PARSED tab real MD
4. Resume E2E Phase 3 — `tests/run_qa.sh`; upload 10 `OCTOPUS-7741` fixtures to `b08426e3`
5. Backfill `training_runs.error` for the 3 pre-`daf9fa1` failed runs from `/tmp/uvicorn.log`
6. Wire optional APIs: `PATCH .../files/{fid}/rename` + per-file hard-purge
7. Optionally `bash scripts/install-service.sh` on fan-dragon for systemd on :7860

## Commands
- Test: `.venv/bin/python -m pytest tests/test_project_export.py tests/test_project_rag.py tests/test_project_models.py tests/test_benchmarks_compare.py tests/test_project_testing.py tests/test_project_training.py tests/test_project_data_browser.py tests/test_project_dashboard.py -q -p no:cacheprovider`
- Ruff: `.venv/bin/python -m ruff check src/finetune_studio/webui/routes/pages.py src/finetune_studio/webui/routes/project_export.py tests/test_project_export.py --select F821,F401,E,W`
- Deploy: `git push`, then `ssh fan-dragon 'cd /home/genortg/finetune-studio && git pull --ff-only'`; restart per `RESTART.md` (start new → kill old pid → `ss -ltnp | grep 7860`)

## Blockers
- genorbox1 is dev-only: pure unit tests + ruff here; visual/E2E/GPU only on fan-dragon after pull + restart
