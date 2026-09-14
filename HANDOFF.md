# HANDOFF — finetune-studio

## Mission
Ships a local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`) with a data-prep tool-calling chat, training/merge/GGUF export, and an E2E browser test suite.
Genor edits on genorbox1 and pushes to GitHub; fan-dragon pulls and runs it as a bare `uvicorn` on port 7860 (RTX 3090). No systemd by default.

## State (verified 2026-09-14 18:35)
| Area | Status |
|------|--------|
| Git | `main` — Phase 2 item 8 (settings log tail) just landed; pull + restart on fan-dragon |
| Settings log tail (`/projects/{pid}/settings`) | ✅ Done: `GET /api/projects/{pid}/logs?lines=N` reads `/tmp/uvicorn.log` (+ fallbacks); card with Refresh + 5s Auto-refresh; project form kept |
| Export expand-row | ✅ Done (`bfc1772` / verified `8c90fee`) |
| RAG docs-indexed | ✅ Done (`7fcf40c` / verified `2871767`) |
| Models expand-row | ✅ Done (`6418257`) |
| Benchmarks / Testing / Training / Data-prep / Phase 1 | ✅ Done (prior commits) |
| Pytest (genorbox1) | ✅ 47 passed (43 prior + 4 `test_project_settings`) |
| Ruff (touched Python) | ✅ `--select F821,F401,E,W` clean |
| E2E suite | ⚠️ Stopped at Phase 3 (project `b08426e3`); Phases 3–8 pending |
| Data-prep PARSED tab | ⚠️ No `GET /files/{fid}/parsed` — heuristics + banner only |

## Next steps
1. On fan-dragon: `cd /home/genortg/finetune-studio && git pull --ff-only` then restart per `RESTART.md`; open `http://fan-dragon:7860/projects/b08426e3/settings` and confirm log tail shows `/tmp/uvicorn.log`
2. Optional: `GET /api/projects/{pid}/files/{fid}/parsed` for data-prep PARSED tab real MD
3. Resume E2E Phase 3 — `tests/run_qa.sh`; upload 10 `OCTOPUS-7741` fixtures to `b08426e3`
4. Backfill `training_runs.error` for the 3 pre-`daf9fa1` failed runs from `/tmp/uvicorn.log`
5. Wire optional APIs: `PATCH .../files/{fid}/rename` + per-file hard-purge
6. Optionally `bash scripts/install-service.sh` on fan-dragon for systemd on :7860

## Commands
- Test: `.venv/bin/python -m pytest tests/test_project_settings.py tests/test_project_export.py tests/test_project_rag.py tests/test_project_models.py tests/test_benchmarks_compare.py tests/test_project_testing.py tests/test_project_training.py tests/test_project_data_browser.py tests/test_project_dashboard.py -q -p no:cacheprovider`
- Ruff: `.venv/bin/python -m ruff check src/finetune_studio/webui/routes/pages.py src/finetune_studio/webui/routes/project_settings.py tests/test_project_settings.py --select F821,F401,E,W`
- Deploy: `git push`, then `ssh fan-dragon 'cd /home/genortg/finetune-studio && git pull --ff-only'`; restart per `RESTART.md` (start new → kill old pid → `ss -ltnp | grep 7860`)

## Blockers
- genorbox1 is dev-only: pure unit tests + ruff here; visual/E2E/GPU only on fan-dragon after pull + restart
