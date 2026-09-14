# HANDOFF — finetune-studio

## Mission
Ships a local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`) with a data-prep tool-calling chat, training/merge/GGUF export, and an E2E browser test suite.
Genor edits on genorbox1 and pushes to GitHub; fan-dragon pulls and runs it as a bare `uvicorn` on port 7860 (RTX 3090). No systemd by default.

## State (verified 2026-09-14 19:00)
| Area | Status |
|------|--------|
| Git | `main` — file-library APIs: parsed + rename + purge (this commit) |
| File APIs | ✅ `GET .../files/{fid}/parsed`, `PATCH .../rename`, `POST .../purge` in `routes/file_library.py` + helpers in `data/fs/file_library.py` |
| Schema note | `project_files` has **no** `parsed_md` / `parsed_path` / `stored_path` / `status` — paths live on `file_versions.raw_path`; trash = `deleted_at IS NOT NULL`; conversions via `file_conversions` |
| UI wiring | ✅ data-prep PARSED tab hits `/parsed` (heuristic fallback kept); rename + per-file purge wired in `data_prep.html` + `project_data.html` |
| Pytest (genorbox1) | ✅ 66 passed (50 prior suite + 16 `test_file_library_apis`) |
| Ruff | ✅ F821/F401 clean on touched modules; new code E501-clean (`pages.py`/`projects.py` still have pre-existing E501) |
| Settings log tail / Export / RAG / Models / Phase 1–2 | ✅ Done (prior commits) |
| E2E suite | ⚠️ Stopped at Phase 3 (project `b08426e3`); Phases 3–8 pending |

## Next steps
1. Deploy: `git push` then on fan-dragon `git pull --ff-only` + restart per `RESTART.md`
2. Smoke on fan-dragon: curl the 3 new endpoints against a real project file
3. Resume E2E Phase 3 — `tests/run_qa.sh`; upload 10 `OCTOPUS-7741` fixtures to `b08426e3`
4. Backfill `training_runs.error` for the 3 pre-`daf9fa1` failed runs from `/tmp/uvicorn.log`
5. Optionally `bash scripts/install-service.sh` on fan-dragon for systemd on :7860

## Commands
- Test: `.venv/bin/python -m pytest tests/test_file_library_apis.py tests/test_project_settings.py tests/test_project_export.py tests/test_project_rag.py tests/test_project_models.py tests/test_benchmarks_compare.py tests/test_project_testing.py tests/test_project_training.py tests/test_project_data_browser.py tests/test_project_dashboard.py -q -p no:cacheprovider`
- Ruff: `.venv/bin/python -m ruff check src/finetune_studio/webui/routes/file_library.py src/finetune_studio/data/fs/file_library.py tests/test_file_library_apis.py --select F821,F401`
- Deploy: `git push`, then `ssh fan-dragon 'cd /home/genortg/finetune-studio && git pull --ff-only'`; restart per `RESTART.md` (start new → kill old pid → `ss -ltnp | grep 7860`)

## Blockers
- genorbox1 is dev-only: pure unit tests + ruff here; visual/E2E/GPU only on fan-dragon after pull + restart
- No DB migration for parsed cache — in-process `_PARSED_CACHE` only (cleared on rename/purge / process restart)
