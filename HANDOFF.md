# HANDOFF — finetune-studio

## Mission
Ships a local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`) with a data-prep tool-calling chat, training/merge/GGUF export, and an E2E browser test suite.
Genor edits on genorbox1 and pushes to GitHub; fan-dragon pulls and runs it as a bare `uvicorn` on port 7860 (RTX 3090). No systemd by default.

## State (verified 2026-09-14 19:10)
| Area | Status |
|------|--------|
| Git | `main` @ `83167d3` — all post-Phase-2 work done |
| File APIs (`/files/{fid}/parsed`, `/rename`, `/purge`) | ✅ Done (`618ec3c` + 5 bug-fix commits through `83167d3`). Rename fallback for files without `file_versions` row fixed twice (path glob → nested-dir glob → null-guard on `ver`). Parsed API streams real converted MD; purge is per-file hard-delete |
| Settings log tail (`/projects/{pid}/settings`) | ✅ Done (`f8fe187`) |
| Phase 1 + Phase 2 (items 1–8) | ✅ Done (prior commits) |
| Training-error backfill | ✅ Already persisted by engine — `bdc217b1` (`_auto_generate_suite`), `47f072c1` (read-only triton cache), `5e09fb54` (list index out of range). No manual backfill needed |
| Pytest (genorbox1) | ✅ 47 passed across all Phase 2 + post-Phase-2 test files |
| Ruff | ✅ F821/F401 clean on touched modules |
| E2E suite | ⚠️ Stopped at Phase 3 (project `b08426e3`); Phases 3–8 pending |

## Next steps
1. Resume E2E Phase 3 — `tests/run_qa.sh`; upload 10 `OCTOPUS-7741` fixtures to `b08426e3` (fan-dragon only — needs Playwright/GPU)
2. Optionally `bash scripts/install-service.sh` on fan-dragon for systemd on :7860

## Commands
- Test: `.venv/bin/python -m pytest tests/test_file_library_apis.py tests/test_project_settings.py tests/test_project_export.py tests/test_project_rag.py tests/test_project_models.py tests/test_benchmarks_compare.py tests/test_project_testing.py tests/test_project_training.py tests/test_project_data_browser.py tests/test_project_dashboard.py -q -p no:cacheprovider`
- Ruff: `.venv/bin/python -m ruff check src/finetune_studio/webui/routes/file_library.py src/finetune_studio/data/fs/file_library.py tests/test_file_library_apis.py --select F821,F401`
- Deploy: `git push`, then `ssh fan-dragon 'cd /home/genortg/finetune-studio && git pull --ff-only'`; restart per `RESTART.md` (start new → kill old pid → `ss -ltnp | grep 7860`)

## Blockers
- genorbox1 is dev-only: pure unit tests + ruff here; visual/E2E/GPU only on fan-dragon after pull + restart
- No DB migration for parsed cache — in-process `_PARSED_CACHE` only (cleared on rename/purge / process restart)
