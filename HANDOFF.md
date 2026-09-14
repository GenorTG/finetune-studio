# HANDOFF — finetune-studio

## Mission
Ships a local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`) with a data-prep tool-calling chat, training/merge/GGUF export, and an E2E browser test suite.
Genor edits on genorbox1 and pushes to GitHub; fan-dragon pulls and runs it as a bare `uvicorn` on port 7860 (RTX 3090). No systemd by default.

## State (verified 2026-09-14 21:46)
| Area | Status |
|------|--------|
| Git | `main` @ `8bb4084` — E2E QA failure sweep landed |
| E2E QA (`tests/run_qa.sh` → fan-dragon:7860) | ✅ **70 pass / 0 fail** (was 51/19). Fixes: memTimer, chat_v2 braces, `/api/favorites` export, palette parseQuery + data-aware asserts, aesthetic check scoped to `.card` |
| File APIs / Phase 1–2 | ✅ Done (prior commits through `83167d3`) |
| Pytest (genorbox1 Phase-2 set) | ✅ 63 passed; full suite 304 pass / 3 pre-existing update-lifecycle fails (unrelated) |
| Ruff (touched modules, F821/F401) | ✅ Clean |
| fan-dragon WebUI | ✅ Restarted; pid ≠ pre-restart; `/api/favorites` → 200 `[]` |

## Next steps
1. Optional: fix the 3 pre-existing `test_update` / `test_db_lifecycle` failures (`options_json` type + `/api/updates/latest` 404)
2. Optional: `bash scripts/install-service.sh` on fan-dragon for systemd on :7860
3. Continue product work beyond E2E Phase 3 (training/export flows) as needed

## Commands
- E2E: `bash tests/run_qa.sh` then `tail -90 ~/.openclaw/workspace/media/qa_nightly.log | grep -E 'PASS|FAIL|PASSED|FAILED'`
- Pytest Phase-2: `.venv/bin/python -m pytest tests/test_file_library_apis.py tests/test_project_settings.py tests/test_project_export.py tests/test_project_rag.py tests/test_project_models.py tests/test_benchmarks_compare.py tests/test_project_testing.py tests/test_project_training.py tests/test_project_data_browser.py tests/test_project_dashboard.py -q -p no:cacheprovider`
- Deploy: `git push`, then `ssh fan-dragon 'cd /home/genortg/finetune-studio && git pull --ff-only'`; restart per `RESTART.md` (kill old pid → start new → `ss -ltnp | grep 7860`)

## Blockers
- genorbox1 is dev-only: pure unit tests + ruff here; visual/E2E/GPU only on fan-dragon after pull + restart
- `run_qa.sh` footer `done: N pass / M fail` is cumulative across log appends — trust the suite’s `PASSED … FAILED … TOTAL 70` line
