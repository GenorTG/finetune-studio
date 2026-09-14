# HANDOFF — finetune-studio

## Mission
Ships a local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`) with a data-prep tool-calling chat, training/merge/GGUF export, and an E2E browser test suite.
Genor edits on genorbox1 and pushes to GitHub; fan-dragon pulls and runs it as a bare `uvicorn` on port 7860 (RTX 3090). No systemd by default.

## State (verified 2026-09-14 14:00)
| Area | Status |
|------|--------|
| Git | pending push — `/projects/{pid}/data` file browser |
| Project data `/projects/{pid}/data` | ✅ Real file browser: stats (files/library/trash/storage), search/folder/MIME filters, Active/Trash/All, bulk delete+move, preview/versions/tags modals, collapsible upload, restore + empty trash. Helper: `webui/project_data_browser.py`. |
| `list_files` | ✅ Now returns `tags`, `notes`, `folder_id` (was missing vs documented contract) |
| DB migration | ✅ Additive `project_files.tags` + `notes` via `_safe_alter` (existing DBs lacked columns; tags PATCH was broken) |
| Project dashboard | ✅ Prior: stats + recent runs/models/files + activity (`249de9f`) |
| Ruff (touched Python) | ✅ `ruff check …/pages.py …/project_data_browser.py --select F821,F401,E,W` → All checks passed |
| Pytest (genorbox1) | ✅ 226 passed with HANDOFF ignores. Pre-existing fails: `test_create_with_options`, `test_latest_update_endpoint` (+ intermittent `test_list_updates_returns_recent`) |
| E2E suite | ⚠️ Stopped at Phase 3 (project `b08426e3`); Phases 3–8 pending |

## Next steps
1. On fan-dragon: `cd /home/genortg/finetune-studio && git pull --ff-only` then restart bare uvicorn per `RESTART.md` — done when `http://fan-dragon:7860/projects/b08426e3/data` shows Files/In library/In trash/Storage + files table
2. Visual check file browser at 1280×800 and 1920×1080 — done when screenshots confirm table + upload collapse + trash view
3. Optional follow-up: add `PATCH /api/projects/{pid}/files/{fid}/rename` (UI button present; notifies “API not available”) — data_prep already calls a non-existent `/rename`
4. Optional: per-file hard-purge (only bulk `trash/purge` exists; row “Purge” empties all trash)
5. Resume E2E Phase 3 — `tests/run_qa.sh`; upload 10 `OCTOPUS-7741` fixtures to `b08426e3`
6. Fix pre-existing `options_json` parse in `system_updates` — done when `pytest …::test_create_with_options` passes
7. Continue UI audit Phase 1 item #13 — `/projects/{pid}/data/{dataset_path}` data editor

## Commands
- Test (genorbox1): `.venv/bin/python -m pytest tests/ -q --tb=no -p no:cacheprovider --ignore=tests/unit/test_vram_profiler.py --ignore=tests/test_breakpoints.py --ignore=tests/test_breakpoints_visual.py --ignore=tests/test_phase_bd.py --ignore=tests/test_phase_bd_api.py`
- Ruff (touched): `.venv/bin/python -m ruff check src/finetune_studio/webui/routes/pages.py src/finetune_studio/webui/project_data_browser.py --select F821,F401,E,W`
- Deploy: `git push`, then `ssh fan-dragon 'cd /home/genortg/finetune-studio && git pull --ff-only'`; restart per `RESTART.md` (start new → kill old pid → `ss -ltnp | grep 7860`)

## Blockers
- genorbox1 is dev-only: pure unit tests + ruff here; visual/E2E/GPU only on fan-dragon after pull + restart
- User's `ruff check …/project_data.html` path is invalid (ruff parses Jinja as Python → thousands of syntax errors) — lint the Python modules instead
- No file-rename API and no per-file purge API (documented above)
