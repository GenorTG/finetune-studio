# HANDOFF — finetune-studio

## Mission
Ships a local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`) with a data-prep tool-calling chat, training/merge/GGUF export, and an E2E browser test suite.
Genor edits on genorbox1 and pushes to GitHub; fan-dragon pulls and runs it as a bare `uvicorn` on port 7860 (RTX 3090). No systemd by default.

## State (verified 2026-09-14 14:30)
| Area | Status |
|------|--------|
| Git | `main` @ `24cf1a1`, in sync with `origin/main` — data editor UI |
| Data editor `/projects/{pid}/data/{dataset_path}` | ✅ Real editor: stats, filter pills (default Pending), search, bulk approve/reject, paginated review table (50/page), row modal with Save/Approve/Reject. Template: `data_editor.html`. Calls `/api/data-editor/...` |
| API contract note | Existing API uses `dataset`+`index` (not `path`+`row_id`); prefix `/api/data-editor`. Save = `PATCH /row` (single) or `POST /save` (batch). Status from `GET /review` decisions table. |
| Project data browser | ✅ Prior (`ade32f0`) |
| Ruff (touched) | ✅ `--select F821,F401,E,W` clean on `pages.py` + `test_data_editor.py` |
| Pytest (genorbox1) | ✅ `test_data_editor` + `test_project_data_browser` + `test_project_dashboard` = 12 passed |
| E2E suite | ⚠️ Stopped at Phase 3 (project `b08426e3`); Phases 3–8 pending |

## Next steps
1. On fan-dragon: `cd /home/genortg/finetune-studio && git pull --ff-only` then restart bare uvicorn per `RESTART.md` — done when `ss -ltnp | grep 7860` shows new pid
2. Open `http://fan-dragon:7860/projects/b08426e3/data/output_aethermere_hq/suite_aethermere_sharegpt.json` — done when Pending filter + 50-row table + Open modal work
3. Resume E2E Phase 3 — `tests/run_qa.sh`; upload 10 `OCTOPUS-7741` fixtures to `b08426e3`
4. Optional: `PATCH /api/projects/{pid}/files/{fid}/rename` (file-browser UI still notifies missing)
5. Optional: per-file hard-purge (only bulk `trash/purge` exists)
6. Fix pre-existing `options_json` parse — done when `pytest …::test_create_with_options` passes
7. Continue UI audit Phase 1 remaining items after visual check of data editor

## Commands
- Test (genorbox1): `.venv/bin/python -m pytest tests/test_data_editor.py tests/test_project_data_browser.py tests/test_project_dashboard.py -q -p no:cacheprovider`
- Ruff (touched): `.venv/bin/python -m ruff check src/finetune_studio/webui/routes/pages.py tests/test_data_editor.py --select F821,F401,E,W`
- Deploy: `git push`, then `ssh fan-dragon 'cd /home/genortg/finetune-studio && git pull --ff-only'`; restart per `RESTART.md` (start new → kill old pid → `ss -ltnp | grep 7860`)

## Blockers
- genorbox1 is dev-only: pure unit tests + ruff here; visual/E2E/GPU only on fan-dragon after pull + restart
- Brief’s `path`/`row_id` schema does not match shipped `data_editor.py` (`dataset`/`index` under `/api/data-editor`) — UI adapted to the real API; no new endpoints invented
