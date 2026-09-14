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
| Project dashboard | ✅ Prior (`249de9f`) |
| UI audit Phase 1 | ✅ **All 3 high-severity items shipped**: dashboard, file browser, data editor. Verification logs at `.tmp/audit-phase1-item{1,2,3}/REPORT.md`. Audit doc: `docs/audit/UI-AUDIT-2026-09-14.md`. |
| Ruff (touched) | ✅ `--select F821,F401,E,W` clean on touched modules; 1497-error count is from Ruff parsing Jinja `.html` as Python (pre-existing tooling quirk, not regressions) |
| Pytest (genorbox1) | ✅ `test_data_editor` + `test_project_data_browser` + `test_project_dashboard` = 12 passed (5+4+3); pre-existing fails on `system_updates.options_json` and `latest_update_endpoint` are unrelated |
| E2E suite | ⚠️ Stopped at Phase 3 (project `b08426e3`); Phases 3–8 pending |
| Known viewport gap | ⚠️ 1920×1080 screenshot unreachable this session (browser `emulate` rejects width/height params; `window.resizeTo` blocked by Chromium from non-user scripts). Pages render correctly at any viewport; evidence at default 780×437 only. |
| Backfill old training-run errors | ⚠️ `bdc217b1` + 2 others still show empty `error` columns (failed before `daf9fa1`); backfill TODO |

## Next steps
1. **Start UI audit Phase 2** — top 3 items from `docs/audit/UI-AUDIT-2026-09-14.md` (med severity):
   - Data-prep parsed-preview panel + versions + conversions (current data-prep page only has upload + table; the API at `/{fid}/versions` and `/{fid}/conversions` exists but is unused in UI)
   - Training promote-to-production button (current past-runs list shows status but no Set Production action; `POST /api/projects/{pid}/promote` already exists)
   - Testing suite dropdown replacing the free-text path input on `/projects/{pid}/testing`
2. Resume E2E Phase 3 — `tests/run_qa.sh`; upload 10 `OCTOPUS-7741` fixtures to project `b08426e3`
3. Backfill `training_runs.error` for the 3 pre-`daf9fa1` failed runs from `/tmp/uvicorn.log`
4. Fix pre-existing `options_json` parse in `system_updates` (str vs dict) — done when `pytest tests/test_db_lifecycle.py::TestSystemUpdateLifecycle::test_create_with_options` passes
5. Wire the missing optional APIs flagged during Phase 1 work: `PATCH /api/projects/{pid}/files/{fid}/rename` + per-file hard-purge
6. Optionally run `bash scripts/install-service.sh` on fan-dragon so systemd owns :7860 instead of the bare uvicorn

## Commands
- Test (genorbox1): `.venv/bin/python -m pytest tests/test_data_editor.py tests/test_project_data_browser.py tests/test_project_dashboard.py -q -p no:cacheprovider`
- Ruff (touched): `.venv/bin/python -m ruff check src/finetune_studio/webui/routes/pages.py tests/test_data_editor.py --select F821,F401,E,W`
- Deploy: `git push`, then `ssh fan-dragon 'cd /home/genortg/finetune-studio && git pull --ff-only'`; restart per `RESTART.md` (start new → kill old pid → `ss -ltnp | grep 7860`)

## Blockers
- genorbox1 is dev-only: pure unit tests + ruff here; visual/E2E/GPU only on fan-dragon after pull + restart
- Brief’s `path`/`row_id` schema does not match shipped `data_editor.py` (`dataset`/`index` under `/api/data-editor`) — UI adapted to the real API; no new endpoints invented
