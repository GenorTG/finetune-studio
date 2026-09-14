# HANDOFF — finetune-studio

## Mission
Ships a local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`) with a data-prep tool-calling chat, training/merge/GGUF export, and an E2E browser test suite.
Genor edits on genorbox1 and pushes to GitHub; fan-dragon pulls and runs it as a bare `uvicorn` on port 7860 (RTX 3090). No systemd by default.

## State (verified 2026-09-14 15:20)
| Area | Status |
|------|--------|
| Git | `main` @ `b691173` — training past-runs Actions + data-prep parsed-preview |
| Training Actions (`/projects/{pid}/training`) | ✅ Done (`b691173`): 7-col past-runs with Set production / Open in inference / Export; production pill live-updates; `POST /promote` now `{ok, run}`; export `?run=` preselects radio |
| Data-prep parsed-preview (`/projects/{pid}/data-prep`) | ✅ Done (`115c127`): 4-tab preview modal (RAW / PARSED / VERSIONS / CONVERSIONS); 📝 icon on rows where parsed-MD is available; prefetched conversions + sources caches |
| Promote API | ✅ `POST /api/projects/{pid}/promote` → `{ok, run}` (was raw project dict) |
| Dashboard helper | ✅ `resolve_production_run` → `{id, name}` exposed as `production_run` in ctx |
| Pytest (genorbox1) | ✅ 22 passed (test_project_training + test_data_editor + test_project_data_browser + test_project_dashboard). Pre-existing fails on `system_updates.options_json` and `latest_update_endpoint` unrelated |
| Ruff (touched Python) | ✅ `--select F821,F401,E,W` clean on touched modules; 1497-error count is from Ruff parsing Jinja `.html` as Python (pre-existing tooling quirk, not regressions) |
| Verification logs | 📝 `.tmp/audit-phase1-item{1,2,3}/REPORT.md` + `.tmp/audit-phase2-item{1,2}/REPORT.md` |
| E2E suite | ⚠️ Stopped at Phase 3 (project `b08426e3`); Phases 3–8 pending |
| Data-prep PARSED tab | ⚠️ No `GET /files/{fid}/parsed` — heuristics + banner only |
| Backfill old training-run errors | ⚠️ `bdc217b1` + 2 others still show empty `error` columns |
| Known viewport gap | ⚠️ 1920×1080 screenshot unreachable this session (browser `emulate` rejects `width`/`height`; `window.resizeTo` blocked by Chromium) |

## Next steps
1. **UI audit Phase 2 item 3** — testing suite dropdown on `/projects/{pid}/testing` replacing free-text path input (in flight in this batch)
2. **UI audit Phase 2 remaining** — benchmarks compare tab (uses existing `/compare/load` + `/compare/run` API), models expand-row + Open in inference on `/projects/{pid}/models`, RAG docs-indexed panel on `/projects/{pid}/rag`, export expand-row, settings log tail panel (new `/api/logs` endpoint)
3. Optional: add `GET /api/projects/{pid}/files/{fid}/parsed` so the data-prep PARSED tab can stream real converted MD instead of the placeholder banner
4. Resume E2E Phase 3 — `tests/run_qa.sh`; upload 10 `OCTOPUS-7741` fixtures to project `b08426e3`
5. Backfill `training_runs.error` for the 3 pre-`daf9fa1` failed runs from `/tmp/uvicorn.log`
6. Wire optional APIs: `PATCH /api/projects/{pid}/files/{fid}/rename` + per-file hard-purge
7. Optionally `bash scripts/install-service.sh` on fan-dragon for systemd on :7860

## Commands
- Test (genorbox1): `.venv/bin/python -m pytest tests/test_project_training.py tests/test_data_editor.py tests/test_project_data_browser.py tests/test_project_dashboard.py -q -p no:cacheprovider`
- Ruff (touched): `.venv/bin/python -m ruff check src/finetune_studio/webui/routes/pages.py src/finetune_studio/webui/project_dashboard.py tests/test_project_training.py --select F821,F401,E,W`
- Deploy: `git push`, then `ssh fan-dragon 'cd /home/genortg/finetune-studio && git pull --ff-only'`; restart per `RESTART.md` (start new → kill old pid → `ss -ltnp | grep 7860`)

## Blockers
- genorbox1 is dev-only: pure unit tests + ruff here; visual/E2E/GPU only on fan-dragon after pull + restart
- `projects.py` still has pre-existing ruff noise (E501/RUF100/S110) unrelated to this change
