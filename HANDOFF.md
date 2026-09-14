# HANDOFF — finetune-studio

## Mission
Ships a local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`) with a data-prep tool-calling chat, training/merge/GGUF export, and an E2E browser test suite.
Genor edits on genorbox1 and pushes to GitHub; fan-dragon pulls and runs it as a bare `uvicorn` on port 7860 (RTX 3090). No systemd by default.

## State (verified 2026-09-14 15:20)
| Area | Status |
|------|--------|
| Git | `main` — training past-runs Actions (Set production / Open in inference / Export) |
| Training Actions | ✅ Past-runs 7th column; promote confirm modal; `production_run_changed` updates header pill; export `?run=` preselects radio |
| Promote API | ✅ `POST /api/projects/{pid}/promote` → `{ok, run}` (was raw project dict) |
| Dashboard helper | ✅ `resolve_production_run` → `{id, name}` exposed as `production_run` in ctx |
| Pytest (genorbox1) | ✅ `test_project_training` + data-editor/browser/dashboard = 18 passed |
| E2E suite | ⚠️ Stopped at Phase 3 (project `b08426e3`); Phases 3–8 pending |
| Data-prep PARSED tab | ⚠️ No `GET /files/{fid}/parsed` — heuristics + banner only |
| Backfill old training-run errors | ⚠️ `bdc217b1` + 2 others still show empty `error` columns |

## Next steps
1. **fan-dragon verify** — `git pull --ff-only`; restart per `RESTART.md`; open `http://fan-dragon:7860/projects/b08426e3/training`; confirm Actions column + Set production updates pill.
2. **UI audit Phase 2 remaining** — testing suite dropdown on `/projects/{pid}/testing`.
3. Optional: add `GET /api/projects/{pid}/files/{fid}/parsed` so PARSED tab can stream real converted MD.
4. Resume E2E Phase 3 — `tests/run_qa.sh`; upload 10 `OCTOPUS-7741` fixtures to project `b08426e3`.
5. Backfill `training_runs.error` for the 3 pre-`daf9fa1` failed runs from `/tmp/uvicorn.log`.
6. Wire optional APIs: `PATCH /api/projects/{pid}/files/{fid}/rename` + per-file hard-purge.
7. Optionally `bash scripts/install-service.sh` on fan-dragon for systemd on :7860.

## Commands
- Test (genorbox1): `.venv/bin/python -m pytest tests/test_project_training.py tests/test_data_editor.py tests/test_project_data_browser.py tests/test_project_dashboard.py -q -p no:cacheprovider`
- Ruff (touched): `.venv/bin/python -m ruff check src/finetune_studio/webui/routes/pages.py src/finetune_studio/webui/project_dashboard.py tests/test_project_training.py --select F821,F401,E,W`
- Deploy: `git push`, then `ssh fan-dragon 'cd /home/genortg/finetune-studio && git pull --ff-only'`; restart per `RESTART.md` (start new → kill old pid → `ss -ltnp | grep 7860`)

## Blockers
- genorbox1 is dev-only: pure unit tests + ruff here; visual/E2E/GPU only on fan-dragon after pull + restart
- `projects.py` still has pre-existing ruff noise (E501/RUF100/S110) unrelated to this change
