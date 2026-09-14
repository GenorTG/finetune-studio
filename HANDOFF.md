# HANDOFF — finetune-studio

## Mission
Ships a local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`) with a data-prep tool-calling chat, training/merge/GGUF export, and an E2E browser test suite.
Genor edits on genorbox1 and pushes to GitHub; fan-dragon pulls and runs it as a bare `uvicorn` on port 7860 (RTX 3090). No systemd by default.

## State (verified 2026-09-14 13:30)
| Area | Status |
|------|--------|
| Git | `main` @ (this commit) — project overview dashboard redesign |
| Project dashboard `/projects/{pid}` | ✅ Real dashboard: stats (files/datasets/runs/models), quick actions, recent runs/models/files, activity timeline. Helper: `webui/project_dashboard.py`. Smoke TestClient 200 OK. |
| File tags PATCH | ✅ Fixed broken `fl._cursor` + restored missing `@router.patch` on folder rename; `get_file` now returns `tags`/`notes` |
| Ruff (touched Python) | ✅ `ruff check …/pages.py …/project_dashboard.py --select F821,F401,E,W` → All checks passed |
| Pytest (genorbox1) | ✅ 222 passed with HANDOFF ignores. 2 pre-existing fails unrelated (`test_create_with_options` options_json str vs dict; `test_latest_update_endpoint` 404) — fail on clean main too |
| Data-prep layout | ✅ `4bebc37` (file library visible at 1280×800 / 1920×1080) |
| Deploy docs | ✅ bare-uvicorn restart in `RESTART.md`; optional `scripts/install-service.sh` |
| E2E suite | ⚠️ Stopped at Phase 3 (project `b08426e3`); Phases 3–8 pending |
| Old training runs | ⚠️ `bdc217b1` + 2 others still show empty `error` (pre-`daf9fa1`) |

## Next steps
1. On fan-dragon: `cd /home/genortg/finetune-studio && git pull --ff-only` then restart bare uvicorn per `RESTART.md` — done when `http://fan-dragon:7860/projects/b08426e3` shows Files/Datasets/Runs/Models stats + Recent training table
2. Visual check dashboard at 1280×800 and 1920×1080 (stats row + 4 tables + activity) — done when screenshots confirm no clipped sections
3. Resume E2E Phase 3 — `tests/run_qa.sh`; upload 10 `OCTOPUS-7741` fixtures to `b08426e3` — done when JSONL exists with every row containing `OCTOPUS-7741`
4. Backfill `training_runs.error` for 3 pre-`daf9fa1` failed runs — done when `/projects/b08426e3/training?run=bdc217b1` shows real traceback
5. Fix pre-existing `options_json` parse in `system_updates` (str vs dict) — done when `pytest tests/test_db_lifecycle.py::TestSystemUpdateLifecycle::test_create_with_options` passes
6. Continue UI audit Phase 1 item #3 — `/projects/{pid}/data` real file browser (`docs/audit/UI-AUDIT-2026-09-14.md`)
7. Optional: `bash scripts/install-service.sh` on fan-dragon — done when `systemctl status finetune-studio` is active

## Commands
- Test (genorbox1): `.venv/bin/python -m pytest tests/ -q --tb=no -p no:cacheprovider --ignore=tests/unit/test_vram_profiler.py --ignore=tests/test_breakpoints.py --ignore=tests/test_breakpoints_visual.py --ignore=tests/test_phase_bd.py --ignore=tests/test_phase_bd_api.py`
- Ruff (touched): `.venv/bin/python -m ruff check src/finetune_studio/webui/routes/pages.py src/finetune_studio/webui/project_dashboard.py --select F821,F401,E,W`
- Deploy: `git push`, then `ssh fan-dragon 'cd /home/genortg/finetune-studio && git pull --ff-only'`; restart per `RESTART.md` (start new → kill old pid → `ss -ltnp | grep 7860`)

## Blockers
- genorbox1 is dev-only: pure unit tests + ruff here; visual/E2E/GPU only on fan-dragon after pull + restart
- User's `ruff check …/project.html` path is invalid (ruff parses Jinja as Python → thousands of syntax errors) — lint the Python modules instead
