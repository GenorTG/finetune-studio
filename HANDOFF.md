# HANDOFF — finetune-studio

## Mission
Ships a local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`) with a data-prep tool-calling chat, training/merge/GGUF export, and an E2E browser test suite.
Genor edits on genorbox1 and pushes to GitHub; fan-dragon pulls and runs it as a bare `uvicorn` on port 7860 (RTX 3090). No systemd by default.

## State (verified 2026-09-14 17:15)
| Area | Status |
|------|--------|
| Git | `main` — Phase 2 item 5 (models expand-row) ready to pull |
| Models expand-row (`/projects/{pid}/models`) | ✅ Done: click name → source run + training settings + dir contents; `GET /api/projects/{pid}/models/{path}/contents`; ▶ Open in inference |
| Benchmarks compare | ✅ Done (`291e577`) |
| Testing suite dropdown | ✅ Done (`7ec9ad4` + docs `5e16f83`) |
| Training Actions | ✅ Done (`b691173`) |
| Data-prep parsed-preview | ✅ Done (`115c127`) |
| Pytest (genorbox1) | ✅ 31 passed (`test_project_models` + prior 26) |
| Ruff (touched Python) | ✅ `--select F821,F401,E,W` clean |
| E2E suite | ⚠️ Stopped at Phase 3 (project `b08426e3`); Phases 3–8 pending |
| Data-prep PARSED tab | ⚠️ No `GET /files/{fid}/parsed` — heuristics + banner only |

## Next steps
1. **fan-dragon deploy** — `git pull --ff-only` then restart uvicorn per `RESTART.md`; verify expand-row at `http://fan-dragon:7860/projects/b08426e3/models`
2. **UI audit Phase 2 remaining** — RAG docs-indexed panel; export expand-row; settings log tail (`/api/logs`)
3. Optional: `GET /api/projects/{pid}/files/{fid}/parsed` for data-prep PARSED tab real MD
4. Resume E2E Phase 3 — `tests/run_qa.sh`; upload 10 `OCTOPUS-7741` fixtures to `b08426e3`
5. Backfill `training_runs.error` for the 3 pre-`daf9fa1` failed runs from `/tmp/uvicorn.log`
6. Wire optional APIs: `PATCH .../files/{fid}/rename` + per-file hard-purge
7. Optionally `bash scripts/install-service.sh` on fan-dragon for systemd on :7860

## Commands
- Test: `.venv/bin/python -m pytest tests/test_project_models.py tests/test_benchmarks_compare.py tests/test_project_testing.py tests/test_project_training.py tests/test_project_data_browser.py tests/test_project_dashboard.py -q -p no:cacheprovider`
- Ruff: `.venv/bin/python -m ruff check src/finetune_studio/webui/routes/pages.py src/finetune_studio/webui/routes/project_models.py tests/test_project_models.py --select F821,F401,E,W`
- Deploy: `git push`, then `ssh fan-dragon 'cd /home/genortg/finetune-studio && git pull --ff-only'`; restart per `RESTART.md` (start new → kill old pid → `ss -ltnp | grep 7860`)

## Blockers
- genorbox1 is dev-only: pure unit tests + ruff here; visual/E2E/GPU only on fan-dragon after pull + restart
- `/api/compare/load` + `/api/compare/run` load **models** for output A/B — not run-score diffs. Score compare is `GET /api/benchmarks/projects/{pid}/compare?run_a=&run_b=`
