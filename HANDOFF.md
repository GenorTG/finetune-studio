# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-15 ~16:00 Europe/Warsaw)
| Area | Status |
|------|--------|
| Git | Testing+Export fixes ready to push |
| Testing auto-load | ✅ Accepts status `done`/`completed`; requires ready `merged/` weights |
| Testing model list | ✅ Project-scoped exports from `_scan_run_models` (not global HF list); latest merged pre-selected |
| Export failures | ✅ Sync path returns HTTP 400 + `{ok:false,status:failed,error}`; UI notifies + `role=status` |
| GPTQ missing dep | ✅ Fail-fast when `auto_gptq` absent (no fake success) |
| Regression | ✅ 46 focused tests green; ruff clean on touched files |

## Next steps
1. On fan-dragon: `cd /home/genortg/finetune-studio && git pull --ff-only && systemctl --user restart finetune-studio`
2. Browser: open `http://fan-dragon:7860/projects/04954e70/testing` — model selector must list the merge-at-export path for run `598d9d14` (pre-selected); Run suite must not say “no completed training run”.
3. Browser: Export GPTQ for that run — status must show `✗ gptq: …auto_gptq…` (not “✓ Done”); toast error via `fts.notify`.
4. Optional successful path: Export `merged` (or GGUF) only, then re-check Testing selector.
5. Service truth check: `ss -ltnp | grep 7860` → `/proc/<pid>/cgroup` contains `finetune-studio.service`.

## Commands
- Focused: `.venv/bin/python -m pytest tests/test_testing_auto_load.py tests/test_testing_models.py tests/test_project_testing.py tests/test_run_export.py tests/test_export.py -q --tb=short`
- Lint: `.venv/bin/python -m ruff check src/finetune_studio/webui/testing_models.py src/finetune_studio/webui/routes/testing.py src/finetune_studio/training/run_export.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- Parent verifies browser on fan-dragon after pull/restart (no CLI substitute).
- GPTQ still needs `auto-gptq` installed on fan-dragon for a successful GPTQ export; failure is now correctly surfaced.
