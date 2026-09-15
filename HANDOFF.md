# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-15 ~15:40 Europe/Warsaw)
| Area | Status |
|------|--------|
| Git | `6a5e5f8` pushed; ancestors include activity `32fe913` + training observability `cc57f94` |
| Export API | ✅ Adapter-only runs merge at export via `base_model`; formats: gguf / gptq / abliterated / merged. AWQ removed (clear error) |
| Export UI | ✅ Raw completed runs selectable; “adapter only — merge at export” badge; compatible base input; no AWQ checkbox |
| Training run | ✅ Browser-started Qwen3-4B run `38867d1b` is `done`, `merge_on_save: false` — now exportable |
| Tests | ✅ `37 passed` (`test_run_export` + `test_project_export` + `test_export`); ruff clean on touched files |

## Next steps
1. On fan-dragon: `cd /home/genortg/finetune-studio && git pull --ff-only && systemctl --user restart finetune-studio`
2. Export run `38867d1b` as GGUF `q8_0` + GPTQ (+ optional abliterated) with a 16-bit Qwen3-4B base.
   `http://fan-dragon:7860/projects/04954e70/export?run=38867d1b`
3. Run the project test suite against each exported 4B format on Testing.
   `http://fan-dragon:7860/projects/04954e70/testing`
4. Verify per-case results and unload/load transitions with screenshots.
5. If activity drawer has a live item, expand it through one 2s poll; confirm expansion persists.

## Commands
- Focused: `.venv/bin/python -m pytest tests/test_run_export.py tests/test_project_export.py tests/test_export.py -q --tb=short`
- Lint: `.venv/bin/python -m ruff check src/finetune_studio/training/run_export.py src/finetune_studio/webui/routes/exports.py src/finetune_studio/webui/routes/training.py src/finetune_studio/webui/routes/project_export.py`
- Deploy: `git push`; on fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`
- Service truth check: `systemctl --user show -p MainPID --value finetune-studio; grep finetune-studio.service /proc/<pid>/cgroup`

## Blockers
- Export/test execution on fan-dragon not yet re-run after this fix.
- Activity persistence still needs a live expanded-row browser check when an activity item exists.
