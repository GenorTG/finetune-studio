# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-16 Europe/Warsaw)
| Area | Status |
|------|--------|
| File-library "all files" count | Fixed: API `total_count` + `_flTotalCount` (not filtered `_flFiles.length`) |
| Training sprite idle copy | Fixed: `READY TO TRAIN` + poll `/api/training/status`; SPA clears mounts (`sprites.js?v=15`) |
| Inference model info | Fixed: client GET `/api/models/info?path=`; route returns `total_layers`; POST → 405 |
| Training Start gate | Fixed: `#start-btn` disabled until dataset; honest merge size wording |
| Tests | 28/28 `test_ui_reliability` + `test_list_files_total_count_unfiltered`; Ruff clean on touched routes |

## Next steps
1. Parent: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`.
2. Browser Data Prep: with files present, "all files" badge matches table; stay correct after folder select + upload.
3. Browser Training: idle sprite says READY TO TRAIN; Start disabled until dataset picked/uploaded.
4. Browser Inference: select a model — no 405; GPU layers slider updates when layers known.
5. Fix legacy `test_db_lifecycle` / `test_parsed_converts_txt`, then `make test`.

## Commands
- Focused: `.venv/bin/python -m pytest tests/test_ui_reliability.py tests/test_file_library_apis.py::test_list_files_total_count_unfiltered -v --tb=short`
- Lint: `.venv/bin/ruff check src/finetune_studio/webui/routes/file_library.py src/finetune_studio/webui/routes/models.py tests/test_ui_reliability.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- `tests/test_db_lifecycle.py::TestSystemUpdateLifecycle::test_create_with_options` still expects dict vs JSON string.
- `tests/test_file_library_apis.py::test_parsed_converts_txt` still flaky on sibling artifacts.
- Fan-dragon visual confirm of this reliability bundle pending after deploy.
