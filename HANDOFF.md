# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-16 Europe/Warsaw)
| Area | Status |
|------|--------|
| `/models` mobile (live) | Fixed on **models_index.html** (not models.html): `#models-index-table-scroll` + rem column floors; Copy stays reachable |
| Data Editor Review 404 | Fixed: `_resolve_path` no longer double-prefixes `data/` + `data/projects/...`; project-scoped; rejects `..` / cross-project |
| Prior audit (8d2cf36) | models.html / bench scroll, inference mobile, journal log, bench base empty, HF Esc — still in tree |
| Tests | 10/10 focused green; Ruff clean on `data_editor.py` + `test_data_editor.py` |

## Next steps
1. Deploy when ready: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio` (do not deploy from this task).
2. Browser 375px: `/models` — horizontal scroll + Copy reachable; category filters still work.
3. Data Prep → Review in Data Editor on an exported dataset (`data/projects/{pid}/datasets/*.jsonl`) — preview loads, not 404.
4. Confirm traversal/cross-project still 400/403.

## Commands
- Focused: `.venv/bin/python -m pytest tests/test_data_editor.py tests/test_ui_reliability.py::test_models_index_mobile_table_scroll_contract tests/test_ui_reliability.py::test_models_and_bench_mobile_table_scroll_contract -v --tb=short`
- Lint: `.venv/bin/ruff check src/finetune_studio/webui/routes/data_editor.py tests/test_data_editor.py`

## Blockers
- None for this correction (browser confirm pending after deploy).
