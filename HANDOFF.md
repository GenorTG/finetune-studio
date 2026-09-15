# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-15 Europe/Warsaw)
| Area | Status |
|------|--------|
| Review UI | Landed in `0fa134a` (`data_prep.html`: filter/stats, Approve/Reject, Export approved → Training) |
| Export registry | **Uncommitted fix** in `data_prep.py`: import `get_dataset_by_path` from `db.datasets` (was silent `AttributeError` → Training empty) |
| Training empty copy | **Uncommitted**: points at approve + export |
| Tests | **Uncommitted** `tests/test_data_prep_review_export_ui.py` (5 passed); Ruff clean on touched Python |
| Note | Working tree also has unrelated dirty files (models registry / app.js modal / training selector) — do not mix into this fix |

## Next steps
1. Commit only bridge files: `data_prep.py`, `project_training.html` (empty-state hunk), `tests/test_data_prep_review_export_ui.py`, `HANDOFF.md`, `AGENTS.md` — exclude unrelated dirt.
2. `git push`; fan-dragon: `git pull --ff-only && systemctl --user restart finetune-studio`.
3. Browser: pending → **Approve all pending** → **Export approved → Training** → Training lists dataset.
4. Confirm exported panel links: Review in Data Editor / Select in Training.

## Commands
- Tests: `.venv/bin/python -m pytest tests/test_data_prep_review_export_ui.py -v --tb=short`
- Lint: `.venv/bin/python -m ruff check src/finetune_studio/webui/routes/data_prep.py tests/test_data_prep_review_export_ui.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- GGUF conversion still needs llama.cpp tooling on the GPU host.
- Unrelated local edits in registry/models/app.js — park or separate commit.
