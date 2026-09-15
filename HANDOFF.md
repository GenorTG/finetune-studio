# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-15 Europe/Warsaw)
| Area | Status |
|------|--------|
| Git | `91d5f01` ahead of origin by 1 (tests); UI promote landed in `675fd34` on origin |
| Data prep | Per-row **Use as source** + bulk **Use text/markdown from file library** for older uploads |
| Promote API | Existing `POST …/data-prep/sources` + `promote_file_library_upload` |
| Regression | 10 focused promote tests passed; Ruff clean on touched Python |

## Next steps
1. Push tests: `git push` (commit `91d5f01`).
2. On fan-dragon: `cd /home/genortg/finetune-studio && git pull --ff-only && systemctl --user restart finetune-studio`.
3. Browser: open Data Prep with an older `.md` in the library and empty Source file → click **Use as source** (or bulk button) → picker lists it with chunk_count.
4. Confirm status line / toast is plain text (no raw JSON dump).

## Commands
- Tests: `.venv/bin/python -m pytest tests/test_data_prep_promote_ui.py tests/test_data_prep_promote.py -v --tb=short`
- Lint: `.venv/bin/python -m ruff check src/finetune_studio/webui/routes/data_prep.py tests/test_data_prep_promote_ui.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- GGUF conversion still needs llama.cpp tooling on the GPU host.
- Test commit `91d5f01` not pushed yet.
