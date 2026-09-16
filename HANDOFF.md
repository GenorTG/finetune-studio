# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-16 Europe/Warsaw)
| Area | Status |
|------|--------|
| Header @640 | Session-bar contained; tab gutter `padding-right`; ≤700px stacks `.sb-right`; labels ≥13px |
| Data Prep card-head | `.card-head` stacks column below 700px |
| Workflow explainer | `.rag-workflow-steps` always `flex-direction: column` + `nowrap` (`app.css?v=20`) |
| Model vs RAG IA | Unchanged — workflow CTA + sections 1–8 |
| Tests | 21/21 `test_ui_reliability.py`; Ruff clean |

## Next steps
1. Parent: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`.
2. Browser @1280 Data Prep: `#dp-workflow ol.rag-workflow-steps` stacks 1–4 vertically; markers not colliding.
3. Browser @640: header still no horizontal page overflow; Uploaded files card-head stacked.
4. Fix legacy `test_db_lifecycle` / `test_parsed_converts_txt`, then `make test`.

## Commands
- Focused: `.venv/bin/python -m pytest tests/test_ui_reliability.py -v --tb=short`
- Lint: `.venv/bin/ruff check tests/test_ui_reliability.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- `tests/test_db_lifecycle.py::TestSystemUpdateLifecycle::test_create_with_options` still expects dict vs JSON string.
- `tests/test_file_library_apis.py::test_parsed_converts_txt` still flaky on sibling artifacts.
- Fan-dragon visual confirm of workflow @1280 and header @640 still pending after deploy.
