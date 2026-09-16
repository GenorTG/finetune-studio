# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-16 Europe/Warsaw)
| Area | Status |
|------|--------|
| Header @640 | Session-bar contained (`overflow-x` clip/hidden); tab gutter is `padding-right` not margin; ≤700px stacks `.sb-right` under tabs; labels ≥13px (`app.css?v=19`) |
| Data Prep card-head | `.card-head` stacks column below 700px so Uploaded-files description is not squeezed |
| Model vs RAG IA | Unchanged — workflow CTA + sections 1–8 |
| Tests | 43/43 focused UI suite; Ruff clean on `tests/test_ui_reliability.py` |

## Next steps
1. Parent: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`.
2. Browser @640: `document.documentElement.scrollWidth <= document.documentElement.clientWidth`; project tabs horizontally scrollable; Uploaded files card-head stacked.
3. Browser @780: tab labels still ≥13px; header usable.
4. Fix legacy `test_db_lifecycle` / `test_parsed_converts_txt`, then `make test`.

## Commands
- Focused: `.venv/bin/python -m pytest tests/test_ui_reliability.py tests/test_workspace_nav.py tests/test_data_prep_route.py tests/test_project_rag.py tests/test_readable_results.py -v --tb=short`
- Lint: `.venv/bin/ruff check tests/test_ui_reliability.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- `tests/test_db_lifecycle.py::TestSystemUpdateLifecycle::test_create_with_options` still expects dict vs JSON string.
- `tests/test_file_library_apis.py::test_parsed_converts_txt` still flaky on sibling artifacts.
- Fan-dragon visual confirm of 640px header still pending after deploy.
