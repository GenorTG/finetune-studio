# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-16 Europe/Warsaw)
| Area | Status |
|------|--------|
| Model vs RAG IA | RAG: Upload→Parse→Prepare/QA→Embed→Test explainer + Data Prep CTA; sections 1–8 unambiguous; no upload claim on RAG; external memory vs weights clarified (`app.css?v=18`) |
| Data Prep layers | Visible **Uploaded files** / **Parsed sources** / **Training / Q&A output** + empty states; project Data page labeled Uploaded files + `table-scroll` |
| Header / tables | Workspace subnav 14px; 780px keeps tab labels ≥13px; `#fb-files-table` / `.fl-table` min-width + scroll alignment |
| Prior RAG export | Unchanged — slim bundle isolation + citation labels still land |
| Tests | 41/41 focused UI/RAG/data-prep suite; Ruff clean on touched Python |

## Next steps
1. Parent: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`.
2. Browser: `/projects/{pid}/rag` first-load — CTA + workflow + no “Build with AI / agentic”; 780/900/1280 header + RAG/docs tables (`python3 tests/test_breakpoints_visual.py`).
3. Browser: data-prep shows three named sections; promote/upload still refreshes Parsed sources.
4. Rebuild one RAG corpus so citation labels regenerate (existing corpora OK via source path).
5. Fix legacy `test_db_lifecycle` / `test_parsed_converts_txt`, then `make test`.

## Commands
- Focused: `.venv/bin/python -m pytest tests/test_ui_reliability.py tests/test_workspace_nav.py tests/test_data_prep_route.py tests/test_project_rag.py tests/test_readable_results.py -v --tb=short`
- Lint: `.venv/bin/ruff check tests/test_ui_reliability.py tests/test_data_prep_route.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- `tests/test_db_lifecycle.py::TestSystemUpdateLifecycle::test_create_with_options` still expects dict vs JSON string.
- `tests/test_file_library_apis.py::test_parsed_converts_txt` still flaky on sibling artifacts.
- Fan-dragon visual confirmation of IA + 780px header still pending (no deploy this session).
