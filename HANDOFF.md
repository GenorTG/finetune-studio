# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-16 Europe/Warsaw)
| Area | Status |
|------|--------|
| Header nav overflow | Fixed: `.sb-nav-shell` + scroll buttons + `[nav]` menu; `app.css?v=22`, `nav_overflow.js?v=1`, `spa.js?v=15` |
| RAG docs Actions | Fixed rem Actions col + `.rag-doc-actions` (no flex-on-td) |
| Bench sprites idle | `READY TO RUN` (not COMPUTING SCORES); `sprites.js?v=14` |
| Training base filter | `models_for_training` / `?for_training=1`; GGUF/GPTQ excluded |
| Tests | 32/32 focused nav+UI suite; Ruff clean on touched Python |

## Next steps
1. Parent: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`.
2. Browser @1260px + 640px: open a project page — Testing/Benchmarks/Chat/Export/[tools] reachable via strip scroll or `[nav]`; no page-level horiz overflow from header.
3. Browser: active tab scrolls into view; `[nav]` lists same routes; SPA data-link still works.
4. Browser RAG docs: Rebuild + View chunks fully visible.
5. Fix legacy `test_db_lifecycle` / `test_parsed_converts_txt`, then `make test`.

## Commands
- Focused: `.venv/bin/python -m pytest tests/test_header_nav.py tests/test_ui_reliability.py tests/test_nav_routes.py -v --tb=short`
- Lint: `.venv/bin/ruff check tests/test_header_nav.py tests/test_ui_reliability.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- `tests/test_db_lifecycle.py::TestSystemUpdateLifecycle::test_create_with_options` still expects dict vs JSON string.
- `tests/test_file_library_apis.py::test_parsed_converts_txt` still flaky on sibling artifacts.
- Fan-dragon visual confirm of header `[nav]` / scroll affordance pending after deploy.
