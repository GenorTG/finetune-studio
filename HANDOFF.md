# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-16 Europe/Warsaw)
| Area | Status |
|------|--------|
| SPA shell nav | `spa.js` `syncShellNav` creates/updates/hides `#project-breadcrumb` + `#workspace-subnav` outside `#content` (fix: /projects → /projects/{pid} missing Model/RAG subnav) |
| Workspace nav | Model vs RAG subnav still in `base.html`; full-page loads unchanged |
| Prior RAG patch | dcba378 corpus registration + embedder dim safety still on main |
| Tests | 16/16: workspace_nav, breadcrumb*, spa_page_scripts |
| Live reproduce | Still needed on fan-dragon after pull + restart |

## Next steps
1. Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`.
2. Browser-verify: open `/projects`, click a project card → Model/RAG `#workspace-subnav` must appear without full reload.
3. On `774610a3`: rebuild RAG; confirm chat ≥1 attached corpus and search works.
4. Fix legacy failures (`test_db_lifecycle`, `test_parsed_converts_txt`), then `make test`.

## Commands
- Focused: `.venv/bin/python -m pytest tests/test_workspace_nav.py tests/test_breadcrumb_page_label.py tests/test_breadcrumb.py tests/test_spa_page_scripts.py -v --tb=short`
- Lint: `.venv/bin/ruff check tests/test_workspace_nav.py tests/test_breadcrumb_page_label.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- Live E2E click-path for SPA subnav not re-verified on fan-dragon (genorbox1 has no Playwright app run).
- `tests/test_db_lifecycle.py::TestSystemUpdateLifecycle::test_create_with_options` still expects dict vs JSON string.
- `tests/test_file_library_apis.py::test_parsed_converts_txt` still flaky on sibling artifacts.
