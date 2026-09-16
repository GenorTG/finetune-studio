# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-16 Europe/Warsaw)
| Area | Status |
|------|--------|
| Header/nav readability | Session-bar tabs ≥14px; workspace subnav labeled + active fill |
| Project tables | `table-layout: fixed`, ellipsis/wrap, empty `colspan` rows |
| Data-prep upload refresh | Renders file list before conversion prefetch; resets to ALL FILES |
| Model load semantics | `status=error` + `loaded=false` on failure; UI confirms `/api/inference/status` |
| RAG `saveSettings` | Defined; POSTs `/api/projects/{pid}/rag/settings` |
| Tests | 11/11 `test_ui_reliability.py`; Ruff clean on changed Python |

## Next steps
1. Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`.
2. Browser: upload several files on `/projects/{pid}/data-prep` — table + counters must update without full reload.
3. Browser: force a bad `/api/models/load` — toast must show failure (VRAM/RAM detail), never “Model loaded”.
4. Confirm Model vs RAG workspace switch looks obvious on overview + RAG pages.
5. Fix legacy `test_db_lifecycle` / `test_parsed_converts_txt`, then `make test`.

## Commands
- Focused: `.venv/bin/python -m pytest tests/test_ui_reliability.py -v --tb=short`
- Lint: `.venv/bin/ruff check src/finetune_studio/webui/routes/models.py tests/test_ui_reliability.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- `tests/test_db_lifecycle.py::TestSystemUpdateLifecycle::test_create_with_options` still expects dict vs JSON string.
- `tests/test_file_library_apis.py::test_parsed_converts_txt` still flaky on sibling artifacts.
