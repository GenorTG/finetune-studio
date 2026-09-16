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
| Live RAG + 27B | Project `51e2d13b`: 16-document corpus searched successfully; 27B GGUF answered with grounded citations after `cbb3f8c` |
| RAG bundle export | UI reaches download endpoint; `include_models=true` produces a 2.2GB archive. A later `include_models=false` export still included staged embedder files — export isolation bug |
| Visual audit | At 780px, header/project nav and RAG tables are cramped; DOM reports `.sb-tabs` 1516px scroll width and overflow in `#content`/table cells |
| Tests | 11/11 `test_ui_reliability.py`; Ruff clean on changed Python |

## Next steps
1. Fix RAG bundle export isolation: `include_models=false` must never archive embedder/reranker files staged by an earlier export; test archive contents and manifest refs.
2. Run visual responsive sweep at 780/900/1280px across overview, data-prep, RAG, testing, export; fix header overflow and table column alignment, then re-screenshot.
3. Browser: upload several files on `/projects/{pid}/data-prep` — table + counters must update without full reload.
4. Browser: force a bad `/api/models/load` — toast must show failure (VRAM/RAM detail), never “Model loaded”.
5. Confirm Model vs RAG workspace switch looks obvious on overview + RAG pages.
6. Fix legacy `test_db_lifecycle` / `test_parsed_converts_txt`, then `make test`.

## Commands
- Focused: `.venv/bin/python -m pytest tests/test_ui_reliability.py -v --tb=short`
- Lint: `.venv/bin/ruff check src/finetune_studio/webui/routes/models.py tests/test_ui_reliability.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- `tests/test_db_lifecycle.py::TestSystemUpdateLifecycle::test_create_with_options` still expects dict vs JSON string.
- `tests/test_file_library_apis.py::test_parsed_converts_txt` still flaky on sibling artifacts.
- RAG export currently mutates the corpus when bundling models; a later “without models” bundle can remain ~2.2GB.
- 780px visual sweep still finds header/nav and table overflow despite functional navigation.
