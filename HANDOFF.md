# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-16 Europe/Warsaw)
| Area | Status |
|------|--------|
| RAG export isolation | `export_bundle` stages to tempfile; live corpus never mutated. `include_models=false` omits `embedder/`/`reranker/` even if left on disk; manifest uses `shared:…` refs |
| Responsive header/tables | `@media 780/900/1280`: shrink `.sb-tabs` margin-right via `min(…vw)`; hide conn/status at ≤780; RAG docs/hits use `.table-scroll` + fixed cols (`app.css?v=17`) |
| RAG citation names | `source_labels.py`: `parsed.txt`/`0000.txt` → `Original.pdf (parsed)`; build skips `chunks/` and non-`parsed.txt` in store dirs |
| Prior RAG/model work | Unchanged — project `51e2d13b` 27B grounded chat still the last live proof |
| Tests | 29/29: export isolation + source labels + ui_reliability + project_rag; Ruff clean on touched Python |

## Next steps
1. Parent: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio` — then re-export slim bundle on a corpus that previously staged models and confirm archive ≪ 2GB.
2. Browser visual sweep at 780/900/1280 on overview/data-prep/RAG (Playwright: `python3 tests/test_breakpoints_visual.py` against fan-dragon).
3. Rebuild one project RAG corpus so citation labels regenerate from metadata (existing corpora get pretty names via `source` path when metadata still on disk).
4. Browser: data-prep multi-file upload refresh; bad `/api/models/load` toast (still open from prior audit).
5. Fix legacy `test_db_lifecycle` / `test_parsed_converts_txt`, then `make test`.

## Commands
- Focused: `.venv/bin/python -m pytest tests/test_rag_export_isolation.py tests/test_rag_source_labels.py tests/test_ui_reliability.py tests/test_project_rag.py -v --tb=short`
- Lint: `.venv/bin/ruff check src/finetune_studio/data/rag_portable/store.py src/finetune_studio/data/rag_portable/source_labels.py src/finetune_studio/data/rag_portable/query.py src/finetune_studio/webui/routes/project_rag.py tests/test_rag_export_isolation.py tests/test_rag_source_labels.py tests/test_ui_reliability.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- `tests/test_db_lifecycle.py::TestSystemUpdateLifecycle::test_create_with_options` still expects dict vs JSON string.
- `tests/test_file_library_apis.py::test_parsed_converts_txt` still flaky on sibling artifacts.
- Fan-dragon visual confirmation of 780px header + slim-bundle size still pending (no deploy this session).
