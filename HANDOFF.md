# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-16 Europe/Warsaw)
| Area | Status |
|------|--------|
| PortableRAG build → DB | `POST …/rag/build` + rebuild call `db.ensure_portable_rag` so `project_rags.store_path` = corpus dir (chat attachments) |
| Embedder dims | `PortableRAG.load()` no longer falls back to DEFAULT_EMBEDDER; dim mismatch / load failure raise actionable errors; search rejects query dim ≠ vectors |
| Chat retrieval | `chat_v2` uses PortableRAG when `manifest.json`+`vectors.npy` present, else Chroma VectorStore |
| Workspace nav | Model vs RAG subnav on overview/models/training/testing + RAG page; anchors `#rag-ingest` `#rag-search` `#rag-tests` |
| Tests | 9/9 focused: `test_rag_build_registration`, `test_rag_embedder_dims`, `test_workspace_nav` |
| Live reproduce | Still needed on fan-dragon project `774610a3` after pull + restart |

## Next steps
1. Deploy + verify on `774610a3`: `git pull --ff-only && systemctl --user restart finetune-studio`; then rebuild RAG and confirm chat shows ≥1 attached corpus and search works.
2. Install system-only parser tools on fan-dragon when needed: `antiword`, `tesseract`, `poppler`/`pdftotext`.
3. Run a bounded real model evaluation on fan-dragon (`num_samples=50`) and save the JSON report.
4. Fix legacy failures (`test_db_lifecycle`, `test_parsed_converts_txt`), then `make test`.

## Commands
- Focused: `.venv/bin/python -m pytest tests/test_rag_build_registration.py tests/test_rag_embedder_dims.py tests/test_workspace_nav.py -v --tb=short`
- Lint touched: `.venv/bin/ruff check src/finetune_studio/db/rags.py src/finetune_studio/webui/routes/rag.py src/finetune_studio/webui/routes/project_rag.py src/finetune_studio/webui/routes/chat_v2.py src/finetune_studio/data/rag_portable/store.py src/finetune_studio/data/rag_portable/query.py tests/test_rag_build_registration.py tests/test_rag_embedder_dims.py tests/test_workspace_nav.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- Live E2E on `774610a3` not re-verified after this patch (genorbox1 has no GPU/corpus).
- `tests/test_db_lifecycle.py::TestSystemUpdateLifecycle::test_create_with_options` still expects dict vs JSON string.
- `tests/test_file_library_apis.py::test_parsed_converts_txt` still flaky on sibling artifacts.
