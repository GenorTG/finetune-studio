# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-16 Europe/Warsaw)
| Area | Status |
|------|--------|
| PortableRAG build → DB | `POST …/rag/build` registers `project_rags.store_path` |
| Shared embedder cache | `register()` now copies full ST tree (`1_Pooling`/`2_Normalize`); incomplete caches replaced |
| Local ST loader | `prepare_local_sentence_transformer_dir` repairs missing Pooling config (same model, no DEFAULT fallback) |
| Dim safety | `PortableRAG.load()` still rejects dim mismatch / load failure with actionable errors |
| Tests | 12/12: `test_sentence_transformer_local` + `test_rag_embedder_dims`; changed-file Ruff clean |
| Live | fan-dragon `05ee307`, systemd-owned :7860; browser search returned 4 hits and Chat showed 1 attached corpus |
| Navigation | Browser SPA click `/projects` → project overview refreshed Model/RAG workspace subnav |

## Next steps
1. Re-run live RAG search after future embedder/cache changes: browser RAG page, query `What is the Helios live verification token?` → expect `4 hits`.
2. Confirm Chat shows corpus checkbox checked and `1 attached` for `b6db7c96`.
3. Install system parsers on fan-dragon if needed: `antiword`, `tesseract`, `poppler`.
4. Fix legacy `test_db_lifecycle` / `test_parsed_converts_txt`, then `make test`.

## Commands
- Focused: `.venv/bin/python -m pytest tests/test_sentence_transformer_local.py tests/test_rag_embedder_dims.py -v --tb=short`
- Lint: `.venv/bin/ruff check src/finetune_studio/data/sentence_transformer_local.py src/finetune_studio/data/shared_models.py src/finetune_studio/data/rag_portable/embedders.py tests/test_sentence_transformer_local.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- `tests/test_db_lifecycle.py::TestSystemUpdateLifecycle::test_create_with_options` still expects dict vs JSON string.
- `tests/test_file_library_apis.py::test_parsed_converts_txt` still flaky on sibling artifacts.
