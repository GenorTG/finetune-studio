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
| Tests | 12/12: `test_sentence_transformer_local` + `test_rag_embedder_dims` |
| Live | Need fan-dragon pull + restart; search on existing MiniLM corpus should work without rebuild |

## Next steps
1. Deploy: `git push`; fan-dragon `cd /home/genortg/finetune-studio && git pull --ff-only && systemctl --user restart finetune-studio`
2. Verify RAG search on `b6db7c96` (or rebuild once to refresh shared cache): `GET /api/projects/<pid>/rag/search?q=…`
3. Confirm chat attaches corpus when `manifest.json`+`vectors.npy` present.
4. Install system parsers on fan-dragon if needed: `antiword`, `tesseract`, `poppler`.
5. Fix legacy `test_db_lifecycle` / `test_parsed_converts_txt`, then `make test`.

## Commands
- Focused: `.venv/bin/python -m pytest tests/test_sentence_transformer_local.py tests/test_rag_embedder_dims.py -v --tb=short`
- Lint: `.venv/bin/ruff check src/finetune_studio/data/sentence_transformer_local.py src/finetune_studio/data/shared_models.py src/finetune_studio/data/rag_portable/embedders.py tests/test_sentence_transformer_local.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- Live E2E on fan-dragon not re-verified after this ST-local fix (genorbox1 has no production corpus).
- `tests/test_db_lifecycle.py::TestSystemUpdateLifecycle::test_create_with_options` still expects dict vs JSON string.
- `tests/test_file_library_apis.py::test_parsed_converts_txt` still flaky on sibling artifacts.
