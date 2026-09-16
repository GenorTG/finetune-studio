# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-16 Europe/Warsaw)
| Area | Status |
|------|--------|
| Parser extras | `.[parsers]` in `pyproject.toml` (pypdf, python-docx, openpyxl, xlrd, python-pptx, beautifulsoup4, striprtf, Pillow); folded into `.[all]`. System tools (antiword/tesseract/poppler) intentionally excluded |
| Parser coverage | `tests/test_parsers_coverage.py` parametrizes every `PARSERS` extension; text fixtures always assert content; binary/OCR dep-aware |
| RAG MIME eval | `tests/test_rag_mime_ingestion.py` builds PortableRAG with hash embedder (no HF download) + `run_rag_evaluation` |
| QA validate | Edge cases for malformed answer length, stopword-only Q, min-length boundary added |
| Tests | 70 passed; all registered MIME extensions exercised with valid fixtures; Ruff clean on touched files |
| Live production walkthrough | Fresh project `774610a3`: 43 MIME uploads, 20 stored/23 content deduplicated, 0 upload errors; 5 QA pairs accepted/exported; 80-step LoRA + merged model; training leakage eval 4/5 (80%) |
| Live RAG | Build indexed 23 documents / 42 chunks, but search currently fails: 384-d all-MiniLM vectors queried with a 1024-d fallback embedder; chat reports 0 attached corpora |
| Prior benches | Real-bench work still live on fan-dragon (`6032e2b`); unchanged this session |

## Next steps
1. Fix RAG embedder provenance/dimension validation and register the built corpus in project chat attachments; reproduce on `774610a3`.
2. Install system-only parser tools on fan-dragon when needed: `antiword`, `tesseract`, and `poppler`/`pdftotext`.
3. Run a bounded real model evaluation on fan-dragon (`num_samples=50`) and save the JSON report.
4. Fix legacy failures (`test_db_lifecycle`, `test_parsed_converts_txt`), then `make test`.

## Commands
- Parsers + RAG MIME + QA: `.venv/bin/python -m pytest tests/test_parsers_coverage.py tests/test_rag_mime_ingestion.py tests/test_prep_qa_validate.py tests/test_rag_eval.py -v --tb=short`
- Lint: `.venv/bin/ruff check tests/test_parsers_coverage.py tests/test_rag_mime_ingestion.py tests/test_prep_qa_validate.py`
- Install parser deps: `uv pip install --python .venv/bin/python -e '.[parsers]'`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- Live E2E 2026-09-16: RAG search 500s with `ValueError: ... size 1024 is different from 384`; query loader falls back when the corpus-local all-MiniLM embedder fails to load.
- Live E2E 2026-09-16: chat page shows `RAG corpora 0 attached` after RAG build; build route does not create the DB attachment consumed by chat.
- `tests/test_db_lifecycle.py::TestSystemUpdateLifecycle::test_create_with_options` expects a dict but receives the persisted JSON string.
- `tests/test_file_library_apis.py::test_parsed_converts_txt` sees a pre-existing sibling artifact and expects conversion.
- `xlwt` is dev-only and exists solely to generate the legacy `.xls` coverage fixture; it is not a runtime parser dependency.
