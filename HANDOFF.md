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
| Tests | 63 passed, 1 skipped (`.xls` needs xlwt to build fixture); ruff clean on new tests |
| Prior benches | Real-bench work still live on fan-dragon (`6032e2b`); unchanged this session |

## Next steps
1. Deploy parsers reliability: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`.
2. Optional on fan-dragon: `uv pip install --python .venv/bin/python -e '.[parsers]'` (or `.[all]`) so PDF/DOCX/XLSX ingest has Python deps.
3. Run a bounded real model evaluation on fan-dragon (`num_samples=50`) and save the JSON report.
4. Fix legacy failures (`test_db_lifecycle`, `test_parsed_converts_txt`), then `make test`.

## Commands
- Parsers + RAG MIME + QA: `.venv/bin/python -m pytest tests/test_parsers_coverage.py tests/test_rag_mime_ingestion.py tests/test_prep_qa_validate.py -v --tb=short`
- Lint: `.venv/bin/ruff check tests/test_parsers_coverage.py tests/test_rag_mime_ingestion.py tests/test_prep_qa_validate.py`
- Install parser deps: `uv pip install --python .venv/bin/python -e '.[parsers]'`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- `tests/test_db_lifecycle.py::TestSystemUpdateLifecycle::test_create_with_options` expects a dict but receives the persisted JSON string.
- `tests/test_file_library_apis.py::test_parsed_converts_txt` sees a pre-existing sibling artifact and expects conversion.
- `.xls` happy-path fixture needs `xlwt` (not a runtime parser dep); coverage still hits the missing-`xlrd` warning path when xlrd is absent.
