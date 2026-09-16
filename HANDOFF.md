# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-16 Europe/Warsaw)
| Area | Status |
|------|--------|
| RAG docs Actions | Fixed rem Actions col + `.rag-doc-actions` (no flex-on-td); `app.css?v=21` |
| Bench sprites idle | `READY TO RUN` (not COMPUTING SCORES); progress still updates (`sprites.js?v=14`) |
| Training base filter | `models_for_training` / `?for_training=1`; GGUF/GPTQ excluded; inference unchanged |
| Tests | 40/40 focused suite; Ruff clean on touched files |

## Next steps
1. Parent: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`.
2. Browser RAG docs: Rebuild + View chunks fully visible; narrow width → horizontal scroll, no clip.
3. Browser benchmarks/testing idle: caption `READY TO RUN`; after a run, score updates.
4. Browser training: only safetensors/HF bases; empty copy if none; Chat still lists helper GGUF.
5. Fix legacy `test_db_lifecycle` / `test_parsed_converts_txt`, then `make test`.

## Commands
- Focused: `.venv/bin/python -m pytest tests/test_ui_reliability.py tests/test_registry_selector_filter.py -v --tb=short`
- Lint: `.venv/bin/ruff check src/finetune_studio/models/registry.py src/finetune_studio/webui/routes/models.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- `tests/test_db_lifecycle.py::TestSystemUpdateLifecycle::test_create_with_options` still expects dict vs JSON string.
- `tests/test_file_library_apis.py::test_parsed_converts_txt` still flaky on sibling artifacts.
- Fan-dragon visual confirm of RAG Actions / bench caption / training filter pending after deploy.
