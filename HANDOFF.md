# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-16 Europe/Warsaw)
| Area | Status |
|------|--------|
| RAG rebuild / list_sources | Fixed: build clears stale `sources/*.txt`; list/inventory from manifest/chunks only |
| RAG build status copy | Fixed: sync POST returns `building: false` + docs/chunks; UI shows Done |
| Data Prep mobile table | Fixed: ≤780px rem column floors + `#dp-uploaded-files` scroll affordance |
| Tests | 16/16 focused suite green; Ruff clean on touched Python |

## Next steps
1. Parent: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`.
2. Browser RAG `/projects/<pid>/rag`: rebuild → indexed docs == parsed sources count; no orphan/chunk duplicates.
3. Browser RAG Build: status shows Done (not Queued/Building after return).
4. Browser Data Prep at 375px: uploaded-files columns readable via horizontal scroll.
5. Fix legacy `test_db_lifecycle` / `test_parsed_converts_txt`, then `make test`.

## Commands
- Focused: `.venv/bin/python -m pytest tests/test_rag_rebuild_sources.py tests/test_rag_build_registration.py tests/test_rag_source_labels.py tests/test_live_updates.py::test_rag_build_uses_subscribe_not_1_5s_poll tests/test_live_updates.py::test_rag_build_sync_response_not_building tests/test_ui_reliability.py::test_data_prep_uploaded_files_mobile_column_floors tests/test_ui_reliability.py::test_table_fixed_layout_and_empty_colspan -v --tb=short`
- Lint: `.venv/bin/ruff check src/finetune_studio/data/rag_portable/store.py src/finetune_studio/data/rag_portable/query.py src/finetune_studio/webui/routes/rag.py src/finetune_studio/webui/routes/project_rag.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- `tests/test_db_lifecycle.py::TestSystemUpdateLifecycle::test_create_with_options` still expects dict vs JSON string.
- `tests/test_file_library_apis.py::test_parsed_converts_txt` still flaky on sibling artifacts.
- Fan-dragon visual confirm of this reliability bundle pending after deploy.
