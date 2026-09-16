# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-16 Europe/Warsaw)
| Area | Status |
|------|--------|
| Shared-models panel | Fixed: `stats()` emits flat `models`/`items` + kind/`size_human`; RAG JS `normalizeSharedModelList` reads embedders+rerankers |
| RAG raw+parsed twin | Fixed: `should_ingest_source_file` skips `files/raw/{id}_*` when `files/<sha12>/parsed.txt` exists; standalone/orphan raw still ingest |
| Tests | 15/15 focused green (`test_shared_models_stats` + source_labels + rebuild_sources); Ruff clean on touched files |

## Next steps
1. Deploy: `git push`; fan-dragon `cd /home/genortg/finetune-studio && git pull --ff-only && systemctl --user restart finetune-studio`.
2. Live project `51e2d13b`: rebuild → expect 7 docs (not 8); no opaque `*_helios_notes.txt` raw row; search one hit per logical source.
3. Browser RAG shared-models card: show embedder+reranker rows with sizes (not “No shared models cached yet.”).
4. Confirm `:7860` cgroup is `finetune-studio.service`.
5. Full local suite: `.venv/bin/python -m pytest tests/ -v --tb=short`.

## Commands
- Focused: `.venv/bin/python -m pytest tests/test_shared_models_stats.py tests/test_rag_source_labels.py tests/test_rag_rebuild_sources.py -v --tb=short`
- Lint: `.venv/bin/ruff check src/finetune_studio/data/shared_models.py src/finetune_studio/data/rag_portable/source_labels.py tests/test_shared_models_stats.py tests/test_rag_source_labels.py tests/test_rag_rebuild_sources.py`
- Live rebuild: `curl -sS -X POST http://127.0.0.1:7860/api/projects/51e2d13b/rag/rebuild -H 'content-type: application/json' -d '{"reset":true}'`

## Blockers
- Live fan-dragon rebuild/search confirm pending after deploy.
