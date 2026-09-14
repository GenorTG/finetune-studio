# HANDOFF — finetune-studio

## Mission
Ships a local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`) with a data-prep tool-calling chat, training/merge/GGUF export, and an E2E browser test suite.
Genor edits on genorbox1 and pushes to GitHub; fan-dragon pulls and runs it as a bare `uvicorn` on port 7860 (RTX 3090). No systemd by default.

## State (verified 2026-09-14 17:50)
| Area | Status |
|------|--------|
| Git | `main` @ `7fcf40c` — Phase 2 item 6 (RAG docs-indexed) verified |
| RAG docs-indexed (`/projects/{pid}/rag`) | ✅ Done (`7fcf40c`): panel "4. Documents indexed" with 6-col list (name/mime/chunks/status/last indexed/actions); per-row Rebuild + View chunks; PortableRAG corpus read, not SQLite; SSR via `indexed_docs`. Fan-dragon pid 1049634 serves; HTTP 200 0.013s; **40 documents · 45 chunks**; 40 Rebuild + 40 View chunks buttons rendered |
| Models expand-row | ✅ Done (`6418257`) |
| Benchmarks / Testing / Training / Data-prep / Phase 1 | ✅ Done (prior commits) |
| Pytest (genorbox1) | ✅ 38 passed (31 prior + 7 `test_project_rag`) |
| Ruff (touched Python) | ✅ `--select F821,F401,E,W` clean |
| Verification logs | 📝 `.tmp/audit-phase1-item{1,2,3}/REPORT.md` + `.tmp/audit-phase2-item{1,2,3,4,5,6}/REPORT.md` |
| E2E suite | ⚠️ Stopped at Phase 3 (project `b08426e3`); Phases 3–8 pending |
| Data-prep PARSED tab | ⚠️ No `GET /files/{fid}/parsed` — heuristics + banner only |
| RAG schema caveat | ⚠️ Indexed docs live in PortableRAG corpus `~/.finetune-studio/rag_corpora/<pid>/`, NOT SQLite. Per-doc rebuild is whole-corpus scope (vectors.npy is monolithic); `doc_id` accepted for forward-compat only |
| Known viewport gap | ⚠️ 1920×1080 screenshot unreachable this session (browser `emulate` schema-rejects `width`/`height`; `window.resizeTo` blocked by Chromium) |

## Next steps
1. **UI audit Phase 2 item 7** — export expand-row at `/projects/{pid}/export` (same pattern as models expand-row: source-run info + training settings + output dir contents + Open in inference + radio-button preselect). Cursor ACP queued in this batch
2. **UI audit Phase 2 remaining** — settings log tail panel (new `/api/logs` endpoint reading `/tmp/uvicorn.log`)
3. Optional: `GET /api/projects/{pid}/files/{fid}/parsed` for data-prep PARSED tab real MD
4. Resume E2E Phase 3 — `tests/run_qa.sh`; upload 10 `OCTOPUS-7741` fixtures to `b08426e3`
5. Backfill `training_runs.error` for the 3 pre-`daf9fa1` failed runs from `/tmp/uvicorn.log`
6. Wire optional APIs: `PATCH .../files/{fid}/rename` + per-file hard-purge
7. Optionally `bash scripts/install-service.sh` on fan-dragon for systemd on :7860

## Commands
- Test: `.venv/bin/python -m pytest tests/test_project_rag.py tests/test_project_models.py tests/test_benchmarks_compare.py tests/test_project_testing.py tests/test_project_training.py tests/test_project_data_browser.py tests/test_project_dashboard.py -q -p no:cacheprovider`
- Ruff: `.venv/bin/python -m ruff check src/finetune_studio/webui/routes/pages.py src/finetune_studio/webui/routes/project_rag.py tests/test_project_rag.py --select F821,F401,E,W`
- Deploy: `git push`, then `ssh fan-dragon 'cd /home/genortg/finetune-studio && git pull --ff-only'`; restart per `RESTART.md` (start new → kill old pid → `ss -ltnp | grep 7860`)

## Blockers
- genorbox1 is dev-only: pure unit tests + ruff here; visual/E2E/GPU only on fan-dragon after pull + restart
- Per-doc rebuild is whole-corpus scope (vectors.npy is monolithic); `doc_id` is accepted for API forward-compat only
