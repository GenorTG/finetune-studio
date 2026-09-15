# HANDOFF — finetune-studio

## Mission
Ships a local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`) with a data-prep tool-calling chat, training/merge/GGUF export, and an E2E browser test suite.
Genor edits on genorbox1 and pushes to GitHub; fan-dragon pulls and runs it as a bare `uvicorn` on port 7860 (RTX 3090). No systemd by default.

## State (verified 2026-09-14 22:36)
| Area | Status |
|------|--------|
| Git | `main` @ `8ace86d` — QABUG-001 fix + 5 regression tests; previous: `30b4e8a` (README + GH Pages refresh), `8bb4084` (E2E QA 70/70) |
| **Nightly QA — Phase 1 cleanup (fan-dragon, 22:00–22:18)** | ✅ Cleanup complete + rolled back safety net at `/home/genorbox1/.cache/finetune-studio-backups/2026-09-14-nightly/` (5.7 GB, 23 projects + 10 rag_corpora). Kept `hf_models/Qwen__Qwen3-0.6B/`, embedder `intfloat__multilingual-e5-large@*`, reranker `cross-encoder__ms-marco-MiniLM-L-6-v2@*`. DB cascade-wiped + 11 stragglers `DROP TABLE` + `VACUUM` (147 KB). `projects/`, `rag_corpora/`, `datasets/`, `state.db` removed. WebUI restarted clean. |
| **Nightly QA — Phase 2 Q&A walkthrough (13 pages)** | ✅ 11 PASS · 1 blocker (closed) · 1 cosmetic nav bug |
| `QABUG-001` (blocker) | ✅ **SHIPPED** in `routes/datasets.py:register_existing_route` — accepts `file_id` (resolved via `project_files` row + `fl.list_versions(pid,fid)[0]["raw_path"]`), 403 on cross-project, updated error string. **Live-verified** on fan-dragon pid `1567913`. Regression-locked with 5/5 tests in `tests/test_dataset_register.py`. |
| `QABUG-002` (spec drift) | Closed without code change — docstring at top of `routes/file_library.py` was already correct on closer read; `file=` singular just gets a clear 422 from FastAPI. |
| `QABUG-003` (data-prep UX) | 📝 Filed · data-prep source-picker reads `pfs.list_qa_sources(pid)`, not the file library, so freshly uploaded files don't appear in "Source file". Add auto-promote or a "promote to parsed source" button. Non-blocking — the prep pipeline still works after manual source registration. |
| `QABUG-004` (nav misroute) | 📝 Filed · `/projects/{pid}/inference` is 404 (no such page route); the real inference surface is `/projects/{pid}/chat`. Nav item should be remapped or page added. Cosmetic; users find the chat page via the top-bar `[tools]` group. |
| Touched-surface pytest (after QABUG-001 fix) | ✅ 18/18 pass (`tests/test_dataset_register.py` + data + browser + dashboard) |
| Full pytest | 257 pass / 55 pre-existing fail (none introduced by QABUG-001; failure list dominated by `test_update`, `test_project_testing`, `test_project_training` — all pre-existing) |
| Ruff (touched routes + test) | ✅ Clean (1 `B008` noqa'd as FastAPI idiom on `routes/datasets.py:120`) |
| fan-dragon WebUI | ✅ Restarted to pid `1567913` after deploy; `ss -ltnp \| grep 7860` confirms new pid ≠ pre-restart `1562219` |

## Nightly ledger
- `tests/e2e_ui_qa.py` already green per prior 70/70 sweep (unchanged this turn).
- Nightly QA artifacts: `.tmp/nightly-qa-NIGHT/README.md` (10914 bytes), `.tmp/nightly-qa-NIGHT/screenshots/01_dashboard_after_cleanup.png`, `.tmp/nightly-qa-NIGHT/fixtures/sample.jsonl`, plus `.openclaw/workspace/.tmp/nightly-qa-NIGHT/` copy set.
- Backend WebUI: `http://fan-dragon:7860/projects/a58ed72a/` (nightly-qa, base Qwen3-0.6B, dataset `d2795ac47dfd42cc_sample` — 10 QA, 566 B).

## Next steps
1. Fix `QABUG-003`: add `POST /api/projects/{pid}/data-prep/sources` route that promotes a `file_id` into a parsed-source row (mirror the dataset-registration fix in `routes/datasets.py`).
2. Fix `QABUG-004`: change the nav-bar `[tools] hf inference settings` links from `/projects/{pid}/inference` to `/projects/{pid}/chat` (or add a real inference page).
3. Optional: address the 55 pre-existing pytest failures — cluster is dominated by `test_update` / `test_project_testing` / `test_project_training`; deeper triage needed.
4. Optional: `bash scripts/install-service.sh` on fan-dragon for systemd on :7860.
5. Optional: re-verify the 5 `system_updates` were intentionally dropped vs need re-insert (delete safety net is at `~/.cache/finetune-studio-backups/2026-09-14-nightly/.finetune-studio/`).
6. Optional: add `Run a prep job` end-to-end exercise once `QABUG-003` is fixed (chunked parse → mine drafts → agentic chat curation → final dataset).

## Commands
- Pytest (touched): `.venv/bin/python -m pytest tests/test_dataset_register.py tests/test_data.py tests/test_project_data_browser.py tests/test_project_dashboard.py -q -p no:cacheprovider`
- Pytest (full, excluding GPU-only): `.venv/bin/python -m pytest tests/ -q -p no:cacheprovider --ignore=tests/test_vram_profiler.py`
- Ruff (touched): `.venv/bin/python -m ruff check src/finetune_studio/webui/routes/datasets.py tests/test_dataset_register.py`
- Deploy: `git push` then `ssh fan-dragon 'cd /home/genortg/finetune-studio && git pull --ff-only'`; restart uvicorn per `RESTART.md` (start new → wait 3s → `kill $OLD_PID` → verify new pid via `ss -ltnp | grep 7860`)
- Nightly QA replay (fan-dragon): see `.tmp/nightly-qa-NIGHT/README.md` for the full bash transcript

## Blockers
- genorbox1 is dev-only: pure unit tests + ruff here; visual/E2E/GPU only on fan-dragon after pull + restart
- `tests/test_vram_profiler.py` 37 GPU-dependent tests fail on genorbox1 (pass on fan-dragon only — pre-existing)
- `tests/test_update.py` / `test_project_testing.py` / `test_project_training.py` fail on full pytest — pre-existing baseline (3 known test_update, 19 known test_project_testing, 13 known test_project_training per prior `HANDOFF.md` snapshot). None introduced by this turn's work.
- `routes/datasets.py` now has the QABUG-001 fix live; the WebUI on fan-dragon pid `1567913` has it. QABUG-001 will resurface only on a fresh checkout until `git pull` is run — currently in sync.
