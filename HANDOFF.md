# Handoff — finetune-studio

## Mission

Local fine-tune + data-prep WebUI. Current thread: close the WebUI coverage gaps in `docs/audit/UI-COVERAGE-2026-10-03.md` (private ledger) in its fix order, with regression tests.

## State (verified 2026-10-03)

| Area | State |
|---|---|
| Branch | `main`, all work pushed. |
| Source read | Complete: 212 app Python modules, 13 scripts, 37 WebUI Python modules, 28 templates, 11 static assets. See the app audit. |
| Test read | Incomplete: 8/166 `test_*.py` files in `docs/audit/TEST-AUDIT-2026-10-01.md`. Do not claim complete test review. |
| Tests | Full suite 2026-10-03 (`--ignore=tests/test_vram.py`): 1432 passed, 0 failed. |
| E2E pass 2026-10-02 | Real runs on RTX 3090 in isolated sandboxes: UI 70/70, 12 parsers, RAG, `.ftsrag` + archive round-trips, PortableRAG, LoRA Qwen3-0.6B train/eval/merge/GGUF/chat (learning proven: base invents founder, tuned answers 40/40 held-out), CLI sweep, install scripts in a clone. 16 bugs fixed with regression tests. |
| Formatter parity | Done (`docs/audit/FORMATTER-PARITY-2026-10-02.md`). `system_prompt` callers are NOT yet wired to it. |
| Dead code | Removed in lanes A-D (see git diff); `_CORPORA` constant gone from `routes/rag.py`. |
| Ruff | `ruff check src/ scripts/` clean (broad handlers narrowed; intentional boundaries carry a justified `noqa`). `tests/` still has ~108 legacy findings (deferred with test review). |
| Fixed (with tests) | Chat RAG ownership; file versions/conversions project scoping; RAG run attribution; RAG source root honors `FTS_ROOT`; CLI no-op flags removed, `fts suite` judges before scoring, `fts validate` exits nonzero; VRAM profiler init + safe cleanup; installer diagnostics (3 defects); augment holdout/training disjoint. |
| Fresh-DB browser run 2026-10-02 | Fresh instance (:7871, throwaway root), real browser: project, files, RAG (index/search/encrypted export+import), pair gen, LoRA train, merged+GGUF export, testing 95.2%, benchmark base 9.5% vs tuned 95.2%, set production, Chat with tuned model + RAG. Fixed (with tests, `tests/test_fresh_run_regressions.py`): base benchmark results hidden; `__base_model__` placeholder leaking into run lists; Chat missing the tuned model; model registry stale after export; export gate now names the files with no Q&A + UI "export anyway"; styled 404; `rag/settings` 200 before index; pair-gen status scrolls into view. Full suite 1417 passed. |
| CODEMAP | Regenerated 2026-10-03 (`make codemap`). |
| UI coverage audit 2026-10-03 | 255 API ops: ~182 used by UI, ~40 zero-caller legacy routes, ~27 real gaps; 5 dead templates; readability poor on nav/Pairs/RAG/Benchmarks/Export. Ledger: `docs/audit/UI-COVERAGE-2026-10-03.md`; lane detail + screenshots `.tmp/ui-coverage/`. |
| UI gap slice 1 (2026-10-03) | Training form fields `save_checkpoints`/`early_stopping`/`save_limit`/`eval_steps` now reach TRL (held-out 10% passed as `eval_dataset`, `EarlyStoppingCallback`, best checkpoint kept, early stop named in status); Unsloth toggle on the live form; dataset upload converts .json/.csv (Q/A, Alpaca, text columns) instead of renaming, 400 with reason otherwise; `csv_to_jsonl` now writes answers. Tests: `tests/test_training_checkpoint_eval.py`, `tests/test_dataset_upload_convert.py`. Real RTX 3090 runs (standard + Unsloth) stopped early at 35/420 and 50/420. |

## Next steps

1. ~~External-path ingestion~~ **DONE 2026-10-02 (Genor decision):** data prep never reads/writes outside the project dir; one fence `data.fs.paths.resolve_in_project`/`resolve_within`; `tests/test_data_prep_path_fence.py`. `routes/quality.py` (`/api/data/{analyze,augment,optimize,hallucination-check,convert}`) fenced too (path + `output`; optional `project_id` → project dir, else `settings.data_dir`).
2. ~~PortableRAG server exposure~~ **DONE 2026-10-02:** shipped server binds 127.0.0.1, non-loopback requires a bearer token, config layered (flags>env>`rag.config.json`>defaults), export encrypted at rest by default (AES-256-GCM, passphrase-derived key never shipped). Studio `/rag/bundle` export is now an encrypted `.ftsrag` (`secure_bundle.py`, kept in `<project>/rag-bundles/`; import stages in project dir). Tests: `tests/test_rag_encrypted_package.py`, `tests/test_rag_secure_bundle.py`.
3. ~~Project archive round-trip~~ **DONE** (lanes A-D).
4. ~~`rag_corpora` root on FTS_ROOT helpers~~ **DONE** (`rag_corpus_dir`).
5. Formatter parity done; wire `system_prompt` callers. RAG coverage claims are filename-based, not content-hash.
6. Holdout disjointness is exact-question only; reworded duplicates can still leak.
7. **E2E findings NOT fixed:** CLI has no project-dir fence (design call); `/api/data/*` relative `output` resolves to project dir but relative `path` to cwd; `rag/sources` lists raw + parsed copies (double-index?); imported-corpus hit `source` paths point at old project; engine has no resume-from-checkpoint; many endpoints return 200 `{error}`; `compare/load` missing path still shows HF repo-id text; project-flow `start_run` ignores `lora_alpha`/grad-accum overrides; `compare` vs `suite` score differ (judge); `compare --models` swallows trailing positional; `validate` accepts empty jsonl, `rag ingest` accepts binary; Windows installers diverge (no GPU pin, no run.ps1/update.ps1); `FTS_ROOT`/`FTS_DB` ignored by app (tests leak empty dirs into `~/.finetune-studio/projects`); `install.sh --repair` wipes venv. Untested: chat-with-RAG, QA mining (need LM Studio), dark/light visual pass, `rag.config.json`-only launch.
8. **UI coverage fix order** (ledger §Functional gaps): ~~1 training fields + Unsloth + upload conversion~~ DONE → ~~2 dataset health~~ DONE (`data/dataset_health.py`, training-page panel + "remove duplicates" copy; browser-verified; the old `DataQualityAnalyzer`/`TrainingDataValidator` give misleading advice — PL/EN balance, "I don't know" as risk, regex "fabricated" dates — so they were NOT wired; follow-up: move `fts analyze` + `/api/data/analyze` onto `dataset_health`, retire `/api/data/hallucination-check`) → ~~3 benchmark review~~ DONE (`?bid=` opens any result from "Recent scores", per-case verdict override, "Re-check score" audit, delete; one `_rescore_benchmark` after every verdict change — before, only the heuristic re-judge rescored and it dropped training-eval metadata; delete route always 500'd on a nonexistent `project_id` column; run/benchmark delete orphaned cases; `tests/test_benchmark_review.py`, browser-verified). Still open from 3: AI judge is env-only (`FTS_JUDGE_*`) and the run route downgrades `ai`/`local` → heuristic — belongs with the Settings slice → 4 nav + wording → 5 delete ~40 dead routes + 5 dead templates. Ask Genor first: is the terminal "hacker" chrome deliberate?
9. Finish test-file review (158 left) and tests-scope Ruff. Follow-up filed: `training/data_quality.generate_fixes` suggests nonexistent CLI commands (`tests/test_data_quality_fixes.py` is its test).

## Fresh-run UI findings (2026-10-03: all fixed, pushed)

Round 2 of the fresh-DB browser run fixed F1-F26 (files table at 390/768px, actions wrap, empty-upload message, upload list auto-poll, "unreadable" pill for corrupt files, junk csv coverage-fill pairs, UTC tooltip on pair times, local-time display, plus earlier overview/card/preset/export items) and live-file re-upload with new bytes now adds a version instead of a raw SQL error. Regression tests: `tests/test_fresh_run_regressions.py`. Re-verified in sandbox: file rename/delete/restore/purge, archive export/import/delete round-trip, HF search/info/guard, training start guard, responsive sweep (no overflow). Not exercised: GPU training failure mid-run, PortableRAG launch (done 2026-10-02), dark/light visual pass.

## Commands

- Ruff: `.venv/bin/ruff check src/ scripts/ tests/`
- Tests: `.venv/bin/python -m pytest -q -p no:cacheprovider --ignore=tests/test_vram.py` (~21 min)
- Codemap: `make codemap`

## Guardrails

- After test runs, check `~/.finetune-studio/test-fixtures/` (3 pre-existing dirs dated 2026-10-01 are not from this work).
- Never use the GTX 1070; RTX 3090 only.
- Live E2E needs `FTS_ALLOW_LIVE_E2E=1`; `tests/run_qa.sh` can contact remote services and mutate data.
- `docs/audit/*-AUDIT-*.md` are private ledgers (excluded via `.git/info/exclude`). Do not commit/push developer docs or deploy to fan-dragon until visibility and `docs/WORKPLAN.md` gates are decided.

## Blockers

Policy items resolved. Committed locally; nothing pushed without his OK.

## Idle resource release (2026-10-03)
- `FTS_IDLE_TIMEOUT` (default 300s, read live via `testing.inference.idle_timeout()`, 0 = off) unloads the engine; busy guard in `generate()` defers unload mid-run.
- `webui/app.py::_idle_reaper` (lifespan task, 60s tick): when no training/inference, past timeout calls `release_idle_memory()` (GC, CUDA cache, malloc_trim, `_PARSED_CACHE`).
- `ModelComparator.run_comparison` drops idle-unloaded engines. Tests: `tests/test_idle_release.py`.

