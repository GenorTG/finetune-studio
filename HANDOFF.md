# Handoff — finetune-studio

## Mission

Local fine-tune + data-prep WebUI. Current thread: act on `docs/audit/APP-AUDIT-2026-10-02.md` (fix verified bugs with regression tests, keep lint honest).

## State (verified 2026-10-02)

| Area | State |
|---|---|
| Branch | `main` at `e8c358f`. All audit-fix work is **local/uncommitted**; nothing pushed. New test files are `git add -N` (intent-to-add) so `tests/test_repo_hygiene.py` passes. |
| Source read | Complete: 212 app Python modules, 13 scripts, 37 WebUI Python modules, 28 templates, 11 static assets. See the app audit. |
| Test read | Incomplete: 8/166 `test_*.py` files in `docs/audit/TEST-AUDIT-2026-10-01.md`. Do not claim complete test review. |
| Tests | Full suite after lanes A-D (`--ignore=tests/test_vram.py`): 1323 passed, 2 failed. Both fixed and re-run green (7/7): `test_repo_hygiene` (untracked files; resolved via `git add -N`) and `test_rebuild_vectors_applies_pending_embedder_from_settings_patch` (stale test patched removed `rag_routes._CORPORA`; now patches `_corpus_dir`). No second full run yet. |
| Formatter parity | Done (`docs/audit/FORMATTER-PARITY-2026-10-02.md`). `system_prompt` callers are NOT yet wired to it. |
| Dead code | Removed in lanes A-D (see git diff); `_CORPORA` constant gone from `routes/rag.py`. |
| Ruff | `ruff check src/ scripts/` clean (broad handlers narrowed; intentional boundaries carry a justified `noqa`). `tests/` still has ~108 legacy findings (deferred with test review). |
| Fixed (with tests) | Chat RAG ownership; file versions/conversions project scoping; RAG run attribution; RAG source root honors `FTS_ROOT`; CLI no-op flags removed, `fts suite` judges before scoring, `fts validate` exits nonzero; VRAM profiler init + safe cleanup; installer diagnostics (3 defects); augment holdout/training disjoint. |
| CODEMAP | Regenerated 2026-10-02 (`make codemap`). |

## Next steps

1. ~~External-path ingestion~~ **DONE 2026-10-02 (Genor decision):** data prep never reads/writes outside the project dir; one fence `data.fs.paths.resolve_in_project`/`resolve_within`; `tests/test_data_prep_path_fence.py`. `routes/quality.py` (`/api/data/{analyze,augment,optimize,hallucination-check,convert}`) fenced too (path + `output`; optional `project_id` → project dir, else `settings.data_dir`).
2. ~~PortableRAG server exposure~~ **DONE 2026-10-02:** shipped server binds 127.0.0.1, non-loopback requires a bearer token, config layered (flags>env>`rag.config.json`>defaults), export encrypted at rest by default (AES-256-GCM, passphrase-derived key never shipped). Studio `/rag/bundle` export is now an encrypted `.ftsrag` (`secure_bundle.py`, kept in `<project>/rag-bundles/`; import stages in project dir). Tests: `tests/test_rag_encrypted_package.py`, `tests/test_rag_secure_bundle.py`.
3. ~~Project archive round-trip~~ **DONE** (lanes A-D).
4. ~~`rag_corpora` root on FTS_ROOT helpers~~ **DONE** (`rag_corpus_dir`).
5. Formatter parity done; wire `system_prompt` callers. RAG coverage claims are filename-based, not content-hash.
6. Holdout disjointness is exact-question only; reworded duplicates can still leak.
7. Finish test-file review (158 left) and tests-scope Ruff. Follow-up filed: `training/data_quality.generate_fixes` suggests nonexistent CLI commands (`tests/test_data_quality_fixes.py` is its test).

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

Open policy items 1-2 (as listed in the lane report) need Genor's decision. Nothing committed/pushed without his OK.
