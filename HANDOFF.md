# HANDOFF — finetune-studio

## Mission
Self-hosted local-LLM workshop for document prep, RAG, fine-tuning, evaluation, and chat.
Correctness is judged from actual data flow and reviewed transcripts, not optimistic UI or auto-scores.

## State (verified 2026-10-01)
| Area | State |
|---|---|
| Branch | `main` at `cffbe04`, pushed. History was rewritten 2026-10-01 to purge `.openclaw/trajectory-exports/` (leaked internal session data); old clones/forks must re-clone. |
| Audit | Every app module, test, script and `app.css` read; per-module docs in `docs/modules/*.md`, overview in `docs/ARCHITECTURE.md` + `docs/DEVELOPER.md`. Dead code deleted (parsers/prep/unsloth_engine/compare/samplers/tool_calling/cli, legacy benchmark class hierarchy). |
| Fixes | Multi-quant export DB rows; quality routes (augment/optimize/hallucination/convert); project delete now removes files on disk; run_benchmark unloads shared engine first (GH-AAA); RAG clear/remove_source; export ownership checks; upload traversal + awaited writes; DOCX tables; coverage_fill; purge_trash; suite pass_rate; create_case columns; UI/installer/QA-runner fixes. |
| Pipeline proof | Run `ff471547`: loss 0.0842, benchmark 91.1% pass / 91.9 weighted. |
| Verification | Full pytest: 1250 passed; sole failure was the untracked-files hygiene test, which passes after commit. `git diff --check` clean. |
| Visibility | Repo is public, including developer docs. |

## Known issues
1. Three overlapping export-list routes.
2. Orphaned `/rags/*` and `/api/compare/*` routes; legacy `rag/` package only live via those plus the `chat_v2` fallback.
3. Duplicate RAG settings fields in `config.py`.
4. `ruff check src/` still has ~129 pre-existing findings (mostly facade F401 re-exports); no blind autofix.
5. Local branches `backup/pre-history-rewrite` and `openclaw/fix-gptqmodel-...` still hold the pre-purge history (local only; delete when no longer needed).

## Rules
- After every test run, delete leftover artifacts (exported GGUFs, test projects, output runs). Fixtures kept in `~/.finetune-studio/test-fixtures/`.
- GTX 1070 is never used; RTX 3090 only.
- Live E2E needs `FTS_ALLOW_LIVE_E2E=1`; the external E2E runner sends Discord notifications, so don't invoke it locally.

## Commands
- Full: `.venv/bin/python -m pytest -q -p no:cacheprovider` (~16 min)
- Lint: `.venv/bin/ruff check <changed-python-files>`
