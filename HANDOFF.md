# Handoff — finetune-studio

## Mission

Local fine-tune, data-prep and RAG WebUI. GPU-first across vendors; CPU only on GPU-less hosts, never a silent fallback.

## State (verified 2026-10-06)

| Area | State |
|---|---|
| Git | Local `main` is 9 commits ahead of `origin/main` (`e8844e1`); current worktree also has an olefile dependency fix and a new CI workflow pending commit. Nothing from this batch is pushed. |
| Service | genorbox1 :7860 is active; `/` and `/docs` return 200. Latest deployment remains the prior pushed revision. RTX 3090 only; GTX 1070 is excluded by the service drop-in. |
| App proof | `.tmp/e2e-final/REPORT.md`: prior live 3090 end-to-end run passed 12/12 workflow items (upload through train/merge/export, RAG, CPU path, OOM and visual checks). |
| Recent work | Coverage-fill now requires scoped, specific questions; repo test-data leakage fixed; training failure panel/OOM hint and wizard epochs control added; Windows toolchain hints fixed; chosen-GPU-index training/merge/inference verified. |
| Verification | Full suite on the earlier tree: 1773 passed, 1 skipped. Lane-C focused run: 245 passed. Current parser regression: 3 passed; Ruff and `git diff --check` clean. A later isolated shard run had a failure because its CPU install lacked `olefile`; parser extra now declares it. That run used overlapping shard copies and is not a clean aggregate result. |
| GitHub | Latest pushed revision `e8844e1`: Pages build/deploy checks succeeded. Historical failed run `37373751446` had successful build+deploy; only its status-report job was cancelled. No CI workflow has run on GitHub yet. |

## In flight

- Local CI workflow and sharder are staged as pending work; GitHub CI result requires a push.
- Isolated CI clone shard processes may still be finishing; inspect `.tmp/lane-c/ev/ci-shard-*.log` before reusing their results.

## Next steps

1. Finish diagnosing the isolated shard failure after its run exits; rerun the affected test on the repaired parser dependency.
2. Commit the workflow and olefile manifest/test fix; run `.venv/bin/ruff check src/ scripts/ tests/test_doc_parser_olefile.py` and `make codemap-check`.
3. Push only with Genor's OK; inspect the new GitHub Actions run and report any failures.
4. Deploy only after approval; the active :7860 instance still runs the last pushed revision.

## Known issues

- AMD, Intel and Apple paths are unit/plan-tested, not proven on real hardware; multi-GPU Trainer placement was recently verified on the 3090 with the 1070 isolated.
- One 0.6B RAG/training evaluation scored 58% on its local suite and 0/3 held-out; coverage-fill question quality has since changed and those scores need a fresh rerun. RAG chat's 4/5 result is one small single-run sample, not a benchmark.
- Distractor-chunk behavior and repeated-seed grounding comparison remain unmeasured.
- fan-dragon deployment is intentionally deferred.

## Commands

- Tests: `.venv/bin/python -m pytest -q -p no:cacheprovider --ignore=tests/test_vram.py`
- Lint: `.venv/bin/ruff check src/ scripts/`
- Codemap: `make codemap-check`
- GPU: `.venv/bin/fts accel`; plan: `bash install.sh --plan`

## Blockers

- Push/deploy awaits Genor's OK.
