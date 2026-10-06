# Handoff — finetune-studio

## Mission

Local fine-tune, data-prep and RAG WebUI. GPU-first across vendors; CPU only on GPU-less hosts, never a silent fallback.

## State (verified 2026-10-06)

| Area | State |
|---|---|
| Git | Local `main` is 10 commits ahead of `origin/main` (`e8844e1`); CI workflow and olefile dependency fix are committed locally (`8a1f45d`). Nothing from this batch is pushed. |
| Service | genorbox1 :7860 is active; `/` and `/docs` return 200. Latest deployment remains the prior pushed revision. RTX 3090 only; GTX 1070 is excluded by the service drop-in. |
| App proof | `.tmp/e2e-final/REPORT.md`: prior live 3090 end-to-end run passed 12/12 workflow items (upload through train/merge/export, RAG, CPU path, OOM and visual checks). |
| Recent work | Coverage-fill now requires scoped, specific questions; repo test-data leakage fixed; training failure panel/OOM hint and wizard epochs control added; Windows toolchain hints fixed; chosen-GPU-index training/merge/inference verified. |
| Verification | Full suite on the earlier tree: 1773 passed, 1 skipped. Lane-C focused run: 245 passed. Current CI-sharder + parser tests: 8 passed; Ruff, codemap and `git diff --check` clean. An older isolated clone shard run failed when its CPU install lacked `olefile`; the parser extra now declares it and a regression test pins the manifest. Overlapping clone shards make that aggregate run inconclusive. |
| GitHub | Latest pushed revision `e8844e1`: Pages build/deploy checks succeeded. Historical failed run `37373751446` had successful build+deploy; only its status-report job was cancelled. No CI workflow has run on GitHub yet. |

## In flight

- Local CI workflow is committed but has not run on GitHub; its first result requires a push.
- Isolated CI clone shard processes may still be finishing; they predate the olefile fix and must not be treated as final evidence.

## Next steps

1. Push only with Genor's OK; inspect the new GitHub Actions run and report any failures.
2. Run the full suite with `.venv/bin/python -m pytest -q -p no:cacheprovider --ignore=tests/test_vram.py` when the host is clear of other test runs.
3. Deploy only after approval; the active :7860 instance still runs the last pushed revision.
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
