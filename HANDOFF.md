# Handoff — finetune-studio

## Mission

Local fine-tune, data-prep, and RAG WebUI. Context-grounded dataset rows are validated; the full wizard chain (helper mining → dataset → train → merge/GGUF → test/benchmark → RAG chat) was exercised live on 2026-10-05.

## State (verified 2026-10-05)

| Area | State |
|---|---|
| Git | `main` is 6 local commits ahead of `origin/main` (`51337f8`), not pushed: `43ce1c2` auto-suite role pick, `12a09c7` table-cell truncation, `1fea084` HF search `filter=`, `229d1c2` nvidia-smi policy filter, `e1e12a7` project-delete registry rescan, `636325b` per-test `FTS_ROOT`. |
| Service | genorbox1 :7860 active on `e1e12a71` code; RTX 3090 only, GTX 1070 masked (`gpu-mask.conf`). DB has 0 projects, all models unloaded, GPU0 ~0.6 GiB idle. |
| E2E proof | `.tmp/e2e-final/REPORT.md` (12 items PASS, evidence in `.tmp/e2e-final/ev/`): Gemma-4-12B helper mined 5 docs on the 3090; 24-row dataset with 10 grounded rows; Qwen3-0.6B 180 steps loss 0.0315; merge + GGUF Q8_0; suite 58 %, held-out 0/3, benchmarks; RAG chat 4/5 grounded; encrypted RAG round trip; dark+light visual probe; honest CUDA OOM; CPU path + degraded banner. |
| Tests | Full suite on this tree: 1745 passed, 1 skipped, 1 failed (untracked new test file, fixed by commit `636325b`; `test_repo_hygiene.py` passes), 22m18s. Ruff, codemap-check pass. |
| Deploy | No fan-dragon deploy; its checkout was reported to have a stub `.git`. |

## In flight

- Nothing running. Local commits await Genor's OK to push.

## Next steps

1. Push when approved, reconcile fan-dragon's checkout, then `bash update.sh` there.
2. Cache the RAG embedder/reranker across requests: `PortableRAG.load()` reloads e5-large + reranker per request (~2.5 s per `rag/search`); needs an idle-expiry / release-before-training design.
3. Improve coverage-fill question quality ("What does the source say about “It”?" is unanswerable and caps suite/held-out scores).
4. Surface the training error in the Live Status panel (today only the Past Runs row) with a "reduce batch / sequence length" hint on CUDA OOM.
5. Stop the suite leaking `data/projects/<id>` into the repo cwd (252 stale dirs, ~3 MB).
6. Repeat grounded-vs-plain evaluation on several seeds; measure distractor rows.

## Known issues

- `~/.finetune-studio/projects` keeps 9 pre-existing test-debris dirs with files (no DB project).
- RAG export response `download_url` lacks the `/api` prefix (the UI adds it).
- `/api/hf/search` `last_modified` is the string "None" with huggingface_hub 1.x.
- `/api/models/load` failures return HTTP 200 with `status:"error"`.
- Wizard `wizTrain()` has no epochs control; ≥150-step runs on small data need the Training page or API.
- No `fts` command builds datasets; grounded options are WebUI/API only.

## Commands

- Full tests: `.venv/bin/python -m pytest -q -p no:cacheprovider --ignore=tests/test_vram.py`
- Lint: `.venv/bin/ruff check src/ scripts/`
- Codemap: `make codemap-check`
- GPU plan/health: `bash install.sh --plan`, `.venv/bin/fts accel`
- Visual probe: `node ~/.openclaw/workspace/.tmp/qa-sweep/e2e-visual.cjs <pid> <out.jsonl>` (sets `fts.tutorial.seen`).

## Blockers

- Fan-dragon deploy requires resolving its invalid checkout; no deployment or host changes made.
