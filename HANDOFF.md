# HANDOFF — finetune-studio

Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## Mission
Make the studio show all real activity and keep training/RAG/testing quality
honest — real root-cause fixes, real fan-dragon verification, no fake greens.

## State (verified 2026-09-18 08:30 CEST · commit cf6e703)

| Area | Status |
|------|--------|
| **Activity feed** | 34 tasks on fan-dragon; all subsystems tracked (training, benchmark, export, data_prep, rag_build, hf_download, system_update, operation middleware) |
| **Startup reconciliation** | On restart, stale `queued`/`running` rows across all 5 durable tables are marked `failed`: training_runs, system_updates, data_prep_runs, rag_corpora, model_exports (cf6e703) |
| **Event loop** | All blocking GPU/inference routes use `asyncio.to_thread`: benchmarks `_execute_benchmark`, `judge_benchmark`; testing `load_model`, `chat`, `run_test_suite`; RAG `rag_rebuild` (ac84053 + 4b550be) |
| **Benchmark judging** | Heuristic judging runs inline; all cases get verdicts. RAG benchmarks: 93-100% pass; held-out: 17-26%; source-disjoint run: 0% (training quality, not a bug) |
| **RAG corpora tracking** | `rag_rebuild` now creates `rag_corpora` rows with running/done/failed lifecycle; 36 docs/36 chunks confirmed via API (4b550be) |
| **Theme toggle** | Light/dark toggle confirmed working; `data-theme` attribute swap + localStorage persistence |
| **Tests** | 922 passed, 0 failed (genorbox1; 37 GPU-only in test_vram_profiler.py skip) |

## Next steps
1. Training quality: source-disjoint run scored 0% — the augmented dataset run `8b1dd006` scored 93-100% on RAG but only 17-26% on held-out. Consider a longer training run (200+ optimizer steps) on a merged augmented dataset.
2. Benchmark panel: "Dataset evaluation" and "RAG-grounded suite" cards in testing page — verify end-to-end with a live run.
3. Data-prep pipeline: verify the `scripts/augment_dataset.py` augmentation produces correct sharegpt format and that exporting to training works.
4. Prune dev DB junk on genorbox1 if desired (gitignored runtime artifact: `data/finetune_studio.db`).

## Commands
```bash
# Full test suite (genorbox1)
make test
# Lint (run directly; Makefile hides failures)
.venv/bin/python -m ruff check src/
# Deploy to fan-dragon
git push
ssh fan-dragon 'bash -c "cd /home/genortg/finetune-studio && git fetch origin -q && git reset --hard origin/main && systemctl --user restart finetune-studio && sleep 4 && ss -ltnp | grep 7860"'
# Verify cgroup after restart
ssh fan-dragon 'bash -c "ss -ltnp | grep 7860 | awk \"{print \$6}\" | grep -oP \"pid=\\K[0-9]+\" | xargs -I{} cat /proc/{}/cgroup"'
# Check activity feed
curl -s http://fan-dragon:7860/api/activity | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d['tasks']), 'tasks')"
```

## Blockers
None current.
