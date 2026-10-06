# Handoff — finetune-studio

## Mission

Local fine-tune, data-prep and RAG WebUI. GPU-first on every vendor (NVIDIA/AMD/Intel/Apple), CPU only on GPU-less hosts, never a silent fallback (Genor 2026-10-05).

## State (verified 2026-10-06)

| Area | State |
|---|---|
| Git | `main` = `origin/main` = `4cfb458` after this doc commit; nothing unpushed, working tree clean. Genor allows commits and pushes to origin. |
| CI | `.github/workflows/ci.yml`: ruff+codemap, `install.sh --plan` smoke (`scripts/ci_plan_smoke.py`, Linux+macOS), 5 CPU test shards packed by measured per-file weights (`scripts/ci_shard.py` `WEIGHTS`; pytest wall 40–120 s → 50–71 s), `ci-ok` aggregate for branch protection; every action SHA-pinned, Dependabot bumps them weekly (PR #1 `ci/rework`, not merged until Genor says). Last full green on `main`: `4cfb458`. |
| Service | genorbox1 :7860 active, user unit, RTX 3090 only (`finetune-studio.service.d/gpu-mask.conf`: `FTS_GPU_EXCLUDE=GTX 1070`). It still runs the `e8844e1` build — `systemctl --user restart finetune-studio` to pick up the rest. DB: 0 projects (backup `.tmp/db-backup/`). |
| Live proof | `.tmp/e2e-final/REPORT.md` — 12/12 PASS on the 3090 (helper mining, grounded dataset, train 180 steps loss 0.03, merge+GGUF, trained-run benchmark, RAG chat 4/5, encrypted RAG round trip, OOM honesty, CPU/degraded path, dark+light probe). Wheel verification: `.tmp/accel-verify/REPORT.md` (D1–D12 fixed). Multi-GPU index proof: `.tmp/lane-c/` (3090 as visible index 1, 1070 untouched). |
| Done since 2026-10-05 | accel layer + installers · grounded rows · RAG model cache · D1–D12 · failed-run panel + OOM hint · wizard epochs control · specific coverage-fill questions · no test leaks into `data/projects` · multi-GPU index correctness · OS-aware accel hints · GitHub CI. |
| Local tests | Last local full suite (5 foreground chunks, 2026-10-05): 1773 passed, 1 skipped, 0 failed; the 2026-10-06 lanes ran focused tests locally; CI now runs everything on CPU. Ruff + codemap clean. |

## In flight

- Nothing. Lane C has reported (multi-GPU index fix resolved; its sandbox/clone/venv scratch removed; evidence in `.tmp/lane-c/ev/`, gitignored). Its note that CI shards 2/3 may download ~4.4 GB of HF models on a cold cache did not materialise: the green run on `4cfb458` finished every shard in 1.5–3 min.

## Next steps

1. `systemctl --user restart finetune-studio`, then `curl localhost:7860/api/system/accelerator` and `.venv/bin/fts accel` to confirm the final build runs on the 3090.
2. Fresh live rerun of the quality numbers after the coverage-fill change (old vague questions gave suite 58 %, held-out 0/3) on a throwaway `e2e-*` project on :7860; delete it afterwards.
3. Measure grounded rows over several seeds and with distractor chunks (n=1 so far).
4. `fts` command for dataset build (grounded options are WebUI/API only); measure rag/chat latency end-to-end with a loaded model.
5. Merge PR #1 (`ci/rework`), then in repo settings require the single `ci-ok` check on `main` (it must have run once on main before GitHub lists it).
6. AMD/Intel/Apple stay plan/unit-tested only (no hardware here); fan-dragon (RTX 5080) deploy deferred by Genor.

## Known issues

- RAG chat grounding 4/5 is one small single-run sample on a 0.6B model — not a benchmark.
- `~/.finetune-studio/projects` may hold ~9 pre-existing test-debris dirs (not in DB).
- abetlen cu121–cu124 wheels: SM lists never inspected (floor 0).
- Windows HIP/Vulkan hints are OS-aware but untested on real Windows.
- `GET /api/system/version` `git_commit` reads `.git/HEAD` per request (correct; not stale).

## Commands

- Tests (≤10 min per foreground call → chunk): `ls tests/test_*.py | grep -v test_vram.py | awk -v k=0 'NR%5==k' > .tmp/chunk0.txt` then `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q -p no:cacheprovider $(cat .tmp/chunk0.txt)`
- Lint: `.venv/bin/ruff check src/ scripts/` · Codemap: `make codemap-check`
- GPU plan/health: `bash install.sh --plan` (stdout = JSON), `.venv/bin/fts accel`
- GitHub: `gh run list --limit 5`, `gh run view <id> --log-failed`

## Blockers

- None.
