# Handoff — finetune-studio

## Mission

Local fine-tune, data-prep and RAG WebUI. GPU-first on every vendor (NVIDIA/AMD/Intel/Apple), CPU only on GPU-less hosts, never a silent fallback (Genor 2026-10-05).

## State (verified 2026-10-06)

| Area | State |
|---|---|
| Git | `main` = 13 commits ahead of `origin/main` (`e8844e1`) before this handoff commit; this handoff + AGENTS commit is pushed together with them (see Next steps 1). Working tree otherwise clean. Genor allows commits and pushes to origin. |
| Pushed so far | up to `e8844e1`: accel layer, installers, grounded rows, RAG model cache, accel defects D1–D12, E2E fixes. |
| Unpushed (13 + doc) | `ec93d63` OS-aware accel hints · `d3f7aa7`/`db30995` failed-run panel + OOM hint · `89a6fc7` wizard epochs control · `f020f69` coverage-fill specific questions · `ca16ba1` tests stop leaking `data/projects` · `88b18c9` multi-GPU index correctness · `fd240a7` CI prerequisites (`scripts/ci_shard.py`) · `8a1f45d` CI workflow + `olefile` in `parsers` extra · `4c1514d` CPU-only-safe tests · HANDOFF/AGENTS doc commits. |
| CI | `.github/workflows/ci.yml` (ruff + codemap job, 5 CPU test shards, `FTS_DEVICE=cpu`); action tags checked to exist (checkout/setup-python/setup-uv v7). It has NEVER run on GitHub — the push triggers its first run: check `gh run list --limit 5` and fix any red shard. |
| Service | genorbox1 :7860 active, user unit, RTX 3090 only (`finetune-studio.service.d/gpu-mask.conf`: `FTS_GPU_EXCLUDE=GTX 1070`). It runs the `e8844e1` build; restart it after the push to pick up the rest. DB: 0 projects (backup `.tmp/db-backup/`). |
| Live proof | `.tmp/e2e-final/REPORT.md` — 12/12 PASS on the 3090 (helper mining, grounded dataset, train 180 steps loss 0.03, merge+GGUF, trained-run benchmark, RAG chat 4/5, encrypted RAG round trip, OOM honesty, CPU/degraded path, dark+light probe). Wheel verification: `.tmp/accel-verify/REPORT.md`. |
| Tests | Last full suite (5 foreground chunks, before the 2026-10-06 lanes): 1773 passed, 1 skipped, 0 failed. The 2026-10-06 lanes ran focused tests only (lane C: 245 passed); a full re-run on the final tree is still owed. Ruff + codemap clean. |
| GitHub | Pages deploys green on `e8844e1`; old run `37373751446` "failed" only in the cancelled status-report job (build+deploy succeeded). |

## In flight

- Lane C (`agent:main:dashboard:79b10be2-b5ee-4fe2-8b80-8d17b1027599`, multi-GPU index + CI) was still running local CPU-only shard simulations in an isolated clone when this was written. Its commits up to `4c1514d` are in; anything it commits later is unpushed — `git log origin/main..HEAD`. Isolated clone/venv scratch under `.tmp/` may remain: remove after it ends.

## Next steps

1. `git push origin main`; then `gh run list --limit 5` and `gh run view <id> --log-failed` — fix any failing CI shard (expect first-run surprises: missing deps, GPU/network assumptions).
2. Full suite on the final tree in 5 foreground chunks (see AGENTS Gotchas) — confirm 0 failures; `systemctl --user restart finetune-studio`; `curl localhost:7860/api/system/accelerator`.
3. Fresh live rerun of the quality numbers after the coverage-fill change (suite score was 58 %, held-out 0/3 with the old vague questions) — use a throwaway `e2e-*` project on :7860.
4. Measure grounded-rows over several seeds and with distractor chunks (n=1 so far).
5. Remaining backlog: `fts` command for dataset build (grounded options are WebUI/API only); `rag/chat` quality needs a loaded model to measure latency end-to-end.
6. AMD/Intel/Apple stay plan/unit-tested only (no hardware here); fan-dragon (RTX 5080) deploy deferred by Genor.

## Known issues

- RAG chat grounding 4/5 is one small single-run sample on a 0.6B model — not a benchmark.
- `~/.finetune-studio/projects` may hold ~9 pre-existing test-debris dirs (not in DB).
- Wheels for abetlen cu121–cu124: SM lists never inspected (floor 0).
- Windows HIP/Vulkan hints are OS-aware now but untested on real Windows.
- Version endpoint `git_commit` reads `.git/HEAD` per request (correct; not stale).

## Commands

- Tests (≤10 min per foreground call → chunk): `ls tests/test_*.py | grep -v test_vram.py | awk -v k=0 'NR%5==k'` then `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q -p no:cacheprovider $(cat chunk)`
- Lint: `.venv/bin/ruff check src/ scripts/` · Codemap: `make codemap-check`
- GPU plan/health: `bash install.sh --plan` (stdout = JSON), `.venv/bin/fts accel`
- GitHub: `gh run list --limit 5`, `gh run view <id> --log-failed`

## Blockers

- None. Waiting only on the first GitHub CI result after the push.
