# Handoff — finetune-studio

## Mission

Local fine-tune, data-prep, and RAG WebUI. GPU first on every vendor (NVIDIA/AMD/Intel/Apple), CPU only on GPU-less hosts, never a silent CPU fallback (Genor 2026-10-05). The accelerator overhaul, grounded-rows feature and RAG cache are done and proven live on the RTX 3090.

## State (verified 2026-10-05)

| Area | State |
|---|---|
| Git | `main` pushed up to `6dafd77`; the accel-defect fixes (`60df70b`…`a189424`), RAG cache/API nits (`f25dc43`…`df15261`), `fac9494`, `5cdaff4` and the vram-test fix are committed on top — see Next steps for the push. |
| Service | genorbox1 :7860, user unit, `FTS_ROOT=~/.finetune-studio`, RTX 3090 only (`gpu-mask.conf` sets `FTS_GPU_EXCLUDE=GTX 1070`). DB 0 projects. |
| Accelerator | `fts accel`: cuda:0 RTX 3090, CUDA 13.0, torch 2.14.1+cu130, llama.cpp 0.3.36 with CUDA offload. Selection lives only in `scripts/accel_plan.py`. |
| E2E proof | `.tmp/e2e-final/REPORT.md` (12/12 PASS, live :7860): helper mining on 3090, grounded dataset, Qwen3-0.6B 180 steps loss 0.03, merge + GGUF, trained-run benchmark, RAG chat 4/5, encrypted RAG round trip, dark+light visual probe, honest CUDA OOM, CPU path + degraded banner. |
| Wheel check | `.tmp/accel-verify/REPORT.md`: per-vendor torch/bnb/llama wheels confirmed to exist; defects D1–D12 found and fixed with regression tests (Pascal/cu13 wheel, nvcc vs Blackwell, gfx12 fallbacks, `acc.index` memory ops, bf16 on Pascal, numeric policy tokens, `oom` word match, macOS<14, `--plan` JSON). |
| Tests | Full suite on this tree, 5 foreground chunks: 1773 passed, 1 skipped, 0 failed (`tests/test_vram.py` excluded). `ruff check src/ scripts/` and `make codemap-check` clean. |

## In flight

- Nothing running.

## Next steps

1. Push (`git push origin main`), then `systemctl --user restart finetune-studio` on genorbox1 so the service runs the final code.
2. fan-dragon (RTX 5080 = real Blackwell/cu132 test) has a stub `.git`; restoring its checkout is Genor's call, then `bash update.sh` and `bash install.sh --plan` there.
3. Improve coverage-fill question quality ("What does the source say about “It”?" caps suite/held-out scores).
5. Stop the suite leaking `data/projects/<id>` into the repo cwd (~250 dirs).
6. Untested: HF `Trainer` placement when `acc.index != 0`; abetlen cu121–cu124 wheel SM lists; AMD/Intel/Apple on real hardware; grounded-vs-plain over several seeds and distractor rows.

## Known issues

- Failed runs: Training page Live status shows the full error + OOM hint (`training/failure_hint.py`, `GET /api/training/failure`); wizard has an advanced epochs field + <100-step warning; accel_plan missing-toolchain hints are OS-aware (verified live 2026-10-06, `.tmp/qa-shots/laneB-*`). No standalone gradient-checkpointing knob exists (only via Unsloth), so the hint does not offer one.
- RAG embedder/reranker cached process-wide (`data/rag_portable/model_cache.py`, idle expiry `FTS_IDLE_TIMEOUT`=300 s, released by `unload_all_models`, training/merge/export/model load; `GET /api/inference/status` → `rag_models`).
- `~/.finetune-studio/projects` keeps ~9 pre-existing test-debris dirs with files (not in DB).
- No `fts` command builds datasets; grounded options are WebUI/API only.

## Commands

- Full tests (≤10 min per foreground call, so split): `.venv/bin/python -m pytest -q -p no:cacheprovider --ignore=tests/test_vram.py`
- Lint: `.venv/bin/ruff check src/ scripts/` · Codemap: `make codemap-check`
- GPU plan/health: `bash install.sh --plan` (stdout = JSON), `.venv/bin/fts accel`
- Visual probe: `node ~/.openclaw/workspace/.tmp/qa-sweep/e2e-visual.cjs <pid> <out.jsonl>`

## Blockers

- fan-dragon deploy waits on Genor (invalid checkout there).
