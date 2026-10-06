# Handoff — finetune-studio

## Mission

Local fine-tune, data-prep and RAG WebUI. GPU-first on every vendor (NVIDIA/AMD/Intel/Apple), CPU only on GPU-less hosts, never a silent fallback (Genor 2026-10-05).

## State (verified 2026-10-06)

| Area | State |
|---|---|
| Git | `main` = `origin/main`, tree clean, everything pushed. Genor allows commits and pushes to origin and a free hand with CI, service restarts and real-hardware tests. |
| CI | `.github/workflows/ci.yml` (PR #1, merged): actions pinned to SHAs + Dependabot (github-actions, weekly; its setup-uv 10.2.0 bump merged, 9/9 green), lint+codemap, 5 time-balanced CPU test shards, `install.sh --plan` smoke (ubuntu + macOS, 5 fake GPU fixtures), `ci-ok` aggregate. Last run on the merge commit: all green. |
| Service | genorbox1 :7860 active, user unit, RTX 3090 only (`gpu-mask.conf`). Restarted on the final build after the HF-delete fix; `fts accel` + `/api/system/accelerator` = cuda:0 RTX 3090, CUDA 13.0, llama.cpp CUDA offload. DB: 0 projects, 0 scratch models, `output/` 8 K, GPU0 905 MiB idle. |
| Local tests | Full suite 2026-10-06 on the final tree, 5 foreground chunks: **1903 passed, 1 skipped, 0 failed**. Ruff + codemap clean. No test data leaked into `data/projects` or `~/.finetune-studio/projects`. |
| New this session | `fts dataset build` (route + CLI share `data/prep/dataset_build.py`) · grounded rows quizzed with their own CONTEXT · wizard export-gate buttons · contrast/size fixes (CSS v79) · HF-delete rescans the model registry · CI rework. |
| Live E2E rerun | Evidence `.tmp/e2e-rerun/ev/` (project `e2e-rerun-quality`, deleted afterwards). Real browser wizard + live API on the 3090: upload 5 docs, RAG index, helper (Gemma-12B) mining 5 docs, dataset 27 rows / 11 grounded, train Qwen3-0.6B 180 steps loss 0.0246 (170 s), merge, GGUF, quiz, held-out, RAG chat, encrypted bundle round trip, OOM honesty, CPU sandbox, visual sweep. |

## Quality numbers (honest)

- Quiz before the fix: 19/27 = 70.4 %. Cause found: all 7 failures were grounded rows asked bare (plain rows 15/15, grounded rows bare 1/9, **with their own context 9/9**). After the fix, same weights: **26/27 = 96.3 %** (memory 15/16, with context 11/11). This is an instrument fix, not a better model; the earlier 58 % used a different, tinier corpus, so it is not a like-for-like comparison.
- Held-out slice (seed-42, 3 rows never trained on): earlier 0/3. Now the two grounded held rows are answered correctly from their context (auto-judge passes 1, misses a correct paraphrase on the warranty row = judge false negative, human-graded 2/3); the plain held row (Pip, no context) fails, as it must. n=3, not a benchmark.
- RAG chat on the Q8_0 GGUF: 6/6 grounded (all six facts were in the training docs, so this is not an unseen-fact test).

## In flight

- Nothing. Both lanes (dataset-build, CI) reported and are merged; their worktrees and branches are removed.

## Next steps

1. **Decide the Q8_0 problem** (below): e.g. default new GGUF exports to Q6_K/Q4_K_M for small models, or isolate llama.cpp in a subprocess so a native abort cannot kill the service. Repro: `.tmp/e2e-rerun/sandbox_app.py` + `longprompt.py`.
2. Branch protection on `main`: require the single `ci-ok` check (GitHub lists it after its first run on main — it has run).
3. Measure grounded rows over several seeds and with distractor chunks (n=1 so far); measure rag/chat latency end-to-end with a loaded model.
4. Scanned/vague docs (`scan_0042`-type) still block Step 3; the wizard now offers "Export without it" / file library. Consider surfacing the blocked file earlier (Step 2), and the chunk-boundary artefact seen in a coverage-fill question ("what is stated regarding r support Customer support?" — heading cut mid-word).
5. Optional: apply the same coverage gate/persist helper to `POST /datasets/subset` (still duplicates it).
6. AMD/Intel/Apple stay plan/unit-tested only; fan-dragon (RTX 5080) deploy deferred by Genor.

## Known issues

- **Q8_0 GGUF of the trained 0.6B aborts the whole service** on ~1000+-token prompts (`CUDA error: an illegal memory access`, SIGABRT inside llama.cpp mmq/PDL; systemd restarts it in ~5 s). chat-v2 with RAG enabled triggers it every time. Same weights as Q6_K / Q4_K_M: 9/9 raw runs and 8/8 chat-v2+RAG calls fine; Gemma-12B Q4_K_M fine; `GGML_CUDA_DISABLE_GRAPHS=1` does not fix the app path; not reproducible in a plain script (flaky there). Installed llama-cpp-python 0.3.36 is source-built; the abetlen cu13x wheel is untested. Raw `/api/rag/chat` with the same GGUF worked 6/6 at ~600 tokens.
- chat-v2 with RAG retrieves on the event-loop thread and rebuilds embedder/reranker each call ("Loading weights" per request) — slow, not the crash cause.
- RAG chat grounding is a single-run sample on a 0.6B model — not a benchmark.
- `~/.finetune-studio/projects` holds ~9 pre-existing test-debris dirs (not in DB).
- abetlen cu121–cu124 wheel SM lists never inspected; Windows HIP/Vulkan hints untested.
- Wizard "wall of text" paragraphs on wizard/testing/benchmarks/chat are design backlog (advisory in the audit).

## Commands

- Tests (≤10 min per foreground call → chunk): `ls tests/test_*.py | grep -v test_vram.py | awk -v k=0 'NR%5==k' > .tmp/chunk0.txt` then `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q -p no:cacheprovider $(cat .tmp/chunk0.txt)` (k = 0..4)
- Lint: `.venv/bin/ruff check src/ scripts/` · Codemap: `make codemap-check`
- GPU plan/health: `bash install.sh --plan`, `.venv/bin/fts accel` · Dataset: `.venv/bin/fts dataset build --project <name|id> [--force] [--json]`
- GitHub: `gh run list --limit 5`, `gh run view <id> --log-failed`
- Real-browser wizard step: `node .tmp/e2e-rerun/wiz.cjs <pid> wizGenerate|wizExport|wizTrain|wizTest <outPrefix> [epochs] [baseModelSubstr]`

## Blockers

- None.
