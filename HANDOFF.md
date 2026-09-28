# HANDOFF — finetune-studio

Local fine-tune + data-prep WebUI (`src/finetune_studio/`).
Edit on **genorbox1** → push → **fan-dragon** runs GPU work; **both hosts run the service** via the repo scripts.

**Read first:** `docs/WORKPLAN.md` (order is law) · `docs/PRODUCT-BRIEF.md` (north star) · this file · `AGENTS.md`.

## Mission

Trained models must reliably answer the learned corpus — no lying about trained DB sources. Judge by reading transcripts, not auto-greens. **Data guarantee: every parsed chunk must reach the training dataset (no silent holes).**

## State (verified 2026-09-28 11:25 CEST · CPU suite green · helper defaults rewritten · genorbox1 + fan-dragon both `active` on :7860)

| Area | Status |
|------|--------|
| **Project-scoped 404 UX is now complete** | `e37dd3c` closed the class `bc8d018` opened. Measured, not inferred: probe of all 59 project-scoped API GET routes with a bogus pid → **0 return 200** (was 24). Pages deliberately unchanged: 14 of 16 302 a bad pid to `/projects` by design, 2 return 404. Guard mirrors `file_library.py:48` `_project_or_404`, each file keeping its local error style. `/api/projects/{pid}/rag` no longer hands an absolute `corpus_dir` path for a nonexistent project. The 3 SSE routes guard in the **outer** function, so no `200 + data:` error frame is ever emitted. A guard only changes behaviour for a pid that does not exist, so valid-project flows are untouched (a real project with no runs still returns `[]`). |
| **Both hosts deployed + verified identically** | fan-dragon `update.sh` green 23:45, HEAD `b119b50`, service `active`; VRAM observed 4.6 G/16 G before touching anything (never killed a foreign user). genorbox1 needed a restart — it had been serving code from 19:04, i.e. *before* the fix landed. |
| **`make test` no longer eats its own failure** | `b119b50`: the target was `... pytest ... \|\| python -m pytest ...`. The fallback reaches for a bare `python`, absent from this venv, so a genuinely failing suite exited **127** and the pytest exit code was discarded — it only ever bit on the runs that mattered. Real code now, so a red suite surfaces its own failure. |
| **CPU suite is green** | After fixing the remaining test-isolation/version-drift cases, full `make test` passed: **1148 passed, 6 warnings in 541.50s** (2026-09-28). No GPU/model work was performed. Targeted helper tests (`test_helper_defaults.py` + `test_helper_model.py`) = 28 passed. |
| **Helper defaults rewritten** | `52f27ed` swapped the helper seat to **Gemma 4 12B Uncensored Q4_K_M** (`zaakirio/gemma-4-12b-it-uncensored-GGUF`) and added **Qwen3-30B-A3B Uncensored IQ4_XS** (`unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF`) as the alternate. Both helpers live at `~/finetune-studio/models/gguf/` on genorbox1 + fan-dragon. Migration in `manager._ensure_db` moves any lingering `Qwen3-8B-Q5_K_M.gguf` row to the new default + path. `/api/providers` on both hosts returns both rows with `is_helper: true`. The previous Qwen3-8B was a planned retirement — not a regression. |
| **Model cleanup (2026-09-28)** | genorbox1: HF cache untouched (~50 MB of datasets/embedders, nothing to reclaim). fan-dragon: deleted `unsloth/Qwen3-4B` + `Qwen--Qwen3-4B` + `AnkitAI/Parable-Qwen3-8B-Claude-Fable-5-GGUF` stub + `Qwen3.8-27B-abliterated-Q4_K_M.gguf` (17 GB) + `mmproj-Qwen3.8-27B-bf16.gguf`. Reclaimed **~32 GB** on fan-dragon (168 GB → 180 GB free before downloads). The ComfyUI `llama-server` on genorbox1 (`chris-ai-gemma4-e4b-v20.Q4_K_M.gguf` + mmproj on :8088) was not touched — out of scope per standing rule. |
| **Lint baseline** | `ruff check src/` = **134 pre-existing findings**, unchanged by `52f27ed`. Do not blind `--fix` (that is how the 98 F401s in `db/__init__.py` nearly deleted the DB facade). |
| **Blackwell sm_120 auto-fix shipped** | `c4b149a` + `b2670fe` + `223ad5a` (v0.1.0.86). Diagnose now queries `nvidia-smi --query-gpu=compute_cap`; on cc ≥ 12 it emits `REINSTALL_LLAMA_CPP` and the repair branch builds llama-cpp-python from source with `CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120"` + `--no-binary llama-cpp-python` (the cached abetlen cu130 wheel lacks sm_120 kernels → `ggml_cuda_op_scale` ABRT). Two follow-up bugs found in fan-dragon dry run: install.sh crashed with `warn: command not found` when `$HOME/llama.cpp` existed (function defs lived **after** the legacy-warn block) — fixed by hoisting `log/warn/die` above the if-block; install.sh's `install_llama_cpp_cli` ran `pip_install -r requirements-convert_hf_to_gguf.txt` in `--llama-cpp-only` mode, downgrading venv torch 2.12.1+cu130 → 2.11.0+cpu (mixed torchaudio) — fixed by skipping that pip_install in `--llama-cpp-only` and pinning `${CONSTRAINT_ARGS[@]}` in normal mode. All 6 pip calls in `repair()` now use `str(venv_py)` (was `sys.executable` → leaked into miniconda on fan-dragon). **Verified end-to-end**: fan-dragon loaded Gemma 4 12B Q4_K_M onto RTX 5080, `/api/inference/chat` returned `"27 * 14 is 378."`, service stable, VRAM 11 GB during inference. `test_install_diagnose.py`: 20 passed (new `test_blackwell_sm120_triggers_source_build` asserts the venv python, `--no-binary`, and the sm_120 CMAKE arch). genorbox1 untouched — its 3090 (sm_86) was never affected by the wheel mismatch. |
| **Hardware reality** | genorbox1: **RTX 3090 24 GB**, driver 535 → cu124 torch ceiling 2.6.0, root `/` 1.8 TB at **40 %** (1.1 TB free) — HANDOFF’s old “85 %” line was stale. fan-dragon: **RTX 5080 16 GB**, driver 615.71, `/home` 954 GB at **82 %** (175 GB free after cleanup + downloads). GPU tests (`test_vram_profiler.py`) only pass on fan-dragon. |
| **Prior verified pillars** | Coverage-fill 100% gate (`6af6312`+`451facd`); RAG standalone MCP pkg (`90bcc21`,`2b5235d`,`784efe8`); project versions + wizard (`bc1152b`,`90c86dc`); size-aware preset advisor (`b78bf15`); flow-scoped nav + copy (`291b1a4`→`d605f7e`); artifact naming (`7777981`); file workbench (`8efaa80`→`266ffc1`); per-commit VERSION bump via pre-commit (`make hooks`); import-health guard (`9903891`); DB isolation autouse (`d730456`,`be776ff`); hand-rolled client fixture fix (`26199a3`,`a59fe69`). Training: `f76bf64f` (12 ep r128) = 94.8 % on the locked 500-case instrument. |

## Next steps

1. **Versions UX polish** — compare manifests side-by-side, copy-pins-to-new-project: `webui/routes/versions.py` + wizard step 6.
2. **Smoke-load each helper on both hosts** — **fan-dragon Gemma 4 12B verified** (sm_120 source build → load → chat round-trip, 11 GB VRAM). Still pending: fan-dragon Qwen3-30B-A3B (15.25 GB weights + q8_0 KV ≈ 16 GB → tight on 16 GB card, may need IQ4_XS / q4_0 KV cache or partial CPU offload), genorbox1 Gemma + Qwen (3090 has 24 GB, both should fit easily). Use `POST /api/models/load` (the path that fills `inference_engine.model`, not `/api/providers/{pid}/load` which only warms `ModelManager`).
3. **Specialized RAG corpus** for the 10 hand-picked 58d4e331 files: `POST /api/projects/58d4e331/rag/build` on a filtered source set; pin in a version manifest.
4. **60ep/r64 noise-check** (GPU, fan-dragon): `POST /api/training/start {project_id:"58d4e331", dataset_id:<a0ae8778>, preset_id:"standard", overrides:{num_epochs:60, lora_rank:64}}`.
5. **Eyeball bench judging** on the NEW 554-case full-coverage instrument (`f76bf64f/merged`) per `docs/judging/PROTOCOL.md` — first comparable full-coverage verdict (GPU).
6. **genorbox1 SSD**: original ~30 GB line was stale; genorbox1 actually has ~1.1 TB free. Still no model pulls beyond what was just downloaded unless Genor asks — disk health is fine.

## Commands

```bash
# both hosts — install/update/service (non-interactive)
bash install.sh && bash install.sh --check     # deep health (torch pin, llama-cpp, CLI)
bash update.sh                                  # pull + health + pinned sync + migrations + restart
systemctl --user restart finetune-studio       # genorbox1 AND fan-dragon (user units)

# dev loop (genorbox1)
make test                  # CPU suite — real exit code; 1146 passed at last verification
.venv/bin/python -m ruff check src/   # NEVER `make lint` (it ends in `|| true`)
make codemap / make hooks

# the 404 audit (dev tool, untracked, not in git)
.venv/bin/python .tmp/audit/probe_pid_routes.py http://127.0.0.1:7860
# GET-only; walks the app's own route table. FastAPI 0.141 nests routers as
# _IncludedRouter, so a flat `app.routes` scan sees only the 5 built-ins —
# real paths live on `original_router.routes` under `include_context.prefix`.
# Expect: 0/59 API leaks, page_status_counts {'302': 14, '404': 2}.

# deploy
git push && ssh fan-dragon 'cd /home/genortg/finetune-studio && bash update.sh 2>&1 | tail -5; git log --oneline -1'
```

## Blockers

- **fan-dragon VRAM** — free enough to deploy (4.6 G/16 G used), but model pulls stay parked until Genor says so. Rule stands: observe, never kill a foreign user.
- **genorbox1 disk 85 %** → no model pulls; NVMe expansion is Genor's hardware call (PCIEX4 M.2 adapter is the cheap path).
- driver 535 caps torch at cu124 (2.6.0); newer torch needs driver ≥ 580 — only if a feature demands it.
- Visual changes still require real rendered-page checks (sweep + `shots/*.png`); HTTP codes alone don't count.

## Gotchas worth re-reading before data-prep or training work

See repo `AGENTS.md ## Gotchas` + `docs/GOTCHAS.md`. Key: coverage-fill counts `qa` and `qa_coverage_fill` separately; extractive fill answers stay verbatim; never compare strict % across different suite instruments; templates may only use defined CSS tokens; artifact labels via `naming.display_for_path()`. **`benchmarks.py` routes returning `JSONResponse` need `response_model=None`** — a `-> list[...] | JSONResponse` annotation is not a valid Pydantic field and the app fails to *import* without it.
