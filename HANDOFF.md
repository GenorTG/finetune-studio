# HANDOFF — finetune-studio

Local fine-tune + data-prep WebUI (`src/finetune_studio/`).
Edit on **genorbox1** → push → **fan-dragon** runs GPU work; **both hosts now run the service** via the repo scripts.

**Read first:** `docs/WORKPLAN.md` (order is law) · `docs/PRODUCT-BRIEF.md` (north star) · this file · `AGENTS.md`.

## Mission

Trained models must reliably answer the learned corpus — no lying about trained DB sources. Judge by reading transcripts, not auto-greens. **Data guarantee: every parsed chunk must reach the training dataset (no silent holes).**

## State (verified 2026-09-25 ~19:20 CEST · HEAD `1385892` pushed · genorbox1 service active :7860 · fan-dragon deployed `0.1.0.62` @ `e6a8acf`, idle)

| Area | Status |
|------|--------|
| **Install/update scripts are THE install path (fixed + proven)** | `1385892`: install.sh/update.sh fully non-interactive (no prompts, no sudo; `FTS_AUTO_REPAIR=1` self-heals). Three real bugs fixed same day: (1) optional-packages step (unsloth/gptq) silently re-resolved torch off the CUDA index → `ncclCommResume` ImportError on driver 535 — now the torch family is pinned via `.venv/torch-constraints.txt` and every later resolve passes `-c`; (2) uv-created venvs have **no pip module** — `python -m pip` steps replaced by a uv-first `pip_install` helper; (3) update.sh looked for llama.cpp under `$HOME` while install.sh builds `.llama.cpp` project-local → now reuses it (no double build). Full `bash update.sh` green on genorbox1 19:18. |
| **genorbox1 = working second host** | systemd user unit `finetune-studio` active on :7860 (lingering on). venv healthy: torch **2.6.0+cu124** (cu124 index caps at 2.6; driver 535 ceiling) `cuda_ok True` on RTX 3090 24 GB, llama-cpp-python 0.3.35, llama.cpp CLI at `.llama.cpp/build/bin/llama-quantize`. 46/46 api+training smoke tests on the pinned family. No models downloaded (disk). |
| **Batch 25 — copy/label cleanup** | `4b6b73d`: last stale "Data Prep" user-facing labels retired (rag/files/pairs/training/chat + 2 py doc-strings), raw-SQL stat deltas → plain language ("excludes trash", "restore or purge"), pairs breadcrumb `data-prep`→`pairs` (tab/title/path now agree), RAG empty-state links to the files page. 4 test files de-drifted (workspace labels, css v65, RAG section titles, colspan). Touched suites 42/42; full CPU suite 1083 pass. Screenshot-verified at 1440px dark+light (real browser, 6 pages): no stale copy, no overflow/clips, contrast clean. |
| **Hardware reality (2026-09-25)** | genorbox1: **RTX 3090 24 GB** (driver 535 → cu124 torch ceiling), GA-Z270X-Ultra Gaming — **1× M.2 (occupied, 228 GB root @ 85%) + 1× U.2**; extra NVMe needs a PCIEX4 adapter (GPU keeps x16). fan-dragon: **RTX 5080 16 GB** (driver 615.71). GPU tests (`test_vram_profiler.py`) only pass on fan-dragon. |
| **Prior verified pillars** | Coverage-fill 100% gate (`6af6312`+`451facd`); RAG standalone MCP package incl. bundled models + setup.sh (`90bcc21`,`2b5235d`,`784efe8`); project versions + wizard (`bc1152b`,`90c86dc`); size-aware preset advisor (`b78bf15`); flow-scoped nav + copy (`291b1a4`→`d605f7e`); artifact naming (`7777981`); file workbench (`8efaa80`→`266ffc1`); per-commit VERSION bump via pre-commit (`make hooks`); full-coverage suite law (`79f9775`); hub-download guard (`e6a8acf`/`e9bed6e`). Training: `f76bf64f` (12 ep r128) = 94.8% production model on locked 500-case instrument; `e65fafd5` coverage-filled retrain net-neutral. Project 58d4e331 = 554-row dataset, 131/131 chunks. QA sweep tooling: `~/.openclaw/workspace/.tmp/qa-sweep/`. |

## Next steps

1. **Deploy batch 25 + script fixes to fan-dragon when VRAM frees** (Genor's call; do not load models unasked): `ssh fan-dragon 'cd /home/genortg/finetune-studio && bash update.sh 2>&1 | tail -5'`
2. **Versions UX polish** — compare manifests side-by-side, copy-pins-to-new-project: start at `src/finetune_studio/webui/routes/versions.py` + wizard step 6.
3. **Specialized RAG corpus** for the 10 hand-picked 58d4e331 files: `POST /api/projects/58d4e331/rag/build` on a filtered source set; pin result in a version manifest.
4. **60ep/r64 noise-check** on the 48-row subset (GPU, fan-dragon): `POST /api/training/start {project_id:"58d4e331", dataset_id:<a0ae8778>, preset_id:"standard", overrides:{num_epochs:60, lora_rank:64}}`.
5. **Eyeball bench judging** on the NEW 554-case full-coverage instrument (f76bf64f/merged) per `docs/judging/PROTOCOL.md` — first comparable full-coverage verdict (GPU).
6. **genorbox1 disk**: 30 G free — no model downloads until Genor's SSD decision lands.

## Commands

```bash
# both hosts — install/update/service (non-interactive)
bash install.sh            # full env; FTS_AUTO_REPAIR=1 to self-heal silently
bash install.sh --check    # deep health (torch pin, llama-cpp, CLI)
bash update.sh             # pull + health + pinned sync + migrations + restart
systemctl --user restart finetune-studio   # genorbox1 AND fan-dragon (user units)
journalctl --user -u finetune-studio -f

# dev loop (genorbox1)
cd ~/work/finetune-studio
make test                  # CPU suite; GPU tests only pass on fan-dragon
.venv/bin/python -m ruff check src/        # NEVER `make lint` (swallows failures)
make codemap               # commit docs/CODEMAP.md with code moves
make hooks                 # once per clone: pre-commit VERSION bump

# deploy
git push && ssh fan-dragon 'cd /home/genortg/finetune-studio && bash update.sh 2>&1 | tail -5; git log --oneline -1'

# data flow (export self-fills coverage)
POST /api/projects/{pid}/data-prep/start {source_id, qa_per_chunk, difficulty, style}
POST /api/projects/{pid}/data-prep/qa/bulk {ids, action:"approve"}
GET  /api/projects/{pid}/data-prep/export?fmt=sharegpt&only=approved
POST /api/training/start {project_id, dataset_id, model_path, num_epochs, lora_rank, export_gguf, gguf_quants}
GET  /api/training/recommend?tier=&base_model=&pairs=&dataset=
```

## Blockers

- **fan-dragon VRAM busy** (games/other services — Genor's rule: observe, never kill) → deploy + GPU steps parked; this session was GUI/static only.
- **genorbox1 disk 85 %** (30 G) → no model pulls; NVMe expansion is Genor's hardware call (PCIEX4 M.2 adapter is the cheap path).
- genorbox1 driver 535 caps torch at the cu124 index (2.6.0). Newer torch needs a driver upgrade (≥ 580 for cu130) — only if a feature demands it.
- Visual changes still require real rendered-page checks (sweep + `shots/*.png`); HTTP codes alone don't count.

## Gotchas worth re-reading before data-prep or training work
See repo `AGENTS.md ## Gotchas` + `docs/GOTCHAS.md`. Key: coverage-fill counts `qa` and `qa_coverage_fill` separately; extractive fill answers stay verbatim; never compare strict % across different suite instruments; templates may only use defined CSS tokens; artifact labels via `naming.display_for_path()`.
