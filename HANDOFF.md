# HANDOFF — finetune-studio

Local fine-tune + data-prep WebUI (`src/finetune_studio/`).
Edit on **genorbox1** → push → **fan-dragon** runs `finetune-studio.service` on `:7860`.

**Read first:** `docs/WORKPLAN.md` (order is law) · `docs/PRODUCT-BRIEF.md` (north star) · this file · `AGENTS.md`.

## Mission

Trained models must reliably answer the learned corpus — no lying about trained DB sources. Judge by reading transcripts, not auto-greens. **Data guarantee: every parsed chunk must reach the training dataset (no silent holes).**

## State (verified 2026-09-20 ~19:15 CEST · fan-dragon at build `0.1.0.2`, service active)

| Area | Status |
|------|--------|
| **Dataset coverage — 100% enforced** | `coverage_fill.py` (`6af6312`): deterministic second pass; chunks the LLM mining missed get extractive pairs (answer quoted verbatim, `origin=coverage_fill`, status approved, no invention). Runs **at end of every mining run** (runner self-heal, `451facd`) **and before every export** (route gate) — a dataset cannot ship with silently-unmined chunks. Live proof: project 58d4e331 went 515→**554 rows**, chunk coverage **131/131 = 100%**, idempotent on re-export. `fill_all_project_gaps()` also surfaces declared-but-lost parsed artifacts (never crash-swallow). |
| **RAG standalone package + readability** | `90bcc21`: RAG page sections renamed for humans (1 Index status → 8 Shared search models; nav overview/build/search/chat; "Remove all sources"/"Rebuild from scratch" buttons). New export: `GET /api/projects/{pid}/rag/mcp-package` → self-installing tarball (corpus + single-file server.py + install.sh venv/numpy-only + README + MCP config). Speaks **MCP stdio** (`rag_search`, `rag_info`) AND **HTTP** (`/search?q=`); keyword search offline out of the box, semantic via any OpenAI-compatible /v1/embeddings (`RAG_EMBED_BASE_URL`). Proven live: downloaded 575 KB pkg, clean-venv install, real Vaelindrath queries via both modes. `tests/test_rag_mcp_package.py` 4 green. TUTORIAL §8 + README rewritten to match. **`2b5235d` adds `include_models=true`**: the embedder + reranker ship *inside* the package (corpus/embedder/, corpus/reranker/) — install.sh pulls torch-CPU + sentence-transformers, server loads bundled models first, falls back API→keyword if broken. Proven fully offline on genorbox1 clean venv: 1.38 GB pkg, "strange weather event in the heavens" → dense 0.782 correct annal, rag_info mode "offline-semantic (bundled embedder) + rerank". **`784efe8` adds `setup.sh`**: one-command guided deploy — auto venv, port prompt (or `--port`), start-now, optional persistent **systemd user service** (`--service`; `--uninstall` removes the unit + stops the local server), prints + saves `MCP-ENTRY.txt` with real absolute paths. Proven live on genorbox1: `--yes --port 8899` (health + search + valid JSON entry) and service install → active → uninstall. compresslevel=1 for model packs (gzip 9 on 2.2 GB safetensors wasted minutes). 9 package tests green. |
| **Project versioning (goal 34ff4655)** | `project_versions` table + CRUD (`bc1152b`): immutable manifests pin datasets/source_ids/RAG corpora/runs/base model; monotonic version_number, `parent_version_id` lineage → any old version is a branch base. Subset datasets: `POST /api/projects/{pid}/datasets/subset` hand-picks sources → coverage-filled, per-source row counts, registered (`58d4e331-specialized-picks-…` = 48 rows live). RAG parity gate `GET …/rag/coverage` (129/129 = 100% on the live corpus; shares rag route's corpus root — never re-derive the path). Guided flow page `/projects/{pid}/flow` (`7bb24dc`) → **replaced by the Project wizard** (`90bcc21`+`90c86dc`): `/projects/{pid}/wizard` with two modes — **Quick start** (6-step guided pipeline: upload → generate+auto-approve QA pairs → build dataset → train with base-model/preset pickers → auto-suite test with pass rate → pin; plus "Run steps 2-5" chaining, all through the same APIs) and **Step by step** (the old cards incl. RAG index, live status pills, pin form). `/flow` 302-redirects; tab renamed flow→wizard in all 3 nav spots. Tests in `tests/test_api.py::TestWizardPage`. Versions CRUD/lineage live-verified as v1 `302071b6` → v2 `4fca19a4`. Both wizard modes screenshot-verified on fan-dragon. Sticky-breadcrumb defect (user screenshot) fixed in `784efe8`: `.main{grid-row:2}` collided with the auto-placed breadcrumb → bar landed mid-page over Recent files; explicit `.app` grid rows (bar=2, subnav=3, main=4) + cache v34; verified barTop=117 gap=0. |
| **Size-aware training advisor** | `training/preset_advisor.py` (`b78bf15`): parses base size from model name (GGUF quant suffix excluded), scales rank/LR/epochs by base size + dataset size, raises epochs to clear per-tier optimizer-step floor (evidence: 772 steps → 95.1% strict, 257 → 67%). `GET /api/training/recommend?tier=&base_model=&pairs=`. Training page prefills + re-runs when base/dataset changes. 10 evidence-pinned tests. |
| **Auto-suite proven live** | `POST /api/training/runs/{id}/auto-suites/generate` on run `f76bf64f` → 500 cases, deterministic, quality-checked vs the trusted 515-suite (97% normalized-question overlap, 0 degenerate). Selectable via `GET /api/benchmarks/suites?project_id=` (param is `project_id`, **not** `pid`). |
| **Project 58d4e331** | "Vaelindrath Stress": 131 files / 129 sources / 131 chunks, **all parsed, zero failed parses** (18 zero-pair files found + filled=100%). Mining: 581 pairs → 564 approved / 17 rejected (7 id-leak, 10 ambiguous) → dedup → **515**; +39 coverage_fill approved → **554-row dataset on disk now** (551 unique approved questions, all present). |
| **Training evidence** | Run `212035ad` (4 ep, r64) = 67% strict; run `f76bf64f` (12 ep, r128/α256) = 94.8% on the locked 500-case suite; run `e65fafd5` (11 ep, r128/α256, **554-row coverage-filled dataset**) = 94.2% same instrument — **retrain on filled data was net-neutral** (fixed 23, regressed 25). Production model stays `f76bf64f/merged`; GGUF q8_0 exported for both. Eyeball: fails are genuine recall misses (digit/price confusion + similar-fact swaps); no judge false positives. |
| **Fleet (Genor's rule, 3 models)** | 4B safetensors trainer; helper = Qwen3-8B Q5_K_M GGUF (real header topology, `-1` = all layers, no magic 99); 27B abliterated GGUF. |
| **Deployed UX** | Named toasts both paths, counted upload toasts, WCAG-pass pills (5.16–10.19:1), dim-token contrast, mobile toast strip; deployed assets `app.css?v=34` + `app.js?v=21` (served + verified). |
| **Per-commit build versioning** | repo-root `VERSION` = MAJOR.MINOR.PATCH.BUILD; a versioned **pre-commit** hook (`make hooks`, `core.hooksPath`) bumps BUILD on EVERY commit and the bump rides IN that commit — the UI chip ("EARLY BETA release · v0.1.0.N") and `GET /api/system/version` (version+channel+git_commit) always pin the exact deployed code. (First attempt used commit-msg, which runs after git snapshots the index → deployed file lagged one build; caught live by the E2E, fixed + regression-tested.) Escape hatch `FTS_NO_BUMP=1`; VERSION-only commits don't bump. Tests assert the COMMITTED value. |
| **Wizard E2E start-to-finish + fix** | Browser-driven full flow on a fresh project `fb05a690` "E2E Smoke 20260920": create → upload 3 docs (parsed ✓) → **Run steps 2-5** chain → dataset → train → full-coverage test → **v1 pinned** (`e2e-smoke-v1`, pill "v1 LATEST"), screenshot-verified. The run CAUGHT a real bug: with no helper loaded, every doc's mining failed and the chain coasted on into a coverage-fill-only dataset. Fixed in `87a8643`: step 2 auto-loads the helper provider, aborts the chain when no document could be mined, and unloads the helper before training to free VRAM. Re-run: 9 LLM pairs mined + 9 coverage-fill = 18/18 approved, 18 rows → **18-question test** (full-coverage rule proven live), loss reported 6.0121, 2/18 passed — honest smoke result for a 6-step nano-fast run, not a regression. |
| **Full-coverage suite rule (law)** | `79f9775`+`81b4adc`: auto-suites test **every dataset row** — N rows → N cases, no cap. Sampling only via explicit `sample_size` (route body) / wizard "limit to a random sample" input; a sampled suite is labeled everywhere (`-sampledKofN` name, `meta.coverage="sampled"`, seed 42, uniform not head-truncation) so it can never masquerade as full. Suite JSON now `{meta, cases}` (loaders + `audit_suite_cases` backward-compatible, audit checks sampled honesty); dedup'd case names; `row_index` provenance; heldout reads the full pool (was head-5000); engine dead auto-suite block removed; discovery lists newest row per file. **Live: f76bf64f + e65fafd5 regenerated → 554/554 cases, coverage=full, 554 unique names, row_index 0..553; sampled opt-in verified (`-sampled50of554`, seed 42); wizard Step 5 screenshot-verified.** ⚠️ Same-instrument rule: old 94.8%/94.2% are on the 500-case capped instrument — NOT comparable with new 554-case numbers. 12 coverage/dedup tests + 99-file regression green. |
| **Training-start hub-download guard** | `e9bed6e`: `_resolve_model_path` in `routes/training.py` — local paths pass; bare `org/repo` ids resolve via HF hub cache (pass-through) or app `~/.finetune-studio/hf_models/Org__Repo` (rewritten to the local dir, no download); anything else **blocks** with an actionable error unless `allow_download=true`. Kills the silent multi-GB mid-run pull that stalled `c327fa36` #1. 5 tests in `tests/test_training_start_guard.py`; live-verified on fan-dragon (fake repo id → error, run count unchanged, wizard 200). Serena project memory written (`.serena/memories/project-overview.md`, tracked). |

## Next steps

1. **Versions UX polish**: compare manifests side-by-side, copy-pins-to-new-project. Exact surface not built yet — start at `routes/versions.py` + wizard step 6.
2. **Specialized RAG corpus** for the 10 hand-picked files (companion to subset datasets) — build via `POST …/rag/build` on a filtered source set; then pin it in a version manifest.
3. **60ep/r64 variant** on the 48-row subset to noise-check the 91.7%-vs-89.6% split: `POST /api/training/start {project_id:"58d4e331", dataset_id:<a0ae8778>, preset_id:"standard", overrides:{num_epochs:60, lora_rank:64}}`.
4. Bench with eyeball judging per `docs/judging/PROTOCOL.md` on the NEW 554-case full-coverage instrument (f76bf64f/merged) — first comparable full-coverage verdict; then sample the passes.
5. ~~Training-start guard~~ — **done `e9bed6e`** (see State). Next up: Versions UX polish.

## Commands

```bash
# genorbox1
cd ~/work/finetune-studio
make test                     # 1007 passed as of 2026-09-19
make hooks                    # once per clone: installs the commit-msg build-bump hook
bash scripts/install-hooks.sh # same thing; VERSION auto-bumps on every commit
.venv/bin/python -m ruff check src/   # NEVER `make lint` (swallows failures)
make codemap                  # commit docs/CODEMAP.md with code moves

# deploy (scripted route — iron rule)
git push && ssh fan-dragon 'bash -c "cd /home/genortg/finetune-studio && bash update.sh 2>&1 | tail -5; git log --oneline -1; systemctl --user is-active finetune-studio"'

# coverage verify (post-export)
python3 - <<'PY'  # on fan-dragon
import json, glob, re
pairs = [json.load(open(p)) for p in glob.glob("/home/genortg/.finetune-studio/projects/58d4e331/qa/pairs/*.json")]
srcs  = [json.load(open(p)) for p in glob.glob("/home/genortg/.finetune-studio/projects/58d4e331/qa/sources/*.json")]
from collections import Counter
cov = Counter((r["source_id"], r["chunk_idx"]) for r in pairs if r["status"] == "approved")
print("chunk coverage:", len(cov), "/", sum(s["chunk_count"] for s in srcs))
PY

# GPU/ops
ssh fan-dragon 'bash -c "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader"'
# suite run: POST /api/testing/run-suite {suite_path, project_id, max_tokens:256, model_path} — LONG (~10 min for 515+)

# data flow (export now self-fills coverage)
POST /api/projects/{pid}/data-prep/start {source_id, qa_per_chunk, difficulty, style}
POST /api/projects/{pid}/data-prep/qa/bulk {ids, action:"approve"}
GET  /api/projects/{pid}/data-prep/export?fmt=sharegpt&only=approved  # registers dataset, runs coverage gate first
POST /api/training/start {project_id, dataset_id, model_path, num_epochs, lora_rank, export_gguf, gguf_quants}
GET  /api/training/recommend?tier=&base_model=&pairs=&dataset=
```

## Blockers

- None. (Screenshots work via the host browser tool — every visual change gets a rendered-page check now; Genor's standing rule.)
- ComfyUI VRAM the only external GPU pressure; observe, never kill.

## Gotchas worth re-reading before data-prep or training work
See repo `AGENTS.md ## Gotchas`. New: coverage-fill counts `qa` (model pairs) and `qa_coverage_fill` separately — never merge them into one number; extractive fill answers must stay verbatim substrings (asserted in tests).

- 2026-09-19 **Same-instrument scoring only:** yesterday's 95.1% was on a 515-row instrument; the locked 500-case suite gives 94.8% for `f76bf64f`. Never compare strict % across different suite instruments. Retrain on coverage-filled data (e65fafd5) = net-neutral vs f76bf64f.
