# HANDOFF — finetune-studio

Local fine-tune + data-prep WebUI (`src/finetune_studio/`).
Edit on **genorbox1** → push → **fan-dragon** runs `finetune-studio.service` on `:7860`.

**Read first:** `docs/WORKPLAN.md` (order is law) · `docs/PRODUCT-BRIEF.md` (north star) · this file · `AGENTS.md`.

## Mission

Trained models must reliably answer the learned corpus — no lying about trained DB sources. Judge by reading transcripts, not auto-greens. **Data guarantee: every parsed chunk must reach the training dataset (no silent holes).**

## State (verified 2026-09-19 ~17:45 CEST · fan-dragon at `451facd`, service active)

| Area | Status |
|------|--------|
| **Dataset coverage — 100% enforced** | `coverage_fill.py` (`6af6312`): deterministic second pass; chunks the LLM mining missed get extractive pairs (answer quoted verbatim, `origin=coverage_fill`, status approved, no invention). Runs **at end of every mining run** (runner self-heal, `451facd`) **and before every export** (route gate) — a dataset cannot ship with silently-unmined chunks. Live proof: project 58d4e331 went 515→**554 rows**, chunk coverage **131/131 = 100%**, idempotent on re-export. `fill_all_project_gaps()` also surfaces declared-but-lost parsed artifacts (never crash-swallow). |
| **Size-aware training advisor** | `training/preset_advisor.py` (`b78bf15`): parses base size from model name (GGUF quant suffix excluded), scales rank/LR/epochs by base size + dataset size, raises epochs to clear per-tier optimizer-step floor (evidence: 772 steps → 95.1% strict, 257 → 67%). `GET /api/training/recommend?tier=&base_model=&pairs=`. Training page prefills + re-runs when base/dataset changes. 10 evidence-pinned tests. |
| **Auto-suite proven live** | `POST /api/training/runs/{id}/auto-suites/generate` on run `f76bf64f` → 500 cases, deterministic, quality-checked vs the trusted 515-suite (97% normalized-question overlap, 0 degenerate). Selectable via `GET /api/benchmarks/suites?project_id=` (param is `project_id`, **not** `pid`). |
| **Project 58d4e331** | "Vaelindrath Stress": 131 files / 129 sources / 131 chunks, **all parsed, zero failed parses** (18 zero-pair files found + filled=100%). Mining: 581 pairs → 564 approved / 17 rejected (7 id-leak, 10 ambiguous) → dedup → **515**; +39 coverage_fill approved → **554-row dataset on disk now** (551 unique approved questions, all present). |
| **Training evidence** | Run `212035ad` (4 ep, r64) = 67% strict; run `f76bf64f` (12 ep, r128/α256, 2e-4) = **95.1% (490/515)** + q8_0 GGUF (4.3 GB). Eyeball per protocol: 25 fails = 24 real misses + 1 known cross-source conflict; no judge false positives. Auto-scoring trustworthy here. |
| **Fleet (Genor's rule, 3 models)** | 4B safetensors trainer; helper = Qwen3-8B Q5_K_M GGUF (real header topology, `-1` = all layers, no magic 99); 27B abliterated GGUF. |
| **Deployed UX** | Named toasts both paths, counted upload toasts, WCAG-pass pills (5.16–10.19:1), dim-token contrast, mobile toast strip; deployed assets `app.css?v=29` + `app.js?v=21` (served + verified). |

## Next steps

1. **Retrain on the 554-row dataset** (advisor tier=precision proposes 12-13 ep r128 — new coverage-fill pairs from the 18 previously-missed files may lift >95.1%): `/api/training/start` with dataset id from the 554-row export, then auto-suite → strict test.
2. **Finish rag `import_bundle` WIP** (~274 lines, committed unverified in `45077f2`: `rag.py` + `data/rag_portable/store.py`) — complete or strip.
3. Full-corpus bench with eyeball judging per `docs/judging/PROTOCOL.md`; then sample the 490+ passes.
4. Visual UX strict pass with real screenshots (advisory panel, toasts) — needs a paired computer-capable node; verified so far only via curl/DOM.
5. Popups/responsive polish second pass (asked; icon/message/contrast shipped).

## Commands

```bash
# genorbox1
cd ~/work/finetune-studio
make test                     # 1007 passed as of 2026-09-19
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

- Screenshots/visual pass needs a paired computer-capable node (Genor to pair).
- ComfyUI VRAM the only external GPU pressure; observe, never kill.

## Gotchas worth re-reading before data-prep or training work
See repo `AGENTS.md ## Gotchas`. New: coverage-fill counts `qa` (model pairs) and `qa_coverage_fill` separately — never merge them into one number; extractive fill answers must stay verbatim substrings (asserted in tests).
