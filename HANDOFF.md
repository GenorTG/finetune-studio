# HANDOFF — finetune-studio

Local fine-tune + data-prep WebUI (`src/finetune_studio/`).
Edit on **genorbox1** → push → **fan-dragon** runs `finetune-studio.service` on `:7860`.

**Read first:** `docs/WORKPLAN.md` (order is law) · `docs/PRODUCT-BRIEF.md` (north star) · this file · `AGENTS.md`.

## Mission

Trained models must reliably answer the learned corpus — no lying about trained DB sources. Judge by reading transcripts, not auto-greens. **Data guarantee: every parsed chunk must reach the training dataset (no silent holes).**

## State (verified 2026-09-19 ~19:35 CEST · fan-dragon at `7bb24dc`, service active)

| Area | Status |
|------|--------|
| **Dataset coverage — 100% enforced** | `coverage_fill.py` (`6af6312`): deterministic second pass; chunks the LLM mining missed get extractive pairs (answer quoted verbatim, `origin=coverage_fill`, status approved, no invention). Runs **at end of every mining run** (runner self-heal, `451facd`) **and before every export** (route gate) — a dataset cannot ship with silently-unmined chunks. Live proof: project 58d4e331 went 515→**554 rows**, chunk coverage **131/131 = 100%**, idempotent on re-export. `fill_all_project_gaps()` also surfaces declared-but-lost parsed artifacts (never crash-swallow). |
| **Project versioning (goal 34ff4655)** | `project_versions` table + CRUD (`bc1152b`): immutable manifests pin datasets/source_ids/RAG corpora/runs/base model; monotonic version_number, `parent_version_id` lineage → any old version is a branch base. Subset datasets: `POST /api/projects/{pid}/datasets/subset` hand-picks sources → coverage-filled, per-source row counts, registered (`58d4e331-specialized-picks-…` = 48 rows live). RAG parity gate `GET …/rag/coverage` (129/129 = 100% on the live corpus; shares rag route's corpus root — never re-derive the path). Guided flow page `/projects/{pid}/flow` (`7bb24dc`): 7-step files→QA→dataset→RAG→train→test→version dashboard, nav entry, all through the same APIs as curl. Versions CRUD/lineage live-verified as v1 `302071b6` → v2 `4fca19a4`. Timings: flow page auto-checks all 7 stages on load. |
| **Size-aware training advisor** | `training/preset_advisor.py` (`b78bf15`): parses base size from model name (GGUF quant suffix excluded), scales rank/LR/epochs by base size + dataset size, raises epochs to clear per-tier optimizer-step floor (evidence: 772 steps → 95.1% strict, 257 → 67%). `GET /api/training/recommend?tier=&base_model=&pairs=`. Training page prefills + re-runs when base/dataset changes. 10 evidence-pinned tests. |
| **Auto-suite proven live** | `POST /api/training/runs/{id}/auto-suites/generate` on run `f76bf64f` → 500 cases, deterministic, quality-checked vs the trusted 515-suite (97% normalized-question overlap, 0 degenerate). Selectable via `GET /api/benchmarks/suites?project_id=` (param is `project_id`, **not** `pid`). |
| **Project 58d4e331** | "Vaelindrath Stress": 131 files / 129 sources / 131 chunks, **all parsed, zero failed parses** (18 zero-pair files found + filled=100%). Mining: 581 pairs → 564 approved / 17 rejected (7 id-leak, 10 ambiguous) → dedup → **515**; +39 coverage_fill approved → **554-row dataset on disk now** (551 unique approved questions, all present). |
| **Training evidence** | Run `212035ad` (4 ep, r64) = 67% strict; run `f76bf64f` (12 ep, r128/α256) = 94.8% on the locked 500-case suite; run `e65fafd5` (11 ep, r128/α256, **554-row coverage-filled dataset**) = 94.2% same instrument — **retrain on filled data was net-neutral** (fixed 23, regressed 25). Production model stays `f76bf64f/merged`; GGUF q8_0 exported for both. Eyeball: fails are genuine recall misses (digit/price confusion + similar-fact swaps); no judge false positives. |
| **Fleet (Genor's rule, 3 models)** | 4B safetensors trainer; helper = Qwen3-8B Q5_K_M GGUF (real header topology, `-1` = all layers, no magic 99); 27B abliterated GGUF. |
| **Deployed UX** | Named toasts both paths, counted upload toasts, WCAG-pass pills (5.16–10.19:1), dim-token contrast, mobile toast strip; deployed assets `app.css?v=29` + `app.js?v=21` (served + verified). |

## Next steps

1. Versions UX polish (compare manifests, copy-pins-to-new) + train a real specialized model from the 48-row subset build as the flow's end-to-end demo.
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

- 2026-09-19 **Same-instrument scoring only:** yesterday's 95.1% was on a 515-row instrument; the locked 500-case suite gives 94.8% for `f76bf64f`. Never compare strict % across different suite instruments. Retrain on coverage-filled data (e65fafd5) = net-neutral vs f76bf64f.
