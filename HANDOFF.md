# HANDOFF — finetune-studio

Local fine-tune + data-prep WebUI (`src/finetune_studio/`).
Edit on **genorbox1** → push → **fan-dragon** runs `finetune-studio.service` on `:7860`.

**Read first:** `docs/WORKPLAN.md` (order is law) · `docs/PRODUCT-BRIEF.md` (north star) · this file · `AGENTS.md`.

## Mission

Trained models must reliably answer the learned corpus — no lying about trained DB sources. Judge by reading transcripts, not auto-greens.

## State (verified 2026-09-18 ~18:30 CEST · genorbox1 = fan-dragon `a13992c`+3)

| Area | Status |
|------|--------|
| **Fleet** (Genor's rule) | 4B safetensors trainer, helper = **Qwen3-8B Q5_K_M GGUF** (36/36 layers GPU, ctx 32768), 27B abliterated — only 3. Helper load now uses real GGUF header topology (`-1` = all layers, block_count read from file) — **no magic 99 anywhere** (`9bd2913`). |
| **Project 58d4e331** | "Vaelindrath Stress" — **131 fictional files / 43 extensions, ≥2 per ext, byte-unique** (generator: `scripts/gen_vaelindrath_corpus.py`, artifacts `/tmp/vael`). Parsed: 129 sources / 131 chunks / 65KB, all ready incl. real OLE2 `.doc` (olefile) on fan-dragon. |
| **Q&A pairs** | **581 mined** (hard/direct ×8/chunk, one run per source; ~15 min on 8B helper all-GPU). Triage: 564 approved / 17 rejected (7 identifier-leak + 10 ambiguous-bare). |
| **Dataset + suite** | `58d4e331-sharegpt-approved.jsonl` (**515 dedup rows**) registered as dataset `6d7680a0`; `full-ingested-corpus.json` suite auto-built (515 cases). Built together from same pairs (Genor's method). |
| **Training (2 runs this pass)** | Run `212035ad` (4 ep, r64): **67% strict** — failure class = digit scrambling on under-memorized prices/counts. Retrained run `f76bf64f` (**12 ep, r128/α256**): **95.1% strict (490/515)**. Both auto-merged + q8_0 GGUF exported (4.3GB). |
| **Judging (eyeballed 25 fails + passes)** | 24 genuine recall misses (confused similar facts), 1 known cross-source conflict question. Passes sampled = real. No false-positive class found. |
| **Bugs found by the stress pass (all fixed + deployed)** | `e87c9db` olefile .doc parser · `767099e` delete→re-upload revival (trash row poisoned re-uploads) · `7394318` placeholder-parse retry guard · `ba1e561`+`a13992c` GGUF export silently no-ops without merge (now auto-merges on **both** TRL paths, failures land in run row) · `2dc90ae` activity UX pass. |
| **UX pass deployed** | Activity feed: human messages ("Start Q&A mining") + project as separate badge; gear icon → pulse SVG everywhere; kind/status/border contrast kit; top-right decluttered (conn-status chip dropped, release label folded into tooltip). Verified via API + activity feed live during 129-run mining. |

## Next steps

1. Final batch: `/api/benchmarks/*` full-corpus bench run with judge eyeball per `docs/judging/PROTOCOL.md`; then judge the 490 passes sample (auto-scoring trustworthy here, spot-check).
2. Params ladder if <99% wanted: 16-24 epochs or full-corpus augmentation pass (`scripts/augment_dataset.py` precedent, Aethermere 26→93%).
3. Unstaged in tree: rag import_bundle ~274 lines (`rag.py`/`store.py`) + corpus-generator dedup injection — finish + commit separately.
4. UX second pass: popups/toasts, responsive polish (Genor asked; icon/message/contrast done).
5. HANDOFF rewrite at next milestone; archive old to `docs/archive/`.

## Commands

```bash
# genorbox1
cd ~/work/finetune-studio
make test                     # focused: .venv/bin/python -m pytest tests/test_X.py -v
.venv/bin/python -m ruff check src/   # NEVER `make lint` (swallows failures)
make codemap                  # commit docs/CODEMAP.md with code moves

# deploy (scripted route — iron rule)
git push && ssh fan-dragon 'bash -c "cd /home/genortg/finetune-studio && bash update.sh 2>&1 | tail -5; git log --oneline -1; systemctl --user is-active finetune-studio"'

# GPU/ops
ssh fan-dragon 'bash -c "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader"'
# helper load: POST /api/providers/local-default/load
# suite run: POST /api/testing/run-suite {suite_path, project_id, max_tokens:256, model_path} — LONG (~10 min for 515)

# data flow
POST /api/projects/{pid}/files/upload            # upload+parse (auto-promote)
POST /api/projects/{pid}/data-prep/start {source_id, qa_per_chunk, difficulty, style}
POST /api/projects/{pid}/data-prep/qa/bulk {ids, action:"approve"}
GET  /api/projects/{pid}/data-prep/export?fmt=sharegpt&only=approved  # registers dataset
POST /api/training/start {project_id, dataset_id, model_path, num_epochs, lora_rank, export_gguf, gguf_quants}
```

## Blockers

- None. Deep-dive detail for this pass kept at `docs/judging/2026-09-18-vaelindrath-summary.md`.
- ComfyUI VRAM the only external pressure; parrot/observe, never kill.

## Gotchas worth re-reading before training work
See repo `AGENTS.md ## Gotchas` — now includes: olefile must be installed on fan-dragon before re-parsing .doc; upload dedup ignores deleted rows so the revival path matters; merge_on_save off + export_gguf was a silent no-op on both TRL save paths.
