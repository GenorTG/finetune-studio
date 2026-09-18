# HANDOFF — finetune-studio

Local fine-tune + data-prep WebUI (`src/finetune_studio/`).
Edit on **genorbox1** → push → **fan-dragon** runs `finetune-studio.service` on `:7860`.

**Read first:** `docs/WORKPLAN.md` (order is law) · `docs/PRODUCT-BRIEF.md` (north star) · this file · `AGENTS.md`.

## Mission

Trained models must reliably answer the learned corpus — no lying about trained DB sources. Judge by reading transcripts, not auto-greens.

## State (verified 2026-09-18 ~15:45 CEST · genorbox1 `5e17853` = fan-dragon)

| Area | Status |
|------|--------|
| **Fleet** (Genor's rule) | 4B safetensors trainer (`hf_models/Qwen__Qwen3-4B`), helper = **Qwen3-8B Q5_K_M GGUF** (5.85GB, `hf_models/Qwen__Qwen3-8B-GGUF`), 27B `models/gguf/Qwen3.8-27B-abliterated-Q4_K_M.gguf` — only 3, helper seat <27B. |
| **Helper load params** | **32k ctx, 99 layers GPU, q8_0 KV cache** (`type_k/v` now in `_LOADER_KEYS`, defaults in `models/helper.py`). ~13.7GB VRAM. ComfyUI may hold ~10-15GB — never kill it. |
| **Project 57dc3fd7** | "Aethermere Corpus" — 3 fact-dense files (lore 24.9KB/32 chunks, chronicle 10.6KB/14, gazetteer 4.6KB/6). Parse fidelity 1.000/1.000/0.999, 20/20 fact probes. |
| **Q&A pairs** | **613 mined (389 lore/164 chronicle/60 gazetteer), all approved, 593 unique** — two passes (medium/socratic 8-10 per chunk + hard/direct 5). |
| **Dataset + suite** | `57dc3fd7-sharegpt-approved.jsonl` (593 pairs) registered; `suite-full-corpus.json` (593 cases) at project root. Built together from same pairs (Genor's method). |
| **Training run 6f64c46a** | Qwen3-4B, LoRA r64/α128, lr 2e-4, 4 epochs, bs2×ga4 → 268 steps, loss 5.56→0.23 (mean 0.97). Unsloth, merged. |
| **Export** | `output/projects/57dc3fd7/runs/6f64c46a/gguf/`: model-q8_0.gguf (4.0GB), model-q5_k_m.gguf (2.7GB), f16 (7.5GB). |
| **RESULT** | 593-case suite on trained Q8: **auto 89.0% pass (528/593)**. Human-judged: 45 genuine fails, 4 semantic passes, 4 partial inside the 53 auto-fails; 0 FP in 25-pass sample → **true strict ≈ 90%**. Auto-scoring is trustworthy here (bare-answer FP class gone with trained-in-style answers). Digits and named entities are the residual failure class. |
| **Fixes this session** | `fc51313` upload progress bar (pct+bytes+names, XHR onprogress); `9ffd72e` helper→8B + row migration; `3904d99` 32k ctx + KV quant; `5e17853` data_path kept in source manifest after runs (re-runs 404'd before). |

## Next steps

1. Re-judge the remaining unsampled passes if Genor wants a stricter number (read transcripts in `/tmp/suite_new.json` on fan-dragon).
2. If >90% wanted: 3rd mining pass + 6 epochs retrain; or bump helper mine quality via agent-chat mode (`/projects/{pid}/chat?mode=agent`, now 32k).
3. Auto-judge wiring still gated per `docs/judging/PROTOCOL.md`; this run supports it.
4. Consider swapping helper seat to `model-q8_0.gguf` of trained runs for domain-aware mining.
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
# helper load: POST /api/providers/local-default/load (row already 32k/q8KV/99L)
# suite run: POST /api/testing/run-suite {suite_path, project_id, max_tokens:300} — LONG; run nohup'd ON fan-dragon (local exec gets SIGTERM'd)

# data flow
POST /api/projects/{pid}/files/upload            # upload+parse (auto-promote)
POST /api/projects/{pid}/data-prep/start {source_id, qa_per_chunk, difficulty, style}
POST /api/projects/{pid}/data-prep/qa/bulk {ids, action:"approve"}
GET  /api/projects/{pid}/data-prep/export?fmt=sharegpt&only=approved  # registers dataset
python3 scripts/build_full_qa_suite.py <project_dir> <out.json>
POST /api/training/start {project_id, dataset_id, model_path, ...}
```

## Blockers

- None. ComfyUI VRAM is the only external pressure; coordinate before big loads.

## Gotchas worth re-reading before training work
See repo `AGENTS.md ## Gotchas` — especially: unload helper/engines before a train; forked-workers deadlock guard is pinned in `training/engine.py`; merge nf4 rule; fish-quoting on fan-dragon ssh.
