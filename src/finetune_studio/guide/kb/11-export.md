---
id: export
title: Export page (step 5)
page: export
keywords: export gguf quant q4_k_m q6_k q8_0 f16 merged abliterated llama.cpp ollama lm studio quantization overwrite adapter
---
## Purpose
Take a finished run out of the studio: GGUF for llama.cpp, Ollama and LM Studio; Merged (full safetensors); or Abliterated. Adapter-only runs are merged onto their base automatically at export time.

## Workflow
1. Select a finished run in **Training runs**.
2. Tick formats: GGUF with q4_k_m is pre-ticked (recommended). Tick more quants if wanted (f16, q8_0, q6_k, q5_k_m, q4_k_m, q3_k_m, q2_k).
3. **Export selected** (about 20–40 s for a small model). Files appear under *Trained exports* and become selectable in Chat and Testing.
4. **Overwrite existing** re-exports over old files.

## Key controls
- `#export-runs` — training runs to choose from.
- `#gguf-quants` — quantization check boxes.
- `#export-btn` — start the export (disabled until a run is selected).
- `#export-force` — overwrite existing exports.
- `#trained-exports-table` — finished exports.

## What to check
Two GGUF files for two quants under the run's `gguf/` folder. Prefer q6_k or q4_k_m for small models. Q8_0 once crashed the whole service on some models via a native llama.cpp bug; the loader now caps the micro-batch at 256, but there is no reason to pick the riskiest quant for a 0.6B model.

## Common mistakes
Exporting GGUF when it is marked unavailable (the page shows why); ticking Abliterated without understanding it removes refusal behaviour.
