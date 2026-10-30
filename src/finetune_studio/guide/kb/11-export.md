---
id: export
title: Export page (step 4)
page: export
keywords: export gguf quant q4_k_m q6_k q8_0 f16 merged abliterated herectic heretic uncensored decensor refusal llama.cpp ollama lm studio quantization overwrite adapter imatrix importance matrix calibration
---
## Purpose
Take a finished run out of the studio: GGUF for llama.cpp, Ollama and LM Studio; Merged (full safetensors); Abliterated; or Heretic (the optional tool that removes refusals properly). Adapter-only runs are merged onto their base automatically at export time.

## Workflow
1. Select a finished run in **Training runs**.
2. Tick formats: GGUF with q4_k_m is pre-ticked (recommended). Tick more quants if wanted (f16, q8_0, q6_k, q5_k_m, q4_k_m, q3_k_m, q2_k).
3. **Export selected** (about 20–40 s for a small model). Files appear under *Trained exports* and become selectable in Chat and Testing.
4. **Overwrite existing** re-exports over old files.
5. **Importance matrix (imatrix)**: before quantizing, the app measures which weights the run's own training data relies on (llama-imatrix, up to 128 chunks of 512 tokens) and keeps those more precise. Same file size, better q4/q3/q2 quality, a few minutes extra. The files land in the run's `gguf-imatrix/` folder beside the plain `gguf/` ones; f16 and q8_0 are written without it, so tick at least one smaller quant.

## Key controls
- `#export-runs` — training runs to choose from.
- `#gguf-quants` — quantization check boxes.
- `#gguf-imatrix` — quantize with an importance matrix measured on the training data.
- `#export-btn` — start the export (disabled until a run is selected).
- `#export-force` — overwrite existing exports.
- `#trained-exports-table` — finished exports.

## What to check
Two GGUF files for two quants under the run's `gguf/` folder. Prefer q6_k or q4_k_m for small models. Q8_0 once crashed the whole service on some models via a native llama.cpp bug; the loader now caps the micro-batch at 256, but there is no reason to pick the riskiest quant for a 0.6B model.

## Common mistakes
Exporting GGUF when it is marked unavailable (the page shows why); ticking the importance matrix for a run whose training file was deleted (refused with the reason); ticking Abliterated without understanding it removes refusal behaviour.

Terminal alternative: `fts export start <run-id> --project <project-id>` queues the same export job; `fts export list`, `status`, and `cancel` inspect or stop it. Run `fts export start --help` for quant, imatrix, merge, and overwrite options.
Ticking Heretic when it is greyed out (Settings → Heretic installs it; the page says why it cannot run). Heretic also needs more GPU time than Abliterated: it searches parameters across many trials instead of applying one fixed strength.
