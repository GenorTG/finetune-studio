---
id: training
title: Training page (step 3)
page: training
keywords: train training start base model dataset preset smoke balanced precision overkill advisor lora rank learning rate batch accumulation sequence merge unsloth checkpoints live status runs production stop
---
## Purpose
Teach a base model your dataset. Hours, not minutes for large bases; the page keeps live progress and survives reloads.

## Workflow
1. Pick a **Training route** (see training-routes). SFT is the default for Q&A pairs.
2. Pick the **Base model** (Transformers/safetensors only; GGUF is inference-only).
3. Pick data: **From this project** (the dataset exported on Pairs) or **Upload my own** (JSONL/JSON/CSV). Picking one runs the dataset check; Start stays disabled until a dataset is chosen.
4. Pick a **preset** by effort (Smoke, Baseline, Precision, Overkill). The page asks the preset advisor and fills epochs, rank, alpha, learning rate, batch, accumulation and warmup, with the optimizer-step arithmetic and warnings shown. Values stay editable.
5. Check the advisory for a too-few-steps warning, then **Start training**. The helper model is unloaded automatically before training.
6. Watch **Live status** (step, loss, ETA). Afterwards: Testing, then Export; set a run as production from Past runs.

## Key controls
- `#train-form` — the training form.
- `#train-base-model` — base model dropdown.
- `#dataset-select` — registered project datasets.
- `#dataset-upload` — upload your own dataset file.
- `#dataset-health` — dataset check result.
- `#training-preset` — effort preset; fills the fields below via the advisor.
- `#preset-advisory` — the advisor's arithmetic, notes and warnings.
- `[name="num_epochs"]` — epochs (form default 4).
- `[name="lora_rank"]` — LoRA rank (default 64).
- `[name="learning_rate"]` — learning rate (default 8e-5).
- `[name="batch_size"]` — batch size (default 2).
- `[name="max_seq_length"]` — max tokens per example (default 2048).
- `#gradient-accum-steps` — gradient accumulation (default 1).
- `#eval-steps` — evaluate on the held-out split every N steps (0 = never).
- `#early-stopping-check` — stop when validation loss stops improving (needs eval steps above 0).
- `#merge-on-save-check` — also save a merged standalone model (ticked by default; roughly base-model size).
- `#system-prompt-mode` — bake, runtime or none.
- `#start-btn` — start training; `#stop-btn` stops the active run.
- `#train-status` — live status text.
- `#past-runs-table` — all runs, with Details, Set production, Open in inference and Export.

## What to check
Final loss below 0.5 on small factual datasets, status done, a merged model present. With a tiny dataset make sure early stopping did not end the run before the planned steps. On out-of-memory the page shows "Out of GPU memory" with the knobs to lower (batch, max sequence).

## Common mistakes
Choosing Smoke and expecting recall (it only proves the plumbing); trusting epochs instead of optimizer steps (see training-settings); training while another heavy GPU job runs.
