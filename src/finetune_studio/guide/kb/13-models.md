---
id: models
title: Model library and My models
page: models_library
keywords: model library download hugging face safetensors gguf base model my models local models disk trained exports delete search qwen gemma llama
---
## Purpose
**Model library** (`/models/explore`) searches and downloads base models from Hugging Face. **My models** (`/models`) lists everything on disk: downloaded bases, trained exports, helpers.

## Workflow
1. Search (for example `Qwen3-0.6B`) and click **Download** on the card, then confirm. A 0.6B model is ~1.5 GB (about 20 s).
2. Use **safetensors** models for training (LoRA/QLoRA); **GGUF** is inference-only (Chat, Inference, helper).
3. The downloaded model appears in the Training base-model dropdown after the page is reloaded.
4. A model stored outside `models/` and `output/` needs a `model_dirs_extra` entry plus `POST /api/models/refresh`.

## Key controls
- `#hf-q` — search box.
- `#hf-task` — task filter.
- `#hf-sort` — sort order.
- `#hf-results` — result list.

## What to check
Format and size before downloading; free disk space.

## Common mistakes
Downloading a GGUF and expecting it in the training dropdown; choosing a base too large for the GPU.
