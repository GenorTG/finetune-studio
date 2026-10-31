---
id: models
title: Model library and My models
page: models_library
keywords: model library download hugging face safetensors gguf base model my models local models disk trained exports delete search qwen gemma llama
---
## Purpose
**Model library** (`/models/explore`) searches Hugging Face, downloads base models, and lists/deletes downloads in this app's local cache. **My models** (`/models`) lists everything on disk: downloaded bases, trained exports, helpers.

## Workflow
1. Search (for example `Qwen3-0.6B`) and click **Download** on the card, then confirm. A 0.6B model is ~1.5 GB (about 20 s).
   The download runs in the background: the **Activity** drawer shows its progress, and **Cancel download** there stops it.
2. Use **safetensors** models for training (LoRA/QLoRA); **GGUF** is inference-only (Chat, Inference, helper).
3. The downloaded model appears in the Training base-model dropdown after the page is reloaded.
4. Under **Downloaded to this computer**, use **Delete** to remove only a model in this app's Hugging Face cache. A loaded or loading model cannot be deleted; unload it first. A download still in progress cannot be deleted; wait or cancel it first.
5. A model copied in by hand shows up after **Rescan folders** on My models (see the my-models entry).
6. CLI parity: `fts models search`, `local`, `download`, `download-status`, `cancel-download`, and `delete`; the no-argument `fts models` still lists installed models.

**View** opens the model details. **Downloaded files** lists files already in this app's cache; it is not a remote file picker. **Download** downloads the whole repository after confirmation.

## Key controls
- `#hf-q` — search box.
- `#hf-task` — task filter.
- `#hf-sort` — sort order.
- `#hf-results` — result list.
- `#hf-local-models` — downloaded model list and its delete actions.
- `#hf-local-refresh` — refresh the local download list.

## What to check
Format and size before downloading; free disk space.

## Common mistakes
Downloading a GGUF and expecting it in the training dropdown; choosing a base too large for the GPU.
