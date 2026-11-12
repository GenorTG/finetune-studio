---
id: my-models
title: My models (models on disk)
page: my_models
keywords: my models local models on disk rescan folders scan copied by hand missing model not listed trained exports helpers folders hf cache
---
## Purpose
**My models** (`/models`) lists every model the app found on disk: trained exports, base models, Hugging Face repos and local helpers, with a filter by kind.

## Workflow
1. Filter by kind with the pills above the table.
2. After copying or deleting a model by hand, press **Rescan folders**: the app scans its model folders again and the page reloads with what is on disk now.
3. Scanned folders: `models/` and `output/` in the app folder, the Hugging Face cache, `~/.finetune-studio/hf_models` (Model library downloads) and `~/.finetune-studio/shared_models` (embedders and rerankers). A model anywhere else is not listed.

## Key controls
- `#models-rescan` — scan the model folders again.
- `#models-index-table` — the models on disk.

## What to check
A copied model sits in one of the scanned folders, in its own sub-folder with its `config.json` (safetensors) or as a `.gguf` file.

## Common mistakes
Expecting a model outside the scanned folders to appear; forgetting to rescan after copying one in by hand.
