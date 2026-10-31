---
id: project-models
title: Project models (trained exports)
page: project_models
keywords: project models trained exports adapters loras merged checkpoints gguf files run expand copy path delete model what did i train
---
## Purpose
**Models** (`/projects/{pid}/models`) lists everything this project's training produced: adapter weights, merged checkpoints and exported GGUF / safetensors files. It is the per-project view; **My models** (`/models`) is the machine-wide list.

## Workflow
1. Open the project and press **Models** in its navigation, or the **All models →** button on the project overview.
2. Read the row: name, format, size, source run and creation time.
3. Click a **name** to expand the row: the run that produced it, its training settings and the files on disk with their sizes.
4. Use **Copy path** to put the model's path on the clipboard for scripts, clients or another tool.
5. To produce a new file (GGUF quant, merged, abliterated) go to **Export**; to remove one, use the row's action.

## Key controls
- `#trained-exports-table` — trained exports of this project.
- `#trained-exports-table-scroll` — horizontal scroll region for the export table.

## What to check
Each row names a real run and the file exists at the copied path. A finished training run that produced nothing appears under Training with its error, not here.

## Common mistakes
Expecting a base model here (they live in **My models**); assuming a merged checkpoint and its adapter are the same file; expanding a row and reading the run's settings as the model's own settings.
