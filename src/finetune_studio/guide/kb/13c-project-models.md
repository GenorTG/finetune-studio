---
id: project-models
title: Project models (files a run produced)
page: project_models
keywords: project models where are my trained models trained model list adapters loras merged checkpoints model files copy path open in inference load trained model expand row what did i train produced by the run
---
## Purpose
**Models** (`/projects/{pid}/models`) lists everything this project's training produced: adapter weights, merged checkpoints and exported GGUF / safetensors files. It is the per-project view; **My models** (`/models`) is the machine-wide list.

## Workflow
1. Open the project overview and press the **Models** tile (it counts the exports and their size) or the **All models →** button in its models card. The project navigation has no separate Models link.
2. Read the row: name, format, size, source run and creation time.
3. Click a **name** to expand the row: the run that produced it, its training settings and the files on disk with their sizes, plus **View run →**.
4. Use **Copy path** to put the model's path on the clipboard for scripts, clients or another tool.
5. Use **▶ Open in inference** to load the model and go to Inference to chat with it (a failed load says why).
6. To produce a new file (GGUF quant, merged, abliterated, Heretic) go to **Export**. This page has no remove action: its row actions are Copy path and Open in inference.

## Key controls
- `#trained-exports-table` — trained exports of this project.
- `#trained-exports-table-scroll` — horizontal scroll region for the export table.
- `[name="export-copy-path"]` — a row's Copy path button.
- `[name="export-open-inference"]` — a row's Open in inference button.

## What to check
Each row names a real run and the file exists at the copied path. A finished training run that produced nothing appears under Training with its error, not here.

## Common mistakes
Expecting a base model here (they live in **My models**); assuming a merged checkpoint and its adapter are the same file; expanding a row and reading the run's settings as the model's own settings.
