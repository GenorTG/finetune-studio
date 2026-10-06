---
id: troubleshooting
title: Common problems and fixes
page: dashboard
keywords: error fail failed out of memory oom stuck cannot start disabled blocked 409 no dataset helper not loaded slow gpu cpu fallback degraded empty parsed troubleshoot problem fix
---
## Purpose
Quick diagnosis for the problems users hit most.

## Workflow
- **Start training is disabled:** no dataset chosen. Pick one under *From this project* or upload one.
- **Export of approved pairs blocked (409):** a file produced no usable pair; the message names it. Delete the file or export without it.
- **Out of GPU memory:** lower batch size or max sequence on Training; unload other models; the page lists the knobs. Other GPU programs count against VRAM.
- **Helper not loaded / missing:** see helper-model; check `/api/providers/helper/status`.
- **Parsed text empty:** re-parse on Files; a missing document-parser module (`bash install.sh --check`) is the usual cause for `.xls`, `.pptx`, `.rtf`.
- **Model dropdown lacks a model:** reload the Training page after a download; models outside `models/` or `output/` need `model_dirs_extra` and `POST /api/models/refresh`.
- **Running on CPU on a GPU machine:** `GET /api/system/accelerator` shows `degraded_reason`; the Settings compute-device card shows the effective device.

## Key controls
- `#start-btn` — Start training (disabled without a dataset).
- `#dataset-health` — dataset check messages.

## What to check
The header chip never lies about the loaded model; long jobs survive a page reload.

## Common mistakes
Retrying the same failing action repeatedly instead of reading the stated reason.
