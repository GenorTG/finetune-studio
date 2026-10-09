---
id: settings
title: Settings page and compute device
page: settings
keywords: settings update hosting port cors judge test judge auto-judge helper api provider key opencode openai compute device cpu gpu cuda restart tutorial paths storage debug
---
## Purpose
Global settings: updates, tutorial replay, server and hosting options, the test judge, the compute device, debug info and storage paths.

## Workflow
1. **Updates:** Check for updates; apply or repair from there.
2. **Compute device:** choose which device the app uses at the **next start** (Auto = best detected GPU, CPU only if none). The choice is saved to a file and applied only at process start; explicit `FTS_GPU_*`, `FTS_DEVICE` or `*_VISIBLE_DEVICES` environment settings override it and show as "overridden". CPU-only masks every card.
3. **Test judge:** choose which connected model judges saved test answers by default, and whether a finished run starts judging by itself (off by default).
4. **Helper model:** choose who does the model work (mining pairs, suites, the Guide): the local GGUF or an API provider (see helper-model).
5. **Server & Hosting:** port, host, CORS and proxy options.

## Key controls
- `#btn-update-check` — check for updates.
- `#compute-select` — compute device for the next start.
- `#btn-compute-save` — save the compute device (restart required).
- `#judge-provider` — default test judge model.
- `#helper-card` — the Helper model card (local model or API provider).
- `#hosting-port` — server port.
- `#btn-replay-tutorial` — replay the tutorial.

## What to check
After saving a compute device, restart the service; the card shows the effective device.

## Common mistakes
Expecting a compute-device change to apply without a restart.
