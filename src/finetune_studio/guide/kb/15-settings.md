---
id: settings
title: Settings page and compute device
page: settings
keywords: settings update hosting port cors judge benchmark compute device cpu gpu cuda restart tutorial paths storage debug
---
## Purpose
Global settings: updates, tutorial replay, server and hosting options, the benchmark judge, the compute device, debug info and storage paths.

## Workflow
1. **Updates:** Check for updates; apply or repair from there.
2. **Compute device:** choose which device the app uses at the **next start** (Auto = best detected GPU, CPU only if none). The choice is saved to a file and applied only at process start; explicit `FTS_GPU_*`, `FTS_DEVICE` or `*_VISIBLE_DEVICES` environment settings override it and show as "overridden". CPU-only masks every card.
3. **Benchmark judge:** choose how transcripts are judged.
4. **Server & Hosting:** port, host, CORS and proxy options.

## Key controls
- `#btn-update-check` — check for updates.
- `#compute-select` — compute device for the next start.
- `#btn-compute-save` — save the compute device (restart required).
- `#judge-mode` — benchmark judge mode.
- `#hosting-port` — server port.
- `#btn-replay-tutorial` — replay the tutorial.

## What to check
After saving a compute device, restart the service; the card shows the effective device.

## Common mistakes
Expecting a compute-device change to apply without a restart.
