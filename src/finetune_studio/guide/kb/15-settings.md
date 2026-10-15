---
id: settings
title: Settings page and compute device
page: settings
keywords: settings update access password sign in login token api token network address port host cors proxy judge test judge auto-judge helper api provider key opencode openai compute device cpu gpu cuda restart tutorial paths storage debug
---
## Purpose
Global settings: updates, tutorial replay, access (password, API tokens, network address), the test judge, the compute device, debug info and storage paths.

## Workflow
1. **Updates:** Check for updates; apply or repair from there.
2. **Compute device:** choose which device the app uses at the **next start** (Auto = best detected GPU, CPU only if none). The choice is saved to a file and applied only at process start; explicit `FTS_GPU_*`, `FTS_DEVICE` or `*_VISIBLE_DEVICES` environment settings override it and show as "overridden". CPU-only masks every card.
3. **Test judge:** choose which connected model judges saved test answers by default, and whether a finished run starts judging by itself (off by default).
4. **Helper model:** choose who does the model work (mining pairs, suites, the Guide): the local GGUF or an API provider (see helper-model).
5. **Access & network:** who can open the web UI. Default: this computer only (127.0.0.1), no password.
   To use it from other devices: set a password first, then choose "every network" and restart the service.
   API tokens let scripts call the API (`Authorization: Bearer <token>`); a token is shown once.
   Forgot the password: on the machine itself run `fts auth set-password`.

## Key controls
- `#btn-update-check` — check for updates.
- `#compute-select` — compute device for the next start.
- `#btn-compute-save` — save the compute device (restart required).
- `#judge-provider` — default test judge model.
- `#helper-card` — the Helper model card (local model or API provider).
- `#access-card` — password, listen address, API tokens, signed-in browsers.
- `#access-pw-new` — set or change the password.
- `#btn-replay-tutorial` — replay the tutorial.

## What to check
After saving a compute device, restart the service; the card shows the effective device.

## Common mistakes
Expecting a compute-device change or a new listen address to apply without a restart.
Opening the instance to the network without a password (every page then shows a red warning).
