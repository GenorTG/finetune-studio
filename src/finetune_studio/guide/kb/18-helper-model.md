---
id: helper-model
title: The helper model
page: pairs
keywords: helper model gemma load unload gguf mining generation agent guide no helper missing configured silent switch vram idle
---
## Purpose
A dedicated local GGUF (Gemma 4 12B Uncensored Q4_K_M by default) mines Q&A pairs, builds LLM-assisted test suites and powers the guide's Agent mode.

## Workflow
1. It loads automatically when a prep job or the guide needs it (about 50 s, ≈12–14 GiB VRAM) and reloads if the idle timer unloaded it.
2. Training unloads it first; the wizard unloads it before training.
3. If the file is missing, `GET /api/providers/helper/status` lists installed GGUFs and the Pairs page lets you choose one.

## Key controls
- `#dp-helper-label` — the helper name shown on the Pairs page.
- `#prep-helper-picker` — pick a different installed GGUF as helper.
- `#prep-helper-ctx` — helper context window.

## What to check
The helper is never silently replaced by whatever is loaded elsewhere; a clear error is shown instead.

## Common mistakes
Expecting mining to work with no helper GGUF installed; loading a different model on Inference and assuming it is the helper.
