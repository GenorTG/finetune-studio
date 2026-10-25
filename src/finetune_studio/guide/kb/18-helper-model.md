---
id: helper-model
title: The helper model
page: pairs
keywords: helper model gemma load unload gguf mining generation agent guide no helper missing configured silent switch vram idle api provider key opencode deepseek mimo openai remote cloud settings
---
## Purpose
A dedicated local GGUF (recommended: Google's Gemma 4 12B QAT Q4_0, one click under Settings → Helper model) or an API provider mines Q&A pairs, builds LLM-assisted test suites and powers the guide's Agent mode.

## Workflow
1. The Pairs page loads it before mining (about 50 s, ≈12–14 GiB VRAM); the guide loads it when it needs it. A local helper serves one call at a time.
2. Training unloads it first; the wizard unloads it before training.
3. **API alternative:** on Settings → Helper model choose *API provider*, enter URL, key and model (the preset fills OpenCode Go / OpenAI / OpenRouter), press Test connection, then Save. The API helper needs no VRAM, loads instantly and serves several calls at once (Settings → Helper model → parallel calls, default 8), so mining many files is far faster; but the text of your documents and your questions are sent to that provider; the key stays on this computer, stored encrypted (never shown again after saving). Switching back to *Local model* frees nothing extra and the GGUF loads on first use again.
4. If the file is missing, `GET /api/providers/helper/status` lists installed GGUFs and the Pairs page lets you choose one.

## Key controls
- `#dp-helper-label` — the helper name shown on the Pairs page.
- `#prep-helper-picker` — pick a different installed GGUF as helper.

## What to check
The helper is never silently replaced by whatever is loaded elsewhere; a clear error is shown instead.

## Common mistakes
Expecting mining to work with no helper GGUF installed; loading a different model on Inference and assuming it is the helper.
