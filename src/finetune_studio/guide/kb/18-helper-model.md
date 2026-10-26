---
id: helper-model
title: The helper model (main assistant)
page: pairs
keywords: helper model assistant main assistant gemma load unload gguf mining generation agent guide no helper missing configured silent switch vram idle api provider connection key opencode deepseek mimo openai remote cloud settings
---
## Purpose
The helper is the **main assistant** (Settings → AI assistant): a local GGUF (recommended: Google's Gemma 4 12B QAT Q4_0, one click there) or an API connection. It mines Q&A pairs, writes study sets, quizzes and preference pairs, judges tests and powers the Guide, unless a job is set to its own connection.

## Workflow
1. The Pairs page loads it before mining (about 50 s, ≈12–14 GiB VRAM); the guide loads it when it needs it. A local helper serves one call at a time.
2. Training unloads it first; the wizard unloads it before training.
3. **API alternative:** on Settings → AI assistant press *Add API connection*, enter name, URL, key and model (the preset fills OpenCode Go / OpenAI / OpenRouter), press Test connection, Save, then make it the main assistant (or pick it for single jobs). An API connection needs no VRAM, loads nothing and serves several calls at once (parallel calls, default 8), so mining many files is far faster; the Jobs table says what each job sends to that provider. The key stays on this computer, stored encrypted and never shown again. Making the local model the main assistant again unloads nothing extra; the GGUF loads on first use.
4. If the file is missing, `GET /api/providers/helper/status` lists installed GGUFs and the Pairs page lets you choose one.

## Key controls
- `#dp-helper-label` — the helper name shown on the Pairs page.
- `#prep-helper-picker` — pick a different installed GGUF as helper.

## What to check
The helper is never silently replaced by whatever is loaded elsewhere; a clear error is shown instead.

## Common mistakes
Expecting mining to work with no helper GGUF installed; loading a different model on Inference and assuming it is the helper.
