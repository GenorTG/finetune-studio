---
id: chat
title: Chat page (Test and Agent / Guide modes)
page: chat
keywords: chat test mode agent mode guide helper model load rag context system prompt temperature talk to model ask questions app help
---
## Purpose
Two modes. **Test** chats with a loaded model using the project's system prompt, optionally over RAG context. **Agent** is the app guide: it explains the app, inspects the project, recommends settings, moves you between pages and pre-fills fields. The same guide is available from the Guide button on every page.

## Workflow
1. **Test mode:** load a model (the red warning offers an inline loader), ask questions; with RAG corpora ticked it answers from retrieved context ("N source chunks used"). For pure recall untick the corpora.
2. **Agent mode (guide):** uses the configured helper model only, never silently switching to another loaded model; if no helper is available it says so. Ask "how do I train?" or "what should I do next?" — tool calls render inline as they happen.
3. The guide never presses mutating buttons for you: it may create review-pending Q&A pairs when you ask, but approve, export and train stay yours.

## Key controls
- `[name="chat-mode"]` — Test / Agent radios.
- `#chat-input` — message box.
- `#chat-send` — send; `#chat-stop` stops generation.
- `#chat-status-pill` — loaded-model pill.
- `#chat-inline-load-btn` — load a model from the warning panel.
- `#chat-system-prompt` — system prompt (Test mode).

## What to check
The header pill shows the model that is really loaded. In Agent mode compare the answer with the tool card — the numbers must match.

## Common mistakes
Expecting Test-mode answers to be pure recall while RAG corpora are ticked; expecting the guide to start training.
