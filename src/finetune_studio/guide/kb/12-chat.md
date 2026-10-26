---
id: chat
title: Chat page (Test and Agent / Guide modes)
page: chat
keywords: chat test mode agent mode guide helper model load smart brain rag context sources system prompt temperature talk to model ask questions app help
---
## Purpose
Two modes. **Test** chats with a loaded model using the project's system prompt, optionally with the Smart Brain's sources. **Agent** is the app guide: it explains the app, inspects the project, recommends settings, moves you between pages and pre-fills fields. The same guide is available from the Guide button on every page.

## Workflow
1. **Test mode:** load a model (the red warning offers an inline loader), ask questions; with the Smart Brain ticked it answers from the passages it finds ("N source chunks used"). For pure recall untick the corpora.
2. **Agent mode (guide):** uses the AI assistant's Guide chat job only, never silently switching to another loaded model; if no helper is available it says so. Ask "how do I train?" or "what should I do next?" — tool calls render inline as they happen.
3. The guide can prepare actions (mining, dataset, Smart Brain, training, quiz, test, version) that run after you press Run, or at once in Hands-free mode (see guide-actions); it may also create review-pending Q&A pairs when you ask. Approving pairs, exporting a model and deleting stay yours.

## Key controls
- `[name="chat-mode"]` — Test / Agent radios.
- `#chat-input` — message box.
- `#chat-send` — send; `#chat-stop` stops generation.
- `#chat-model-warn` — "No model loaded" panel with the inline loader (the model status itself is the sprite row at the top of the first card).
- `#chat-inline-load-btn` — load a model from the warning panel.
- `#chat-system-prompt` — system prompt (Test mode).

## What to check
The header pill shows the model that is really loaded. In Agent mode compare the answer with the tool card — the numbers must match.

## Common mistakes
Expecting Test-mode answers to be pure recall while the Smart Brain is ticked; expecting the guide to start training.
