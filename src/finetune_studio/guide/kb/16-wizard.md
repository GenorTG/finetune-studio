---
id: wizard
title: Quick work (wizard)
page: wizard
keywords: quick work wizard quick start run everything one page chain automatic beginner new user easiest simplest
---
## Purpose
The whole product from one page. Upload files, then either chain a trained model (files → question-answer pairs → dataset → training → test → pin a version) or build a RAG index with one click. **Step by step** shows every control separately in any order.

## Workflow
1. Step 1 upload files; Step 1b optionally make them searchable now (RAG).
2. Steps 2–5: pairs, dataset, train (base model plus a size-based preset such as Standard; optional epochs override), test.
3. Or use **Run everything at once**.
4. The chain loads the helper in step 2 and aborts if nothing was mined; it unloads the helper before training.

## Key controls
- `#wiz-quick` — the quick-start section.
- `#wiz-step-train` — the training step card.
- `#wiz-base-model` — base model for the quick run.
- `#wiz-preset` — preset by model size (Nano, Standard, Large, QLoRA).
- `#wiz-epochs` — epochs override (empty = the preset's own).
- `#wiz-run-all` — run every step in order.

## What to check
Each step card has a status pill; a failed step stops the chain with the reason. The Training page's effort presets and the wizard's size presets fill the same fields.

## Common mistakes
Leaving the helper model missing (step 2 fails); using the wizard when you want to review pairs carefully — use Pairs.
