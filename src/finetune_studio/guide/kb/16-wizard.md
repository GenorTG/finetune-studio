---
id: wizard
title: Quick work (wizard)
page: wizard
keywords: quick work wizard quick start run everything one page chain automatic beginner new user easiest simplest version pin lineage snapshot builds on
---
## Purpose
The whole product from one page. Upload files, then either chain a trained model (files → question-answer pairs → dataset → training → test → pin a version) or build the Smart Brain with one click. **Step by step** shows every control separately in any order.

## Workflow
1. Step 1 upload files; Step 1b optionally build the Smart Brain now (no training).
2. Steps 2–5: pairs, dataset, train (base model plus a size-based preset such as Standard; optional epochs override), test.
3. Or use **Run everything at once**.
4. Step 6 pins a version: a saved record of the project's datasets, recent training runs and latest Smart Brain build. **Builds on** links it to an earlier version (the latest by default), so **Lineage** shows the chain (v1 → v2 → v3) and what each version held. **Delete** removes only the record; datasets, runs and models stay.
5. The chain loads the assistant's local model in step 2 and aborts if nothing was mined; it unloads it before training.

## Key controls
- `#wiz-quick` — the quick-start section.
- `#wiz-step-train` — the training step card.
- `#wiz-base-model` — base model for the quick run.
- `#wiz-preset` — preset by model size (Nano, Standard, Large, QLoRA).
- `#wiz-epochs` — epochs override (empty = the preset's own).
- `#wiz-run-all` — run every step in order.
- `#wiz-version-parent` — the earlier version a new one builds on.
- `#wiz-versions-table` — saved versions with Lineage and Delete.

## What to check
Each step card has a status pill; a failed step stops the chain with the reason. The Training page's effort presets and the wizard's size presets fill the same fields.

## Common mistakes
Leaving the assistant's model missing (step 2 fails); using the wizard when you want to review pairs carefully — use Pairs.
