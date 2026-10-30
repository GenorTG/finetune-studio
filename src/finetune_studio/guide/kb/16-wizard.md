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
3. Or use **Run everything at once**. If the selected preset would produce fewer than 100 optimizer steps, Quick Start preserves it and labels the run a workflow smoke test; it never silently increases epochs on a tiny dataset.
4. Step 5's generated quiz asks about the training rows themselves. Its score measures training-set recall/memorization, not held-out generalization; use the Testing page's held-out or reworded quiz for generalization. A missing merged model is retried through the supervised Export job before the test starts.
5. Step 6 pins a version: a saved record of the project's datasets, recent training runs and latest Smart Brain build. **Builds on** links it to an earlier version (the latest by default), so **Lineage** shows the chain (v1 → v2 → v3) and what each version held. **Delete** removes only the record; datasets, runs and models stay.
6. In step 2, Quick Start loads a local assistant model if selected; an API assistant is used on demand. It stops if no document was mined, and unloads a resident model before training.
7. The same snapshot history is available in a terminal: `fts versions pin/list/show/lineage/delete`. Pinning builds on the latest version by default; use `--root` for a separate lineage. Deleting removes the snapshot record, not its datasets or model files.

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
Leaving the assistant's model missing (step 2 fails); using the wizard when you want to review pairs carefully — use Pairs. For scripting snapshot history, use `fts versions`.
