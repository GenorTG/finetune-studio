---
id: guide-actions
title: The Guide panel and its actions (ask or hands-free)
page: overview
keywords: guide panel corner assistant chat agent action actions run skip confirm ask hands free hands-free automatic auto do it for me start mining build dataset smart brain train training test quiz pin version mode
---
## Purpose
The Guide sits in the corner of every page (the guide button in the header). It answers questions about the app, checks your project, recommends settings, opens pages, highlights controls and pre-fills forms. It can also do the work: **start mining** files, **build the dataset**, **build the Smart Brain**, **start training**, **build a reworded quiz**, **run a test** and **pin a version**. It runs on the AI assistant's *Guide chat* job (Settings → AI assistant).

## Workflow
1. Open the Guide and say what you want, e.g. "mine all my files", "start training on the newest dataset with Baseline", "test the latest model on the reworded quiz". Baseline uses the Training page's adaptive `balanced` recommendation.
2. **Ask before each action** (default): an explicit request prepares an action card without asking you to repeat yourself. The Guide checks your project and shows what it will do. **Run** is the confirmation and sends it through the same endpoint as the page's own button; **Skip** leaves it. The Guide then reports the result and suggests the next step.
3. **Hands-free**: pick it in the selector at the top of the Guide. The Guide runs the actions it prepares at once and carries on with the next step, up to 4 actions in a row per message; after that a card waits for Run.
4. Long jobs (mining, training, Smart Brain, tests) start and run in the background; their pages show progress.

## Key controls
- `#guide-panel` — the Guide panel.
- `#guide-mode` — Ask before each action, or Hands-free.
- `#guide-input` — ask the Guide.

## What to check
Each action card ends with "✓ done" and what started (a run id, a build id, the number of jobs) or "✗ failed" with the reason from the app. The Guide never approves or rejects pairs, exports a model or deletes anything: those stay on their pages.

## Common mistakes
Leaving Hands-free on while trying things out: every action it prepares runs (training uses the GPU until it ends). Expecting an action to wait for an earlier long job: mining and training start in the background, so ask the Guide to continue once they finish.
