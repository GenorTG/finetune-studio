---
id: app-map
title: App map and the two flows
page: dashboard
keywords: start new user overview workflow order steps flow rag vs training where begin how app works pages tabs train a model on my documents first time how to train beginner end to end
---
## Purpose
Finetune Studio turns your own documents into either a fine-tuned model (the **model flow**) or a searchable index (the **RAG flow**). Both flows start at the same place: files.

## Workflow
Model flow, in order: 1 Files (`/projects/{pid}/data`) → 2 Pairs (`/data-prep`: Q&A pairs from the files, reviewed, exported as a dataset) → 3 Training (`/training`) → 4 Testing (`/testing`) → 5 Export (`/export`) or Chat.
RAG flow: Files → RAG index (`/rag`: build, search, grounded chat). No training; model weights never change.
One-page shortcut: Quick work (`/wizard`) chains files → pairs → dataset → training → test.
System pages outside any project: Projects (`/projects`), Model library (`/models/explore`), My models (`/models`), Inference (`/inference`), Settings (`/settings`).

## Key controls
- `#sb-tabs` — the top tab strip; the group named after the active project lists that project's pages.

## What to check
Which flow fits: facts that change often or must be cited → RAG; behaviour, style or stable facts the model should know without looking them up → training. They can be combined (a trained model chatting over a RAG corpus).

## Common mistakes
Starting at Training with no dataset (Start stays disabled until a dataset is picked or uploaded). Expecting training to make a model cite sources — that is what RAG is for.
