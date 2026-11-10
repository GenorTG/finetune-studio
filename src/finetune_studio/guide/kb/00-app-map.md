---
id: app-map
title: App map and the two flows
page: dashboard
keywords: start new user overview workflow order steps flow files knowledge smart brain rag vs training where begin how app works pages tabs train a model on my documents first time how to train beginner end to end
---
## Purpose
Finetune Studio prepares, checks and tests what your models should know, then exports it as one encrypted **knowledge package**. Its core is the **Smart Brain**: answers from your own documents with sources, plus who it speaks for and how it answers, no training needed. Training a model on them is an optional extra, for how a model answers rather than what your documents say. Every file follows one route first: **Files & Knowledge**, where it is uploaded and parsed once; training and the Smart Brain both read that one parse.

## Workflow
The main path: 1 Files (`/projects/{pid}/data`: upload, read and edit the parsed text) → 2 Smart Brain (`/brain`: build the index, ask your documents with sources; no training, model weights never change) → 3 Test (`/testing`: score it on your own questions) → 4 Package (`/export`: what is left to do, then the package; trained-model export is below it).
Optional, the Train a model group: Pairs (`/data-prep`: Q&A pairs mined from the files, reviewed, exported as a dataset) → Train (`/training`: LoRA adapter or merged model), plus Compare and Benchmarks. Training is for how a model answers (tone, format, a narrow task); facts in documents belong in the Smart Brain.
One-page shortcut: Quick work (`/wizard`) builds the Smart Brain in one click, then optionally chains pairs → dataset → training → test.
System pages outside any project: Projects (`/projects`), Model library (`/models/explore`), My models (`/models`), Inference (`/inference`), Settings (`/settings`).

## Key controls
- `#workspace-subnav` — the project flow bar: home, 1 files, 2 Smart Brain, 3 test, 4 package, chat, settings, then the Train a model group (quick work, pairs, train, compare, benchmarks).

## What to check
Which route fits: facts that change often or must be cited → Smart Brain; behaviour, style or stable facts the model should know without looking them up → training. They can be combined (a trained model chatting with the Smart Brain's sources).

## Common mistakes
Starting at Training with no dataset (Start stays disabled until a dataset is picked or uploaded). Expecting training to make a model cite sources — that is what the Smart Brain is for.
