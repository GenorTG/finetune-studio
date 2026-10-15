---
id: app-map
title: App map and the two flows
page: dashboard
keywords: start new user overview workflow order steps flow files knowledge smart brain rag vs training where begin how app works pages tabs train a model on my documents first time how to train beginner end to end
---
## Purpose
Finetune Studio turns your own documents into a fine-tuned model, a **Smart Brain** that answers from them without training, or both. Every file follows one route first: **Files & Knowledge**, where it is uploaded and parsed once; training and the Smart Brain both read that one parse.

## Workflow
1 Files & Knowledge: files (`/projects/{pid}/data`: upload, read and edit the parsed text) and pairs (`/data-prep`: Q&A pairs mined from the files, reviewed, exported as a dataset).
2 Train (`/training`: LoRA adapter or merged model) **or** 2 Smart Brain (`/brain`: build the index, search, chat with sources; no training, model weights never change).
3 Test (`/testing`: the model or the Smart Brain on your own questions) → 4 Use it (`/export`) or Chat.
One-page shortcut: Quick work (`/wizard`) chains files → pairs → dataset → training → test, or builds the Smart Brain in one click.
System pages outside any project: Projects (`/projects`), Model library (`/models/explore`), My models (`/models`), Inference (`/inference`), Settings (`/settings`).

## Key controls
- `#workspace-subnav` — the project flow bar: home, quick work, 1 Files & Knowledge (files · pairs), 2 train or 2 Smart Brain, 3 test, 4 use it, and the extra pages.

## What to check
Which route fits: facts that change often or must be cited → Smart Brain; behaviour, style or stable facts the model should know without looking them up → training. They can be combined (a trained model chatting with the Smart Brain's sources).

## Common mistakes
Starting at Training with no dataset (Start stays disabled until a dataset is picked or uploaded). Expecting training to make a model cite sources — that is what the Smart Brain is for.
