---
id: compare
title: Compare page
page: compare
keywords: compare models side by side same questions base tuned which model better answers judge judging ai judge human verdict pass partial fail awaiting quiz suite
---
## Purpose
Which of two or more models answers your questions better? **1 · Run** asks every chosen model every question of one quiz and saves the raw answers next to the correct answer. **2 · Judge** is a separate pass: an AI judge (any model you connected) or **you** read each saved answer and decide pass / partial / fail. Nothing is scored by string matching, so a score appears only after answers are judged.

## Workflow
1. Tick 2 or more **models** (the untrained base model is a good baseline) and pick a **question suite**, then press **Run comparison**. The models answer one after another because there is one GPU; the table fills in as they do.
2. Open the comparison under *Saved comparisons* (it opens by itself after the run starts). Pick a judge and press **Judge all answers**, or tick *judge automatically when the run finishes* before running.
3. Read the answers side by side: each question shows the answer key, then every model's answer with its verdict and the judge's reasoning. Press pass / partial / fail on an answer to give your own verdict; yours always wins over the AI's. *models differ* shows only the questions where the judged models got different verdicts.

## Key controls
- `#cp-models` — the models to compare.
- `#cp-suite` — question suite.
- `#cp-run-btn` — run the comparison.
- `#cp-judge-btn` — judge all saved answers.
- `#cp-cases` — answers side by side.

## What to check
"Awaiting judgement" is not 0 %: a model shows a pass rate only once its answers are judged. The AI judge is a model, not truth — read a few answers it passed and failed. Each model's answers are also an ordinary run on the Testing page.

## Common mistakes
Reading a pass rate while half the answers are still awaiting judgement; comparing models on a quiz built from the training data (it measures recall of training rows, not generalisation).
