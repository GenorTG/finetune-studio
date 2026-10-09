---
id: testing
title: Testing page (step 4)
page: testing
keywords: test testing quiz suite pass rate from memory with context score scores difference recall retrieved grounded run judge judging ai judge human review verdict pass partial fail awaiting auto-judge transcripts answer key rag suite held-out validation memorization leakage results did it learn
---
## Purpose
Did the model actually learn your material? A test is two separate steps. **1 · Run** asks the model every question and saves exactly what it answered next to the correct answer (the raw transcript). **2 · Judge** is a separate pass: an AI judge (any model you connected: the helper seat, an API provider) or **you** read each saved case and decide pass / partial / fail. Nothing is scored by string matching, so a correct answer worded differently still passes.

## Workflow
1. Pick **What to test**: *Project quiz* (a quiz file or the quiz the wizard built), *RAG-grounded* (retrieve from the project's index, answer only from the retrieved text) or *Dataset check* (questions built from the training dataset).
2. Pick **Model under test** (default: the latest merged export; *Base model (untrained)* gives the baseline to compare the training against) and, for a quiz, the **Question suite**; press **Run test**. The run appears under *Saved test runs* and saves each answer as it arrives; you can leave the page.
3. Judge it: open the run and press **Judge unjudged** (pick the judge model next to it), or tick **judge automatically after each run** to make every finished run start the judge by itself (off by default: a run then ends at the raw answers).
4. Review: the list shows every case; the pane shows the question, the answer key, the model answer, the retrieved context (RAG), every judgement with its reasoning, and your own verdict buttons (keys 1 pass, 2 partial, 3 fail, 0 clear). Your verdict always wins over the AI's and is kept apart from it, so the page can show how often the judge agrees with you. When the dataset has retrieved-context rows the summary splits the pass count into **from memory (no context)** and **answering from retrieved context**: scores differ because they test different skills.
5. For the honest generalisation number use *Dataset check* with **Held-out slice** — the deterministic 10 % slice the trainer never saw. *Full training set* measures recall of training rows only.
6. **Export JSON / JSONL** gives the raw transcripts and all judgements for outside review.

## Key controls
- `[name="t-mode"]` — project quiz, RAG-grounded or dataset check.
- `#t-model` — model to test.
- `#t-suite` — question suite.
- `#t-run-btn` — run the test.
- `#t-eval-kind` — held-out versus full-training-set evaluation (dataset check).
- `#t-auto-judge` — judge automatically after each run (saved default).
- `#t-judge-provider` — the judge model (saved default).
- `#t-runs` — saved test runs.
- `#rv-list` — cases of the open run.

## What to check
A run with no verdicts has **no score** — "awaiting judgement" is not 0 %. Read a few judged cases yourself (the AI judge is a model, not truth); the agreement chip compares it with your verdicts. Plain rows measure recall of training facts; grounded rows are asked with their own context. Neither is generalisation — the held-out number is.

## Common mistakes
Reading a high from-memory score as generalisation; trusting verdicts marked *old matcher* (the retired scripted scorer: re-judge them); judging with the same small model that was trained; using Benchmarks to answer "did it learn my material?" (Benchmarks run public exams with their official scoring).
