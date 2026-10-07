---
id: testing
title: Testing page (step 4)
page: testing
keywords: test testing quiz suite pass rate from memory with context score scores difference recall retrieved grounded held-out validation memorization leakage rag suite judge results did it learn
---
## Purpose
Did the model actually learn your material? Testing quizzes a trained run on questions built from your own dataset (one per row by default) and shows pass/fail with the model's raw answer per case.

## Workflow
1. **Model** = auto (latest merged), or pick one; **Test suite** = the auto-generated suite at the end of training; click **Run** (about 40–90 s for ~60 cases; the model auto-loads first).
2. Read the Results card: total, judged, passed, failed, pass rate. When the dataset has retrieved-context rows it splits into **from memory (no context)** and **answering from retrieved context**.
3. For the honest generalisation number use **Run held-out eval** with Evaluation type *Held-out validation (quality)* — the deterministic 10% slice the trainer never saw. *Full training set (memorization check)* measures recall of training rows.
4. **Run with RAG** quizzes retrieval-then-answer.
5. Spot-check rows by hand; the heuristic judge is not truth.

## Key controls
- `#t-model` — model to test.
- `#t-suite` — question suite.
- `#t-run-btn` — run the suite.
- `#t-run-rag-btn` — run the RAG-grounded suite.
- `#t-eval-kind` — held-out versus full-training-set evaluation.
- `#t-train-eval-btn` — run the dataset evaluation.
- `#t-results` — results card.

## What to check
Plain rows measure recall of training facts; grounded rows are asked with their own context. Neither is generalisation — the held-out number is. Never score grounded rows bare.

## Common mistakes
Reading a high from-memory score as generalisation; judging only on training loss; skipping held-out eval; using Benchmarks to answer "did it learn my material?" (Benchmarks measure public exams).
