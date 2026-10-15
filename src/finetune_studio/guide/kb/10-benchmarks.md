---
id: benchmarks
title: Benchmarks page
page: benchmarks
keywords: benchmark mmlu gsm8k hellaswag synthetic offline smoke official compare base model judge score regression
---
## Purpose
Score a run on public exams instead of your own data — to compare with published numbers or check the fine-tune did not damage general ability. Optional step.

## Workflow
1. In the trained-run row choose a suite (for a quick check *synthetic · knowledge MCQ (offline)*) and **Run** (~20 s).
2. The base-model row runs the same suites on the untrained base; official suites (GSM8K, MMLU, HellaSwag; sampled 50) download datasets and are slow — run them deliberately.
3. **Recent scores** lists results; **Compare two runs** shows tuned versus base per suite.

## Key controls
- `#bench-run-latest` — run a new benchmark on the latest run.
- `#bench-base-row` — base-model benchmark row.
- `#cmp-run-a` / `#cmp-run-b` — pick two runs to compare.
- `#cmp-run-btn` — run the comparison.
- `#bench-scores-table` — recent scores.

## What to check
Synthetic and "practice" suites are smoke checks, not industry scores; official suites compare with published results; "from your data" suites test your material.

## Common mistakes
Comparing synthetic scores with published ones; using benchmarks instead of Testing for recall questions.
