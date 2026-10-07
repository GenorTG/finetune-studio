---
id: dataset-quality
title: Preparing a high-quality dataset
page: pairs
keywords: dataset quality high quality prepare parse parsing clean paraphrase phrasing hallucination held-out holdout dedupe duplicate sources vague grounded coverage generation good data size how many pairs
---
## Purpose
Dataset quality decides model quality more than any training setting. This is the checklist the app supports end to end.

## Workflow
1. **Sources first.** On Files, open each parsed file and read its parsed text. Fix or drop files that parsed empty or garbled before generating anything.
2. **Generate per chunk.** On Pairs use 3+ pairs per chunk; for facts that must be memorised, aim for at least 3 differently phrased questions per fact (paraphrase diversity is what lets a model extract a fact later).
3. **Review like a curator.** Filter Pending; reject vague or ambiguous rows, edit sloppy answers. Spot-check ten pairs against the source text.
4. **Export** approved pairs only. If retrieved-context rows are wanted, build RAG first, then tick Include retrieved context.
5. **Check on Training.** Picking the dataset runs the dataset check: usable examples, held-out rows, duplicates, conflicting answers, very short answers. Use the Remove duplicates button it offers; open the Data Editor for single rows.
6. **Keep a held-out slice.** Training reserves a deterministic 10% (seed 42) the trainer never sees; that slice is the honest generalisation number on Testing.

## Key controls
- `#dp-review-card` — the review area on Pairs.
- `#dp-filter-pending` — pending filter.
- `#dp-approve-all-pending` — approve after review.
- `#prep-qpc` — pairs per chunk.

## What to check
Answers quote or closely paraphrase the source; no unanswerable questions; no duplicate or contradicting rows; enough rows for the planned steps (see training-settings). A grounded row (retrieved context in its system turn) teaches answering from context, not memorisation — never judge such rows asked bare.

## Common mistakes
Judging quality by row count; training on pending or unreviewed rows; reading the plain-row recall number as generalisation; generating rejected answers for DPO with a model and not reviewing them.
