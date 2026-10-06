---
id: pairs
title: Question-answer pairs page (step 2)
page: pairs
keywords: pairs qa question answer generate mine helper review approve reject pending export dataset grounded retrieved context difficulty style coverage chunk prep job sources parsed
---
## Purpose
A model learns from questions and answers, not raw documents. This page reads each parsed source chunk by chunk, writes Q&A pairs with the **helper model**, lets you review them, and exports the approved ones as the training dataset.

## Workflow
1. In **Parsed sources** tick the files, then **Generate pairs for selected** (the helper model loads automatically, about 50 s the first time; mining a small file takes a minute or two).
2. Press **Refresh** — the table is not live. Rows move from parsed to training-ready or needs-review.
3. In **Training / Q&A output** filter **Pending**. Reject vague rows, edit sloppy answers, then **Approve all pending**. Counters (total/pending/approved/rejected) must add up.
4. Optionally tick **Include retrieved context** (default 40% of rows) if you will chat with the trained model over a RAG corpus; build the RAG index first.
5. **Export approved → Training**, then **Start training with this dataset**.

## Key controls
- `#dp-parsed-sources` — the Parsed sources card (check boxes per file).
- `#dp-generate-selected` — mine pairs for the ticked files.
- `#prep-qpc` — pairs per chunk (1–10, default 3).
- `#prep-diff` — difficulty: easy, medium, hard, expert.
- `#prep-style` — style: socratic, direct, factual, eli5, code.
- `#prep-helper-ctx` — helper context window in tokens (default 32768).
- `#prep-start-btn` — start a prep job from the Run a prep job form.
- `#dp-filter-pending` — show only pairs awaiting review.
- `#dp-approve-all-pending` — approve every pending pair.
- `#dp-grounded` — include retrieved context in the export.
- `#dp-grounded-pct` — share of rows that carry retrieved context.
- `#dp-export-approved` — export approved pairs as a registered dataset.
- `#dp-open-training` — jump to Training with the exported dataset.

## What to check
Every source has at least one pair; helper pairs arrive **pending**, coverage-fill pairs arrive approved. Read ten pairs by hand: questions self-contained and specific, answers faithful to the source (a number or name not in the file is a hallucination), no pair opening with a pronoun. The ingestion log's rejection counters (`unanswerable_from_chunk`, `ungrounded_answer`) show what the grounding filter dropped. Vague files yield vague or no pairs — that is the validator working.

## Common mistakes
Approving everything unread; exporting while pairs are still pending (they are excluded); expecting the export to proceed when a file produced no usable pair — the export is blocked (409) naming the file: delete the file or export without it.
