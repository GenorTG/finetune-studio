---
id: pairs
title: Question-answer pairs page (step 1)
page: pairs
keywords: pairs qa question answer mine mining helper checklist inventory data points gaps fill review approve reject pending exclude complete export dataset grounded retrieved context coverage chunk rounds files
---
## Purpose
A model learns from questions and answers, not raw documents. This page mines each parsed file into Q&A pairs with the **AI assistant**, checks them against a checklist of the file's data points, lets you review them, and exports the approved ones as the training dataset.

## Workflow
1. In **Mine pairs, file by file** tick the files and press **Mine selected** (a local assistant model loads first, about 50 s; an API connection needs nothing). Each file is one job: **inventory** (its data points: every statement with a number, date, code or name, plus the facts the helper lists, including corrections that span the file), **mining** (pairs written against that checklist), **checking**, **fixing** (open data points sent back, up to three rounds). The table updates live while jobs run.
2. In **Training / Q&A output** review the **Pending** pairs: approve, edit, or reject with a reason. The line above each pair says which data points it states and warns when it is the only pair stating one. **Remove near-duplicates** first rejects pending pairs that ask a fact another pending pair already asks (it shows the count and asks before rejecting).
3. Open a file's **Checklist** to see every data point: approved, awaiting review, open, or excluded. For an open point press **Fill gaps** (the assistant tries again and sees your reject reasons), **Write pair**, or **Exclude** it with a reason when it is not a fact worth training on. **Rounds** shows every helper call with its prompt and raw reply.
4. A file is **complete** when every data point is stated by an approved pair or excluded. Export needs every file complete (or an explicit "export anyway").
5. Optionally tick **Include retrieved context** (default 40% of rows) if the trained model will answer with the Smart Brain's sources; build the Smart Brain first.
6. **Export approved → Training**, then **Start training with this dataset**.
7. Optionally open **Data audit** and **Run the audit**: it re-reads every file from its stored bytes and checks the parse, the passages and that each approved pair points at a real passage (no model involved).

## Key controls
- `#dp-mining` — the Mine pairs card.
- `#dp-mine-selected` — mine the ticked files (inventory, mine, check, fix).
- `#dp-fill-selected` — ask the assistant again for the open data points of the ticked files.
- `#dp-files` — the files table: status, data points (approved/excluded of total, open), pairs, actions.
- `#dp-file-panel` — a file's checklist and rounds.
- `#dp-filter-pending` — show only pairs awaiting review.
- `#dp-dedupe` — reject pending near-duplicates (count first, then confirm).
- `#dp-audit-run` — run the data audit (files, passages, approved pairs).
- `#dp-grounded` — include retrieved context in the export.
- `#dp-grounded-pct` — share of rows that carry retrieved context.
- `#dp-export-approved` — export approved pairs as a registered dataset.
- `#dp-open-training` — jump to Training with the exported dataset.

## What to check
Every file reaches **complete**. Read the pairs, do not bulk-approve: questions self-contained and specific, answers faithful to the source (a number or name not in the file is a hallucination), one fact per pair (atomic pairs train far better than long multi-fact answers). Excluded points carry a reason. When a file keeps the same open points after Fill gaps, write those pairs yourself.

## Common mistakes
Approving everything unread; exporting while pairs are still pending (they are excluded); rejecting the only pair that states a fact and not filling the gap it opens; resetting a file (that deletes its unreviewed pairs and checklist; reviewed pairs stay).
