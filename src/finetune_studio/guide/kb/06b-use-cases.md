---
id: use-cases
title: Use cases (Training page)
page: training
keywords: use case usecase template extraction json fields invoice classification routing labels tickets intent style tone house style distillation distil teacher big model held-out test base versus trained compare mcnemar a/b judge examples file jsonl draft drafting raw texts unlabelled label assistant review approve
---
## Purpose
Train the jobs a LoRA does well, and measure them fairly. Four templates: **extraction** (text → one JSON object with fixed fields), **classification** (text → one of your labels), **house style** (answers in your tone and format), **distillation** (a small local model answers one kind of request like a big API model). Each one checks your examples, keeps a held-out test the model never trains on, proposes training settings sized to your data, and after training compares the run with its untrained base on that test. Facts from your documents are a Smart Brain job, not a use case.

## Workflow
1. On Training, open **New use case…** in the **Start from a use case** card and pick a template; the card shows the row shape, an example row and how many rows it needs (at least 50 to train plus a held-out test of 20 %, at least 20; classification at least 10 per label; distillation at least 100).
2. Pick the examples file (JSONL, one example per line, or a JSON list), add an optional instruction (and labels, a JSON schema, a style guide or a teacher provider for the template), then **Check examples and build**. Unusable rows are listed by row number; nothing is built until they are fixed.
3. The form below is filled from the use case: its dataset, the SFT route, system prompt mode none (the instruction is in every row) and the proposed epochs, learning rate, rank, alpha, batch, accumulation and sequence length. Pick the base model and **Start training**.
3b. **No finished examples?** Open **No finished examples? Draft them with the AI assistant…** (extraction, classification and style): upload raw texts (a .txt with blocks separated by a blank line, or one per line; a JSON list; JSONL), say what they are, and for extraction list the fields (or give a schema above); classification uses the labels above, style the style guide. **Draft with the assistant** asks the assistant's *Drafting use-case examples* job (Settings → AI assistant) for the label, the extracted JSON or the request each text answers (for style, your text stays the answer). Review every row: fix a draft in place, **Approve** or **Reject** it, or **Approve all drafted**; rows the assistant could not draft say why. **Build use case from N approved** runs the normal build on the approved rows only: nothing trains on an unreviewed draft. Up to 500 texts per draft set.
4. When the run is done, press **Evaluate / results** on the use case, pick the run and **Start comparison**: the untrained base and the run answer the held-out questions. Extraction and classification are scored by code; distillation by the AI judge against the teacher's answers (judge on Compare); style by **Run the blind A/B judge**.

## Key controls
- `#uc-card` — the Start from a use case card: your use cases, with Use in form, Evaluate / results and Delete.
- `#uc-new` — the New use case form.
- `#uc-template` — the template (extraction, classification, style, distillation).
- `#uc-file` — the examples file.
- `#uc-build-btn` — check every example and build the dataset and the held-out test.
- `#uc-report` — evaluate a finished run against its base and read the result.
- `#uc-draft` — draft examples from raw texts with the AI assistant.
- `#uc-draft-file` — the raw texts to draft from.
- `#uc-draft-btn` — start drafting.
- `#uc-draft-review` — review the drafts: fix, approve, reject, build from the approved rows.

## What to check
Read the paired line, not the two percentages alone: it counts questions the run gained and lost against the base and gives a McNemar p value; "could be noise" means the difference may be chance (more held-out rows, or a bigger effect, are needed). Style needs both orders of the A/B judge to agree before a win counts. A warning after the build (fewer rows than the recommended range) means the result will be weak.

## Common mistakes
Uploading document facts as a use case (use the Smart Brain); picking an effort preset after the form was filled (it overwrites the sized settings: press Use in form again); evaluating a run trained on a different dataset; reading a distillation score before the judge has finished; approving drafts without reading them (the assistant can be wrong: the held-out test then measures its mistakes, not yours).
