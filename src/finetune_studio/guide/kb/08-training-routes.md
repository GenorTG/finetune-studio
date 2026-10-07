---
id: training-routes
title: Choosing a training route (SFT, DPO, tool SFT, continued pretraining, distillation)
page: training
keywords: route sft dpo preference chosen rejected tool calling tool_sft continued pretraining raw text reasoning distillation orpo kto which route choose upload jsonl format
---
## Purpose
Pick the route from the data shape and the product need. Routes are chosen with the radio buttons in the Training route box.

## Workflow
- **SFT** — curated prompt/answer examples (the Q&A pairs export, or conversational JSONL). Teaches format, style, selected facts.
- **DPO** (Preference tuning) — each prompt has a reviewed preferred (`chosen`) and rejected answer. JSONL rows: `prompt`, `chosen`, `rejected` (strings or role/content message lists). Needs at least 2 rows; 10% is held out. Choosing DPO sets learning rate 1e-6, 1 epoch, no warmup (visible, editable). The app does not author comparisons for you.
- **Tool-calling SFT** — JSONL rows with `messages` (assistant `tool_calls`, `tool` replies, final assistant turn) plus a `tools` array of JSON function schemas; the base model's chat template must accept them. It trains tool-call syntax, not the tool runtime.
- **Continued pretraining** — raw text JSONL, one `{"text": …}` per row, kept unwrapped. Not instruction tuning; no automatic QA quiz; evaluate on held-out domain text.
- **Reasoning distillation** — reviewed teacher demonstrations as `messages` JSONL with a verifiable final answer; same supervised trainer, so review traces and evaluate on independent tasks.
- **RAG** is not a training route: use it for knowledge that changes often or must be cited. Combine freely (SFT + RAG).

## Key controls
- `[name="training_mode"]` — the five route radios.
- `#training-mode-help` — one-line explanation of the selected route.
- `#dpo-format-help` — DPO row format (shown when DPO is selected).
- `#dataset-upload` — upload your own JSONL for any route.
- `#dataset-health` — route-aware dataset check (uses the selected route's formatter).

## What to check
Invalid data returns a visible 400 before any model loads. A DPO run records `training_mode=dpo`; its quiz expects chosen, never rejected. Do not infer quality from loss alone.

## Common mistakes
Using DPO with only chosen answers (that is SFT); uploading preference rows to the SFT route (the check will mark them untrainable); looking for ORPO or KTO — not available, the installed TRL has no such trainer, and DPO is never relabelled as them. Preference-comparison authoring, automatic teacher traces and benchmark decontamination are not in the app yet.
