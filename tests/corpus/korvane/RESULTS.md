# Korvane corpus — first full Track A / Track B run (2026-10-08)

Fresh DB, real browser (`tests/e2e_track_a.py`), Qwen3.5-9B, RTX 3090, one GPU job at a time. Corpus: 19 core files, **1,130 manifest facts**.
Use this file as the baseline when re-running; every number below is reproducible with the commands in `tests/E2E_MANUAL_GUIDE.md` (Track A / B).

## 1. Mining and the manual review (zero trust)

| Step | Result |
|---|---|
| Parse | every manifest fact survived parsing (`scripts/corpus_parse_check.py`): scanned PDF via OCR, `.doc` via LibreOffice, `.eml` via bytes |
| Mine (exhaustive miner, local Gemma 4 12B, temperature 0) | 5,554 pairs written over the whole session (3 discarded runs + 8 files re-mined after the header bug) |
| Reviewed one by one by me (`scripts/corpus_review.py`, ledger `.tmp/review-ledger.jsonl`) | **2,491 approved** (each with an `A` verdict; 110 reviewer-written because the miner never stated the fact, 210 reworded because the question was mislabelled / unit missing / wrong premise), 1,964 rejected with a reason (+ ~1,100 pairs of runs discarded unreviewed after the header bug) (quote echoes, duplicates, wrong values, junk) |
| Approved-only fact coverage | **1130 / 1130 = 100 %**. Levels: 792 facts in a single answer, 242 across several answers of their chunk, 96 token-level (re-worded values) — the last two groups were read and, where the literal wording was missing, a literal reviewer pair was added |
| Plain export (grounded off) | 2,456 rows (35 exact duplicates collapsed); recomputed fact coverage of the exported rows = the approved set (0 facts lost) |

What the review found that no automatic gate would have: wrong column mapping in tables (board pack, CRM chunks 3-9), numbers attached to the wrong row, a
**superseded value** (the Oakhaven email says EUR 14,500, Jarek's correction says EUR 12,750, and "Olek rang at 05:20" is corrected to Jarek), pairs that
silently dropped a unit, and the export gate auto-approving unreviewed extractive pairs.

## 2. Training (SFT, 4-bit QLoRA because bf16 does not fit on 24 GB)

| Run | Settings | Steps / time | Train loss | Eval loss (random 10 % hold-out) |
|---|---|---|---|---|
| 1 | rank 64, LR 1e-4, 3 epochs, early stopping | stopped at 350/417 (epoch 2.5), 20 min | 0.53 | min 1.01 at step 275, then up |
| 2 | same, **6 epochs, no early stopping** | 834 steps, 39 min | 0.18 | 1.01 → 1.42 (rises every epoch) |
| DPO | from run 2, 188 reviewed pairs, 3 epochs | 255 steps, 9 min | 0.10 | n/a |

GGUF export of the 9B (f16 → q4_k_m, then q6_k): 7-8 min, **the app answered every probe in ≤ 40 ms** while it ran.

## 3. Quiz (`scripts/corpus_eval.py`: 102 paraphrased questions, 20 unanswerable; no retrieval)

| Model | Paraphrase recall | Abstains on unknown |
|---|---|---|
| base Qwen3.5-9B, untrained | 2 % | 3 / 20 |
| SFT run 1 (q4_k_m) | 11.8 % | 0 / 20 |
| SFT run 2 (q4_k_m) | 17.6 % | 0 / 20 |
| SFT run 2, merged bf16 (no GGUF quantization) | 25.5 % | 0 / 20 |
| DPO on run 2 (q4_k_m) | 15.7 % | 0 / 20 |
| **Track B: base + RAG (top 5)** | **75.5 %** (retrieval@5 90.2 %) | **20 / 20** |
| Track B: SFT run 2 + RAG | 81.4 % | 0 / 20 |

Reading: fine-tuning on ~2.4 reviewed pairs per fact teaches style and a few easy facts, not paraphrase-robust recall (12-26 %), and it **removes** the base
model's ability to say "not in the context" (it invents a confident value for every unanswerable question). RAG over the same files gets 75-81 % and is bounded by
retrieval (90 %); the base model with RAG abstains correctly on all 20. The eval loss is a poor gate for knowledge injection: its minimum is at epoch ~2 while recall
keeps rising with epochs (and the "best checkpoint" default would keep the epoch-2 weights).

## 4. DPO branch

The in-app pair builder wrote 300 pairs (150 "stick to the source", 150 "admit it doesn't know"). Reviewing every abstain question against the corpus: **112 of 150
were answerable from other files** (the helper only sees one chunk), so only 38 were kept; training on the unreviewed set would have taught refusals of known facts.
After DPO (loss 0.10) recall did not improve (15.7 %) and abstention stayed 0/20.

## 5. Bugs found and fixed in this run

- Miner: the first DATA row of a chunk was carried to the next chunk as the table header (CRM chunks 3-9, 8 files re-mined). Fixed in `exhaustive.table_header_before`.
- GGUF: a trained+merged Qwen3.5 has no `mtp.*` weights but the config declares them, so the GGUF promised a block it lacked and llama.cpp refused to load it. Fixed
  with the converter's `--no-mtp` (`gguf_convert.converter_extra_args`).
- Export gate auto-approves extractive `coverage_fill` pairs for any chunk without an approved pair, at every export (open, see HANDOFF).
- Harness: the GGUF wait was 240 s (a 9B needs ~8 min); the journal watcher flagged the app's own handled merge-OOM fallback as an error.
- One CUDA illegal-memory-access abort of the whole service during preference building with the helper at `n_ctx 16384` (not reproduced at 32768).

## 6. Not done / open

- A paraphrase-augmentation experiment (3-10 reviewed-answer / re-worded-question variants per fact) to see how far SFT recall can go; estimated 1-2 h per run.
- Evaluate with the adapter on the 4-bit base (no merge) to split "QLoRA-merge loss" from "under-training".
- Extended-tier files, API-helper benchmark rerun, delegate/background audit of remaining long operations.
