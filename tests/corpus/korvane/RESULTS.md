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

## 7. Second run: everything through the web UI (2026-10-08, project `korvane-ui-1`, results in `results/2026-10-08-ui/`)

Genor's ask: the whole flow as a user lives it, **review and scoring included**. What went through the browser (Playwright, `tests/e2e_track_a.py`):
create, upload, mining, **review of every pair in the Pairs page (keyboard)**, reviewer additions, export, training, GGUF export, **quiz and
dataset evaluations on the Testing page**, Chat load, RAG index/search, **RAG-grounded quiz on the Testing page**, Guide, delete.
What did not: reading the pairs (text dumps, `corpus_review.py`), the retrieval@5 number (an API loop), coverage checks (scripts, as an independent
gate). Not re-run: DPO branch, base-model baseline (the Testing page offers project exports only), merged-bf16 quiz (Testing page too slow, see below).

### Pair funnel (this is the answer to "do the numbers add up")

| Stage | Pairs |
|---|---|
| Mined by the helper (19 files; model 2,648, gap rounds 790, extractive fallback 235) | **3,673** |
| – verdict identical to the first review (same file, chunk, question, answer; pre-filled, replayed in the UI) | 2,801 |
| – novel, read again by me this time | 872 |
| Rejected (quote echo 217, duplicate 150, junk 107, trivial/vague 136, wrong value 23 ...) | 1,497 |
| Approved from mined pairs (104 of them edited by the reader) | 2,176 |
| Reviewer-written pairs added in the UI (140 planned + 62 for facts still uncovered) | 202 |
| **Approved total = exported rows** (0 exact duplicates collapsed) | **2,378** |
| Fact coverage of the approved pairs (manifest, 19 core files) | 1068/1130 after the review, **1130/1130 after the 62 additions** |

Earlier the "5,554 pairs" figure mixed three discarded mining runs (~1,100 pairs, never reviewed) with the 4,455 reviewed pairs of the first run; it was never one funnel.
Training used the 2,378 rows with a random ~10 % internal hold-out; the quiz (122 questions) is separate and hand-written.

### Review speed (the reason the page was rebuilt first)

Old page: every click re-fetched all pairs (9.5 MB for 5,000, ~255 ms server side, loop blocked) and showed 100 truncated rows with no source text.
New workspace (paged slim queue, stat-validated cache, optimistic verdicts, prefetched chunk, keyboard): **~30 ms per verdict in a real browser**
(`scripts/bench_review_page.py`, 4,455 golden pairs); the UI replay of 3,673 verdicts + 140 additions took **279 s**, the 62 later additions ~1 min.
Replaying the first run's golden review through the UI reproduces its counts exactly (2,491 approved / 1,964 rejected).

### Scores of record (Testing page, Qwen3.5-9B, SFT 6 epochs, loss 0.18, 804 steps, 45 min; q4_k_m)

| What | Result |
|---|---|
| Quiz, 102 paraphrased questions (pass = every expected value) | **21 pass = 20.6 %**, 20 partial, 61 fail |
| 20 unanswerable questions (pass = declines) | **0 / 20** |
| **Memorization control**: 200 exact training questions (heuristic key-word judge) | **134 pass = 67 %**, 16 partial |
| Held-out pairs of the same facts (100, same judge) | 25 % |
| Quiz RAG-grounded with the tuned model (top 5) | **80 / 102 = 78.4 %**; unanswerable 2 / 20 |
| Retrieval@5 over the 102 questions | 92 / 102 = 90.2 % |
| (first run, for comparison) untrained base / SFT / base + RAG | 2 % (3/20 abstain) / 17.6 % (0/20) / 75.5 % (20/20) |

Reading: the low LoRA score is **correct and now explained**. Training works (67 % of its own training questions come back, vs ~2 % untrained; the judge is
looser than the quiz's all-values rule, so the two numbers are not directly comparable), but the model does not carry the facts across re-wordings
(20-25 %), and it loses the ability to say "not in the documents" (0/20). RAG stays the way to serve facts.

### Findings of this run (fixed unless marked)

- Pairs page: no per-pair context, 100-row cap, full refetch per click, loop-blocking listing, background reload rebuilt the pane under the reviewer (fixed, 4 bugs found by the replay).
- Testing page could not take a user's quiz or score "must decline" questions (added: import, `expect_abstain`, per-category summary).
- Merged bf16 on the Testing page: > 40 min for 122 questions (aborted). Use the GGUF. (Open: HF `generate` for Qwen3.5 hybrid layers.)
- The Testing page cannot select the untrained base model, so the base baseline needs the Chat page or the script (open).
- Helper load at `n_ctx 32768` OOMs next to the resident RAG embedder and falls back to 16 GPU layers (handled, slower; open).
- Harness: mining wait 60 min was too short (now 3 h), helper load needed a longer HTTP timeout, handled GGUF OOM retry whitelisted.

## 8. Why RAG is not at 100 % (2026-10-08, project `korvane-ragtrace`, results in `results/2026-10-08-ragtrace/`)

Same 102 + 20 quiz, same 19 core files, same index (retrieval@5 reproduced: 92/102). New: per-question trace in the RAG suite
(`gold_in_retrieved`, `gold_in_context`, `chunks_in_context`), `max_context_chars` as a request parameter, and `scripts/rag_reader_compare.py` /
`scripts/rag_retrieval_ablation.py`. Readers: the **untrained** Qwen3.5-9B converted to q4_k_m (`convert_hf_to_gguf` + `llama-quantize`, deleted after) and the
helper Gemma 4 12B (q4_k_m), temperature 0, top-5.

| Reader | Context cap | Quiz pass (of 102) | Unanswerable declined (of 20) | Lost: not retrieved / cut by cap / reader |
|---|---|---|---|---|
| Qwen3.5-9B base | 5000 (old default) | 83 | 20 | 11 / 4 / 4 |
| Qwen3.5-9B base | 16000 | **87 (85 %)** | 20 | 11 / 0 / 4 |
| Gemma 4 12B | 5000 | 81 | 19 | 11 / 4 / 6 |
| Gemma 4 12B | 16000 | 84 | 19 | 11 / 0 / 7 |
| (run 2) SFT 6 epochs + RAG | 5000 | 80 | 2 | n/a |

- **The 5000-char cap fed the model ~2 of the 5 retrieved chunks** (Korvane chunks average 2,111 chars): mean chunks in context 1.97 vs 5.0. Lifting it: +4 / +3.
- **Retrieval is the main loss: 11 of 102 never reach the reader.** Of the 4 "reader missed" with Qwen, 3 are judge false negatives (`four` vs `4`, `bastion-gdy1` vs the
  full host name, one value of two); real reader misses are ~1.
- **Untrained base as reader beats the tuned model** (87 vs 80 answerable, 20/20 vs 2/20 declined). Tuning for facts hurts RAG.
- Retrieval ablation (recall@k, 102 questions, `results/.../ablation.json`): dense 84 @5; **BM25 alone 96 @5**; hybrid (RRF) 94; **hybrid + the app's
  cross-encoder rerank 90 (the default; 80 @1)**; rerank fused by RRF 94 (91 @3). The ms-marco English MiniLM reranker sharpens the first hit but pushes answers out of the top 5.
  The e5 `query:`/`passage:` prefixes the app does not use change nothing here (dense 84 -> 81, hybrid 94 -> 93). The corpus is IDs and numbers: lexical matching wins.
- Chunker (`rag/ingest.chunk_text`) joins words with spaces, so table rows lose their line structure. Not yet measured as a cause.

Context follow-up (same day): the loader now defaults to the model's native window (auto) and the RAG context cap is derived from it. Gemma 12B reader, auto context
(131072 tokens, all 5 chunks in the prompt): 84/102 answerable, 19/20 declined, identical to the 16000-char run, so the cap was fixed, not the retrieval.
Measured VRAM (3090, q4_k_m GGUF): Gemma 4 12B 18.3 GB at 32k (full-size window cache) -> 10.0 GB at native 131k (`swa_full=False`); Qwen3.5-9B 6.4 GB at 32k, 13.6 GB at native 262k.

### Is the remaining gap the models? (2026-10-09; `scripts/rag_rejudge.py`, auto context, same index)

| Reader | top-k | Page judge (of 102) | Normalised judge | Unanswerable declined |
|---|---|---|---|---|
| Gemma 4 12B | 5 | 84 | 88 | 19/20 |
| Gemma 4 12B | 10 | 87 | 91 | 19/20 |
| Gemma 4 12B | 20 | 89 | 93 | 19/20 |
| **Qwen3.5-9B base** | 20 | **92** | **96 (94 %)** | **20/20** |

The page's keyword judge is a plain substring test: `44.2` vs `44.20`, `185` vs `185.00`, `4` vs `four`, `bastion-gdy1` vs `bastion-gdy1.korvane.example` (4 right answers marked wrong; the same 4 for every reader). Of the 6 misses left for Qwen at top-20:
- e002, e101: gold not in the top 20 (retrieval; dense recall@50 is 99/102, so they exist).
- e076: judge (`Mon-Sat` vs `Monday to Saturday`).
- e097: chunking, not the model. The flattened CSV puts "Contact Rafał Dybek ... Visit 3 Oct 10:00" next to the *next* row's id (`OPP-24-0307 | Vestfold Fisk AS`); both readers correctly answered "don't know".
- e077: the model name sits in another table row of the flattened table; the reader gave half the answer.
- e044: a real reader miss: the answer (`ssh -p 2222 kcops@..., ask Wiktor`) is in the context, Qwen said "I don't know". It passed at top-5, so 15 extra chunks cost one answer.
Reading: ~1 of 102 is a model miss, 2 are retrieval, 2 are chunking/table structure, 1 is the judge. Top-20 retrieval is affordable now (about 12k tokens of 131k-262k) and keeps abstention intact (Qwen 20/20).

## 9. Re-judged by the AI judge (2026-10-09; results in `results/2026-10-09-rejudged/`)

Every score in §3-§8 came from the Testing page's substring / keyword matcher, which the app no longer uses: a test run now saves the raw answers and a separate
judge (an AI model on any provider row, or you) decides pass / partial / fail (`docs/judging/RUN-THEN-JUDGE.md`). All 14 archived reports were judged again from
their saved answers with `scripts/rejudge_reports.py` (judge: Gemma 4 12B, the helper seat, checklist prompt v4; answer key = the quiz's expected values; the 20
unanswerable questions are judged "declining is correct").

**Is the judge trustworthy?** `scripts/judge_eval.py` runs a judge against 50 hand-labelled cases (`eval/judge_gold.json`: 31 wrong/partial answers from run 2 that the
old matcher mis-scored plus clear passes, clear fails and unanswerable questions). Prompt v3 (a verdict only): 44/50 = 88 %, every error too *lenient* (it accepted
"week 46" for "2 December 2024" and a missing `kcops`). Prompt v4 (the judge lists every required fact with a quoted piece of the answer; the verdict follows the list and a
quote that is not in the answer is not trusted): 48/50 = 96 % at first run, 0 too lenient; both misses were label errors of mine on re-reading (relabelled in the file: 50/50). The gold set is small and was partly built from the cases that exposed v3's leniency, so read it as a sanity check, not a benchmark.

| Report (archive name) | Old matcher: quiz pass /102 | AI judge: pass / partial / fail | Unanswerable declined, old -> AI |
|---|---|---|---|
| untrained base, no RAG (`base-alone`) | 2 | 3 / 4 / 95 | 3 -> 6 /20 |
| base + RAG (`rag-base`) | 77 | 76 / 5 / 21 | 0 -> **20** /20 |
| tuned run 2 + RAG (`rag-trained-run2`) | 83 | 82 / 5 / 15 | 0 -> 0 /20 |
| SFT run 1, 3 ep early stop | 12 | 15 / 15 / 72 | 0 -> 0 |
| SFT run 2, 6 ep (`run2-6ep-noearly`) | 18 | 19 / 23 / 60 | 0 -> 0 |
| SFT run 2 merged bf16 | 26 | 28 / 23 / 51 | 0 -> 0 |
| run 3 = DPO on run 2 | 16 | 18 / 20 / 64 | 0 -> 0 |
| §7 scores of record: run 2 q4_k_m, Testing page (`ui-sft6-q4km`) | 21 (+20 partial) | 24 / 20 / 58 | 0 -> 0 |
| §7 tuned run 2 q4_k_m + RAG (`ui-sft6-q4km-rag`) | 80 (+8 partial) | 86 / 3 / 13 | 2 -> 2 |
| §8 Gemma 12B reader, top-5, 16k cap | 84 (+8) | 90 / 5 / 7 | 19 -> 19 |
| §8 Gemma 12B reader, top-10 | 87 (+6) | 92 / 4 / 6 | 19 -> 20 |
| §8 Gemma 12B reader, top-20 | 89 (+8) | 94 / 5 / 3 | 19 -> 20 |
| §8 Qwen3.5-9B base reader, top-5, 16k cap | 87 (+4) | 92 / 2 / 8 | 20 -> 20 |
| §8 **Qwen3.5-9B base reader, top-20** | **92 (+4)** | **97 / 2 / 3** | **20 -> 20** |

Reading: the conclusions stand, the numbers move by +2..+6 points. SFT teaches facts (quiz ~20-25 % pass, another ~20 % partial: right entity, one wrong value) and removes "not in
the documents" (0/20); base + RAG is 76-97 % depending on reader and top-k with 20/20 declines. The matcher's mistakes were mostly format (`44.2` vs `44.20`, `four` vs `4`, a
host's short name) and, for the SFT runs, partial credit it could not give. The old `ok` files of §3 have no partial class, which is why their pass counts barely change. The untrained base
"declines" 6/20 unanswerable questions without RAG because it says it has no public information, not because it read the documents.

