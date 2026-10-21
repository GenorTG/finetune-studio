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
- Export gate auto-approved extractive `coverage_fill` pairs for any chunk without an approved pair, at every export (fixed 2026-10-09: now opt-in; this run predates the fix).
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
| base + RAG (`brain-base`) | 77 | 76 / 5 / 21 | 0 -> **20** /20 |
| tuned run 2 + RAG (`brain-trained-run2`) | 83 | 82 / 5 / 15 | 0 -> 0 /20 |
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


## 10. Third full run after the supervisor split (2026-10-12, project `korvane-core-2`, results in `results/2026-10-12-core2/`)

Same corpus, quiz, base model (Qwen3.5-9B, 4-bit QLoRA, RTX 3090) and judge (Gemma 4 12B Q4_K_M, the helper seat, prompt v4), run entirely through the
web UI on the new architecture: supervisor -> web child + inference worker child (one resident model; training start evicts it).

| Step | Result |
|---|---|
| Mine (exhaustive, Gemma 4 12B in the inference worker) | 3,466 pairs over 19 files in ~1 h 45 min; 96.5 % of the 1,130 facts in some pending answer |
| Manual review (`corpus_review.py`, one reader) | **936 approved** (562 rewritten to add scope, e.g. "Korvane Rate Card 2024 (RC-2024-A)"), 2,530 rejected with a reason; this time one consolidated pair per table / clause instead of one pair per value |
| Reviewer-written additions | 80 (58 new + 22 golden), for facts no approved answer stated literally |
| Approved fact coverage | **1130 / 1130**, 1,016 approved pairs = 1,016 training rows |
| UI replay | 3,466 verdicts in 375 s (Pairs page keyboard flow), 80 additions in 70 s; every pair stamped, every rejection has a reason |
| SFT (rank 64, LR 1e-4, 6 epochs, no early stop, 90/10 split) | 348 steps, 21 min, final train loss 0.15 |
| GGUF q4_k_m + q6_k | 7.5 min |

| Model (q4_k_m) | Quiz pass / partial / fail (of 102) | Unanswerable declined |
|---|---|---|
| SFT, no RAG | 10 / 17 / 75 | 0 / 20 |
| SFT + RAG (top 5) | **94 / 3 / 5** | 2 / 20 |

Retrieval: the expected values were in the retrieved context for 90 / 102 questions; the Track B search check found the facts for 93 / 102.

Reading:
- **Consolidated answers trained worse than atomic ones.** 1,016 long, multi-fact rows x 6 epochs gave 10 % quiz pass vs 24 % for the 2026-10-08 run (2,456 short rows,
  834 steps). Each fact was seen far less often and the 10 % hold-out removed ~100 whole tables / clauses from training. For SFT keep one fact (or one row) per pair; the
  consolidated pairs are good RAG / reading material, not training rows.
- RAG still carries the facts: 94 / 102 with the tuned reader (the same reader scored 86 on 2026-10-08); the tuned reader still never says "not in the documents"
  (2 / 20). Base + RAG remains the recommended setup (§9: 76-97 / 102, 20 / 20 declines); no base Qwen3.5-9B GGUF was on disk for a same-day base run.
- Architecture: the helper (10 GB) was loaded in the worker on purpose before training; the training start dropped it (worker 10,050 -> 312 MiB, same pid, 0
  restarts). The judge swapped models in the same worker after the test run (one model resident at a time) without errors.
- A first training attempt died at step 130 with `CUDA error: an illegal memory access` while the pytest suite ran beside it: a RAG test built a corpus that downloaded
  and loaded a real embedder + reranker onto the 3090 (2.5 GB), card at ~24.6 of 24.5 GB. Re-run alone: clean. Fixed (tests stub the download; registration loads on CPU).

## 11. Helper seat: local Gemma 12B vs API (OpenCode Go `deepseek-v4.1-flash`), 2026-10-12

Same four Korvane files (handbook PDF, rate card XLSX, Oakhaven email thread, KC-Tag 4 quick start PDF: 33 chunks, 255 manifest facts), same exhaustive
miner, upload -> parse -> mine -> done timed per file through the live app (`.tmp/helper_bench.py`). Fact recall = manifest facts stated in any mined answer.

| Helper | Wall time, 4 files | Model load | Pairs | Fact recall |
|---|---|---|---|---|
| Local Gemma 4 12B Q4_K_M (RTX 3090, one request at a time) | 1,225 s | 77 s | 669 | 240 / 255 = 94.1 % |
| API deepseek-v4.1-flash, reasoning **on** (setting silently lost, see below) | 877 s | none | 801 | 239 / 255 = 93.7 % |
| API deepseek-v4.1-flash, reasoning **none**, 6 parallel calls | **81 s** | none | 715 | **246 / 255 = 96.5 %** |

The API helper with reasoning off is ~15x faster than the local GGUF (handbook 39 s vs 634 s), needs no VRAM and found slightly more facts. Raw gateway speed:
~140 tokens/s per call, six parallel calls on one `x-opencode-session` complete in ~5 s, no response caching. The first API run was slow because a seat-only
`PUT /api/settings/helper` rebuilt the provider row without its `reasoning_effort`, so every call thought before answering (fixed: omitted fields keep their stored value).

## 12. Per-file mining jobs (`data/mining/`), API helper, 2026-10-13

All 19 core files, helper deepseek-v4.1-flash (OpenCode Go, reasoning none, concurrency 8), up to 16 files in parallel. Per file: inventory
(rule + helper + whole-file pass) -> mine -> check -> up to 3 fix rounds. Fact recall = manifest facts stated in any pending answer.

| Run | Wall time | Helper calls | Pairs | Manifest recall | Open checklist items (app) |
|---|---|---|---|---|---|
| mining-1 (first version) | 6.5 min | 744 | ~3,950 | 1,100 / 1,130 = 97.3 % | 208 |
| mining-3 (rule items = hard tokens only, union coverage for helper items) | 11.9 min* | 639 | 3,664 | 1,090 / 1,130 = 96.5 % | 185 |
| mining-5 (matcher fixes below + helper skip proposals, prompt mining-v2) | **5.8 min** | 574 | 3,688 | **1,093 / 1,130 = 96.7 %** | **102** (11 proposed skips) |

\* slowed by a 51 s coverage computation the Pairs page polled every 2 s (CPU-bound, starved the job threads); fixed to 1.8 s after the run.
mining-2 died with `database is locked` (fixed: WAL). Old exhaustive miner on the local Gemma 12B for comparison: 3,466 pairs, 96.5 %, 1 h 45 min (§10).

Reading: one file pass reaches the same recall as the old whole-session miner in ~1/10 of the time; no file is `complete` after mining alone
(pairs are pending until reviewed, and every file kept some open items). Most open items are statements the helper judged not to be facts
(headings, chatter): the reviewer excludes them on the file's checklist with a reason, or writes the pair from the item.

Matcher fixes between mining-3 and mining-5 (re-scoring mining-3's own pairs with them alone: 185 -> 148 open):
- a clause/list number at the start ("3.5 ", "17.4 ", "(b) ") is not data; a rule item left with no hard token is no data point;
- "EUR 4,000,000" is one token (it was ``4`` + ``000``, so both amounts in a clause compared as "000"); "57.5m" in an answer also states "57.5";
- two-letter capitals ("AG") and sentence-initial "E-mail" are not codes;
- prose a PDF/RTF parser wrapped mid-sentence is re-joined before the statement split;
- a helper item counts a pair that states ONE of its values when the pair shares the item's other words (it needed half the values
  in one answer, so "327 FTE = 118 + 61 + 49 + ..." could never close: cross-section items 40 -> 12 open). Spot check of 8 items that
  flipped to covered: all 8 really stated by their pairs.

Skip proposals: in a fix round the helper may answer ``{"skip": n, "reason": ...}`` for a checklist entry that is not a trainable fact;
the item stays open until a reviewer confirms. First prompt: 34 proposals, ~14 wrong ("already covered by accepted pairs" - the
helper cannot see the pairs); with "never skip because you think it is covered" (mining-v2): 11 proposals, 7 clearly right (headings,
banter), 4 debatable. Browser walkthrough on mining-5: ops chat file -> confirm 1 skip, write 1 pair, approve 157 -> `complete` ->
"Built 158 training rows from 1 file(s)" without the export override. Found and fixed on the way: checklist rows clipped (global
`.table td` ellipsis), "Approve all pending" approved the WHOLE project while one file was picked, file table stale after a bulk approve.

## 13. Smart Brain on the mining passages (2026-10-14, project `korvane-ragtrace` e9f951f8, results in `results/2026-10-14-passages/`)

After Files & Knowledge the index rows ARE the mining passages (163, median 282 tokens, max 498 of the 512-token e5 window).
Retrieval only (`.tmp/ab_chunking.py`, 102 answerable quiz questions, e5-large + reranking, all expected values in the top-k):

| Index rows | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| mining passages (now) | 73 | **92** | **95** | **96** |
| own 400/80-token chunks (before) | 71 | 85 | 89 | 93 |
| passages + 4,069 mined pairs as fact cards | 76 | 91 | 92 | 95 |

Fact cards sharpen the first hit but crowd multi-value passages out of the top 10, so they stay off by default.

Reader run (`scripts/rag_reader_compare.py`, Gemma 4 12B q4_k_m, top-10, auto context, AI judge deepseek-v4.1-flash):
**93/102 pass, 19/20 unanswerable declined**; misses: not retrieved 2, reader 7, 1 wrongly answered. §8 had 11 never retrieved at top-5.

## 14. Full core mine through the mining worker job, Vision reader on (2026-10-16, project `korvane-mining-12` 496126cb, results in `results/2026-10-16-mining12/`)

Same setup as §12 (19 core files, helper deepseek-v4.1-flash via OpenCode Go, concurrency 8, up to 16 files at once) but mining now
runs in the mining worker job process (b3c611e) and the Vision reader seat is on (local Gemma4-12B QAT Q4_K_M + mmproj): it read the
scanned CAR page and one picture (`[Figure 1 · page 1 · scanned page · read by …]`, `[Figure 1 · page 3 · picture · read by …]`).
Driven over the API (`.tmp/mine_core.py`; the browser walkthrough cannot sign in while the password is on), coverage with
`scripts/corpus_coverage.py --status any` (now sends `FTS_API_TOKEN`).

| Run | Wall time | Helper calls | Pairs | Manifest recall | Open checklist items (app) |
|---|---|---|---|---|---|
| mining-5 (§12) | 5.8 min | 574 | 3,688 | 1,093 / 1,130 = 96.7 % | 102 |
| mining-12 (job process, Vision on) | 8.0 min* | 533 | 3,915 | **1,101 / 1,130 = 97.4 %** | **64** (47 helper, 8 whole-file, 9 rule) |

\* includes the Vision reader load + 2 figure readings at stage 0 and 19 parses (6 s). 18 files ended `gaps`, 1 (`postmortem`) `done`;
no job failed. 487 of 3,915 pairs match no manifest fact (extra or noise: for the reviewer). 254 covered facts are stated by several
answers, 81 only by token-level match.

Open items (`open-items.txt`, 60 of 64 read): about a dozen are chatter, headings or template placeholders (board-pack coffee-machine
aside, "dont panic the owls", the CAR form header, `[address]`), about five are rule fragments with number-only tokens (the MSA
insurance amounts as `1000000 2000000 …`, a depot row, a phone-number fragment `01 47`), the rest (~45) are small real facts the
pairs do not state: notice periods by service length (handbook 12.2, contract), sick-pay days 34-90, the EUR 0.29 rate "do not use",
the 20 cm bulkhead clearance, KC-OPS-114 fields and signatures. Those are the reviewer's: exclude the first group with a reason,
write pairs from the rest (or a fix round with the item texts as the checklist).

The 29 MISSING manifest facts (`coverage-any.json`) are mostly multi-value facts whose values the answers split across pairs
(legal-0049 "successive periods of 12 months" + "120 days' notice", ops-0072 the three KC-OPS-114 fields, product-0010 the
calibration card + serial format); the coverage rule needs every value in ONE answer. Not changed here: the one-fact-per-pair rule
(§10) is what trains; a "set" pair that lists all fields of a form is the Smart Brain's reading material, not a training row.

## 15. Full functional retest through the browser, password on (2026-10-16 evening, throwaway project `ux-walk-func-1016` feeda5da, deleted after)

Track A + B in a real browser against the live service (0.1.0.403 -> 0.1.0.410 as fixes landed), Bearer token for pages and scripts
(`dfd1fc0`). Training data = the 2,455 human-approved golden pairs (`golden/approved_pairs_core_2026-10-08.jsonl.gz`) uploaded through
the Training page, so nothing was approved on the reviewer's behalf; the DPO run used unreviewed builder output (functional only).

| Step | Result |
|---|---|
| Upload + parse (19 core files) | 19/19 parsed in 11 s; manifest facts kept 1,130/1,130 |
| Mining (API helper, Vision on) | 6.5 min, 4,022 pairs, recall 1,099/1,130 = 97.3 % (§14: 97.4 %), no job failed |
| SFT (Qwen3.5-9B, rank 64, LR 1e-4, 3 ep, early stop) | stopped 350/417, 23 min, train 0.48, best eval 1.04 (§1 run 1: 350/417, 0.53, 1.01); adapter + 17 GB merge |
| GGUF export | q4_k_m 5.6 GB + q6_k 7.4 GB in 6 min; app answered in 10 ms throughout |
| Quick index | 30 s, 19 docs / 162 chunks; retrieval@5 98/102 = 96.1 % |
| Quiz, SFT merged (judge deepseek-v4.1-flash) | 14/102 = 13.7 %, unanswerable declined 0/20 |
| Smart Brain quiz, SFT merged reader | 94/102 = 92.2 %, declined 0/20 |
| Smart Brain quiz, untrained base reader | **94/102 = 92.2 %, declined 20/20** (best so far; §8 best was 87) |
| Preference build (11 exact golden replays approved) | 17 pairs (9 hallucination, 8 abstain), 1 min after the VRAM fix (21 min before) |
| DPO from the SFT run (17 unreviewed pairs) | 24 steps, 2 min, loss 0.30 -> 0.04; quiz 7/102, declined 12/20 (abstention moved, recall fell) |
| Guide | tool calls + navigation; after `419b359` it quotes runs, test runs and exports |

Bugs found and fixed in the run: llama.cpp log levels off by one (`6b57c02`: INFO logged as WARNING, WARN as ERROR); transformers
answers ran to max_new_tokens because the base has no generation_config and stops only at `<|endoftext|>` (`9f01951`: 122 quiz
answers 57 min -> 210 s); cuBLAS workspaces pinned a freed 9B's 17 GB segment in the inference worker so the next model spilled to
CPU (`4ff2ff6`); the Guide's readiness summary had no runs (`419b359`); job-run training wrote run state under '<project>-<run>'
(stop notes, post-train errors and the auto-suite FK silently lost, `a8173fb`); the post-training quiz skipped `messages` datasets
(`778cd24`). Harness: token for pages/scripts, folded DPO card and routes, Smart Brain mode renamed from rag (`22865ac`).
Not changed: Qwen3.5's linear-attention layers run on transformers' PyTorch fallback (`flash-linear-attention` / `causal_conv1d`
not installed), so HF training and generation are slower than they need to be.

## 16. Agent browser walkthrough of every feature (2026-10-17, throwaway project `agent-qa-1017` 95061854, deleted after)

One agent drove the live service (0.1.0.413 -> 0.1.0.422) in a real browser, password on, 1440x900. No test scripts: six mixed Korvane
files (pptx, chat export, xlsx, scanned PDF, RTF, eml) were uploaded through Files, mined, reviewed on Pairs, exported, trained on
Training, quizzed on Testing, indexed on Smart Brain, and asked to the Guide. Qwen3.5-9B, preset defaults, 56 approved pairs.

| Step | Result |
|---|---|
| Upload + parse (6 files) | all parsed in < 15 s; scanned CAR form read by the Vision reader (KC-CAR-24-088 complete) |
| Mining (6 files) | 1,003 pairs in < 2.5 min; review UI, reasons, edit+approve, export of 56 approved pairs |
| SFT, full-sequence loss (as shipped) | reported loss 0.046, answer-token loss 0.65 (base 3.21); quiz on its own 56 training questions **16/56 = 28.6 %** |
| SFT, completion-only loss (`fb914ea`) | same data, preset, judge: **30/56 = 53.6 %** |
| Same model, quiz under the trained system prompt (`c4a8626`) | **48/56 = 85.7 %** (2 partial, 6 fail); recall probe 13/16 with the baked prompt vs 4/16 without |
| Merge check | merged weights differ ~0.3 % from base; merged and adapter-on-4-bit recall equal (13/16) |
| Judge | deepseek-v4.1-flash; one call hung 5 min at 21/56 -> 90 s judge timeout (`fd94193`) |
| Smart Brain Quick index during training | correctly refused (GPU busy); message now next to the button, once (`a4362ac`) |
| Guide | tool calls (runs, status, readiness), navigated to Smart Brain and highlighted Build |

Bugs fixed: SPA navigation squeezed the project header and kept the old scroll (`a2fe03d`), Training live panel unsubscribed itself
after SPA navigation (`a2fe03d`), "Select in Training" opened without the dataset (`a2fe03d`), "56 rows · 56 rows" label, Training
template prompt, Testing/Compare progress lines (`a2fe03d`), knowledge search hidden behind "in knowledge" (`2be859b`), full-corpus
quiz listed twice (`2bca201`), full-sequence SFT loss (`fb914ea`), quiz asked without the trained prompt (`c4a8626`), judge timeout
(`fd94193`), Pairs page dead after SPA navigation because the DPO card closed `<details>` with `</div>` (`104b7d1`, plus a tag-balance
test over every template), review list scrolled the whole page (`104b7d1`), activity drawer covered the Training live panel (`2be7045`).
Open: header NEXT link is per page, not per state; mining attributes a chunk that starts mid-record to the whole document (two CAR
pairs) and inverted one board-pack fact (beat -> missed budget), both caught only by human review; warmup 100 steps on a ~100-step
run; export also downloads the JSONL unannounced; Guide reports "no loss recorded yet" mid-run; adapter alpha 128 vs advisor's 256.

## 17. Manual browser walkthrough, local helper (2026-10-18/19, project `korvane-walkthrough-1018` 7d7536dd, runs deleted after)

One agent in a real browser, password on, 0.1.0.429 -> 0.1.0.454 as fixes landed. Helper and Vision reader: Google's Gemma 4 12B QAT
Q4_0 GGUF (local, one call at a time). Six mixed fixtures (pptx, csv, md, scanned PDF, rtf, eml). Training on the 2,491 golden pairs
uploaded through Training (nothing approved on the reviewer's behalf); Qwen3.5-9B, preset "balanced" (rank 64, LR 1e-4, 3 epochs,
batch 2 x 4), QLoRA chosen by the app (18 GiB bf16 did not fit next to training memory). Judge: deepseek-v4.1-flash.

| Step | Result |
|---|---|
| Helper | local seat Test connection: load 72 s, "pong" 0.5 s, unloaded again; Vision "Test reading" 35 s |
| Upload + parse | 6/6 in < 8 s; the Vision reader read the scanned CAR (one long line, markdown escapes: both fixed) |
| Mining, first run (v2 prompts) | cancelled after 28 min: overlap mined twice, filename titles ("the Scanned Corrective Action") |
| Mining, re-run (mining-v3, overlap stripped, real titles) | 48 min, 1,281 pairs, data points 1,495 -> 1,245 (overlap no longer inventoried twice), 98.2 % stated |
| CAR review (every pair against the 3 chunks) | 54 mined: 43 approved, 9 duplicates, 1 echo, 1 wrong value (CAR "due 04.10.2024" = measure M5); 3 reviewer pairs |
| SFT run 1, early stopping | stopped 400/843, best step 250 (eval 1.070), 12.7 min; quiz **8/102**, 15 partial, declined **0/20** |
| SFT run 2, no early stopping | 843 steps, 31.7 min, loss 0.30; quiz **19/102 = 18.6 %**, 19 partial, declined **0/20** |
| GGUF q4_k_m of run 2 | 7 min 47 s, 5.2 GB; chat answers INC-4812 (truth INC-2024-0614): confident and wrong |
| Smart Brain (6 files) | quick index 6 docs / 56 chunks; base Gemma reader: CAR owner + M4 date right with [n] citations, unknown loan margin declined |
| Guide | project_overview tool call, correct counts and next step |

Reading: completion-only loss + trained prompt does not rescue SFT recall on paraphrased questions (§3: 11.8-25.5 %, here 8-19 %), early
stopping on held-out facts keeps an under-trained checkpoint, and fact-tuned models never abstain. Facts belong in the Smart Brain.

Bugs fixed in this run: helper Test connection pinged the API row for the local seat, local seat label from an older default
(`2b53d56`); files table and header layout at 1200 px (`633f601`); flow-bar NEXT refreshes in place (`887380a`, `51a32ea`); mine table
widths (`9d0f5c3`, `54ab6ca`); judge picker offered a missing GGUF (`0ad8d24`); mining titles from reader markers / e-mail subject / caps
heading (`edd2c21`), neighbouring-text context (`d1486d9`), overlap stripped (`f58a021`); Write pair wiped by the 2 s poll (`8541a80`);
bulk reject reason, unticked Approve selected confirms (`5118861`); vision markdown escapes + line breaks (`a43a592`); q/a upload
columns (`1514ba0`); first uploaded dataset missing from the picker (`e264224`); Run test without a model, state-aware banner (`d757c57`);
suite labels (`e4c0809`, `5f24f73`); **GPU probe on the event loop killed the web server twice** (`80f7d02`); Smart Brain settings
loaded the embedder inline, search/chat on the loop, numbered sources (`a74a050`), score column (`4c5ff2e`); run Delete said files stay
(`cead0b4`); every build re-saved the 2.2 GB embedder (`32339ab`).
Open: a correction in a later message is not carried to the chunk with the old value (14,500 vs 12,750 EUR, known since §1); small
records split across chunks still lose their owner with Gemma 12B (merge small chunks into one mining unit, re-measure on the core
corpus); e-mail pairs carry relative times ("this afternoon"); Quick index unloads the chat model without saying so (grounded chat then answers "no model loaded").

## 18. mining-v3 on the 19-file core tier, API helper (2026-10-21, projects `korvane-mining-v3` c17e3255 and `korvane-mining-v3b` 3304cdfe, results in `results/2026-10-21-mining-v3{,b}/`)

Same setup as §14 (helper deepseek-v4.1-flash via OpenCode Go, concurrency 8, up to 16 files at once, Vision reader on: local Gemma 4 12B
QAT), now with mining-v3 (real document titles, neighbouring-text context, overlap stripped). Driven over the API (`.tmp/mine_core.py`),
coverage with `scripts/corpus_coverage.py --tier core --status any`. Two identical runs to see the run-to-run spread.

| Run | Wall time | Helper calls | Data points | Pairs | Manifest recall | Open checklist items (app) | Pairs matching no fact |
|---|---|---|---|---|---|---|---|
| mining-12 (§14, v2) | 8.0 min | 533 | n/a | 3,915 | 1,101 / 1,130 = 97.4 % | 64 | 487 |
| mining-v3 | 6.2 min | 534 | 3,302 | 3,592 | 1,089 / 1,130 = 96.4 % | 58 | 617 |
| mining-v3b | 4.8 min | 516 | 3,191 | 3,189 | 1,088 / 1,130 = 96.3 % | 49 | 350 |

No job failed, no helper call failed. Strict recall is about one point below §14 in both runs; 25 of the 41/42 misses are shared by both
v3 runs, 16 by all three runs, 9 are missed by both v3 runs and covered in §14. Reading those 9 against the pairs:

- most are the strict rule (every value of the fact in ONE answer): "22 years" is stated but not with the message time and job title
  (comms-0031), "three Volvos" without "nothing exciting" (comms-0091), the court without the 30-day referral (legal-0057). v3 answers state
  one value each, which is the mining rule; §14 happened to bundle them.
- a few are real helper misses that the app keeps as **open** checklist items, e.g. "The KC-Tag 4 is the successor of the KC-Tag 3"
  (product-0080): inventoried in both runs, no pair states it, item state `open` (correct). The fix rounds did not write it.
- not a pipeline loss: the datasheet intro reaches the helper unchanged (HTML passages carry no overlap; `strip_overlap` only drops a
  repeated head).

The 350-617 pairs matching no manifest fact are, on a 25-pair sample, real facts the manifest does not list (MSA clause 7.3 exclusions,
2 spares in Rotterdam, leverage 2.74x vs 3.50x covenant), not noise.
Reading: mining-v3 trades about one point of strict manifest recall for 8-19 % fewer pairs (no double-mined overlap) and fewer open
items; the misses that matter stay visible as open items for the reviewer. Small observation: rule-item hard tokens drop single-digit
model numbers ("KC-Tag 4" -> `kc-tag`), so an answer about "KC-Tag 3 end of life" counts toward the KC-Tag 4 rule item.

Addendum (same day): the split-record owner loss of §17 (CAR "due 04.10.2024" = measure M5, local Gemma 12B) no longer
reproduces on the current code. CAR alone, local Gemma 4 12B QAT helper + Vision reader, one job, 9 helper calls, 71 pairs: chunk 2
states "The due date for Measure M5 is 04.10.2024", M1/M2/M4 owners and dates match the form; chunk 3 asks for "the due date for the
record" (vague, not wrong). Joining small passages into one mining unit is therefore not needed for this case (every core passage is
< 1,500 chars, median 994: a size rule would pair up the whole corpus).
