# Quillmoor experiment ledger

Every experiment run on the Quillmoor corpus, in order: the question, what ran, the result, where the raw data is and
what came of it. Numbers come from [`RESULTS.md`](RESULTS.md) (the measured record) and
[`RESULTS-TABLES.md`](RESULTS-TABLES.md) (generated tables). Raw artifacts live in `.tmp/quillmoor/<tier>/results/`
(not committed; regenerate with the commands in [`README.md`](README.md)). Scores marked *judged* are the app's AI
Test-judge (DeepSeek v4.1 Flash), not human labels: pass = 1, partial = 0.5, fail = 0.

Status: **shipped** = a product change merged because of it · **finding** = recorded, no product change · **open** =
unresolved.

## 2026-10-29 — building the benchmark

| # | Question | What ran | Result | Data | Status |
|---|---|---|---|---|---|
| 1 | Do 42 file extensions parse? | Generated S/M/L (415 / 1,927 / 10,407 files), live upload + parse, marker substrings | 99.82 / 99.86 / 99.89 % markers found, 0 parse failures; non-picture formats 100 % | `results/ingest.json` | finding (§1, §4) |
| 2 | Why did pictures lose text? | S ingest before/after | parallel Tesseract each used every core: 137 s per photo vs 1.7 s; 14/46 pictures timed out to "(no text found)" | `ingest-before-ocr-fix.json` | **shipped #39** |
| 3 | Re-embed + export with models | export after Re-embed only | 409 "does not reference a shared embedder"; chosen reranker never bundled | logs | **shipped #40** |
| 4 | NaN during training | Qwen3.5-0.8B SFT | gradient NaN from step 820, run still "done" and exported broken weights | run log | **shipped #41** |
| 5 | Concurrent package searches | 4 clients on an exported package | CUDA illegal memory access after 165 searches | `server-*.log` | **shipped #42** |
| 6 | Which retrieval setup is best? | 53 cells: 7 embedders × 3 rerankers × semantic/hybrid/keyword (S), 4 cells M/L | best S 90.1 % hit@5 (E5-large + hybrid + bge-reranker-v2-m3); the reranker mattered more than the embedder | `results/brain/` | finding (§5) → see #13 |
| 7 | Do embedder prompts help? | `bench/prompt_ab.py`, dense only | model prompts changed nothing measurable | `results/prompt_ab.json` | finding |
| 8 | Do all export combinations work? | 72 package variants + 4 invalid requests | 72/72 pass parity + `/ask`; 4/4 invalid refused; 1 strict recursive audit clean | `results/export_combos.json` | finding (§5) |
| 9 | Answers: closed book vs Smart Brain | 376 fixed questions, API model | 16.0 % closed → **81.1 %** RAG (judged); multi-hop 10 % | `answers/deepseek-flash__*` | finding (§6) → see #14 |
| 10 | Local families with RAG | Gemma 3 4B, Qwen 3.5 4B, Llama 3.2 3B, Qwen 3.5 9B (+0.8B partial) | 70.3 / 80.2 / 71.3 / 79.1 % judged; multi-hop 2–8 % | `answers/<model>__rag-default.jsonl` | finding (§6) |
| 11 | Does SFT teach the facts? | 7 base/tuned family pairs + Llama ×3, 431 seen + 431 held-out facts | tuned seen recall 7–16 %, held-out abstention 0 % in every tuned model | `results/sft-facts/` | finding (§6) → see #18–#21 |
| 12 | Qwen 3.5 9B closed-book eval | 6144 and 8192 context | CUDA illegal memory access after 3/862, 1/862, 0/862 answers | `.tmp/bench/*9b*.log` | **open** |

## 2026-10-30 — improving what the benchmark found

| # | Question | What ran | Result | Data | Status |
|---|---|---|---|---|---|
| 13 | Where do reverse look-ups and multi-hop lose the evidence? | `bench/retrieval_stages.py`: rank after dense, keyword, RRF, reranker | reverse: keyword had 37/40 in the top 5, the MiniLM reranker pushed them out; multi-hop: both hops in the top 10 for 17/50 — the second hop is never searched | stdout (re-runnable) | finding → #14, #16 |
| 14 | Is bge-reranker-v2-m3 worth its cost? | `bench/rerank_ab.py`, one client, 547 questions | hit@5 78.6 → **89.2 %**; fp16 keeps 89.2 % at 190 ms p50 (fp32 731 ms) | stdout | **shipped #47** (GPU default) |
| 15 | E5 `query:`/`passage:` prefixes? | dense only, S chunks | 66.5 → 63.6 % hit@5 (worse) | stdout | finding (not adopted) |
| 16 | Let the model ask for a second search | `answer_with_follow_ups`, API model, 376 questions | MiniLM package 81.1 → 85.8 % (first parser) → **90.3 %** (parser that reads a trailing `SEARCH:` line) | `answers/deepseek-flash__rag-default-fu1*.jsonl` | **shipped #47** |
| 17 | Both together | bge package + follow-up, API model | 88.4 % (bge) → **96.1 %**; multi-hop 10 → 84 %, reverse 70 → 100 %, unanswerable 100 % | `answers/deepseek-flash__rag-e5-bge[-fu1].jsonl` | **shipped #47** |
| 18 | Do local models gain too? | Qwen 3.5 4B, Llama 3.2 3B, Gemma 3 4B × {MiniLM, bge, bge + follow-up} | bge: +8.6 / +6.4 / +6.0; follow-up: +2.3 / −0.6 / −0.2, Gemma abstained less (79.2 → 65.8 %) | `answers/<model>__rag-e5-bge[-fu1].jsonl` | **shipped #48** (chat switch) |
| 19 | Can SFT learn to say "I don't know"? | Llama 3.2 3B, ×3 phrasings + 1,893 abstention rows (`bench/train_abstain.py`) | held-out abstention 0 → **14.5 %**; trained-fact recall 99.8 → 95.9 % | `sft-facts/*x3-abstain*` | finding (§9) |
| 20 | Is it phrasings or repetition? | 1 phrasing × 12 epochs vs ×3 phrasings × 4 epochs | 12.5 % (4 passes) → **97.0 %** (12 passes) vs 99.8 % (×3) | `sft-facts/*12ep*`, `*x3-grouped*` | finding → #21 |
| 21 | What do the app's own presets give on 4,306 facts? | Training-page advisor via `finetune.py --tier` | Balanced (2 epochs) **5.8 %**, Precision (4 epochs) **26.7 %** trained-fact recall: the advisor shrank epochs by √(500/pairs) | `sft-facts/*tier-*` | fix in progress (#22) |
| 22 | Advisor keeps the tier's epochs on large datasets | same runs with the corrected advisor (Balanced 6, Precision 12 epochs) | running | `sft-facts/*tier-*-fixed` | **open** |

## Bugs and caveats found along the way

- The SFT write-up said "two question forms per fact"; the evaluation asks one question per fact (fixed in #47).
- The judge is strict on terse answers (a correct "B" judged fail) and inconsistent on partial credit (19-case audit 12/19).
- Records in XML/JSON/YAML files are split across passages (follow-up task filed); most remaining API misses come from it.
- With bge, small models pick the stale value more often on changed-value questions (Llama 3.2 3B 57.7 → 42.3 %).
- The Testing page's "with context" suite does not use the follow-up search, so its score understates the chat.
