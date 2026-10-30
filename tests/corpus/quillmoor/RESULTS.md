# Quillmoor — measured results

Every number here comes from `tests/corpus/quillmoor/bench/` run against the live app (one machine: RTX 3090 24 GB,
6 cores, 62 GB RAM). The tables marked *generated* come from `bench/report.py` over result files in
`.tmp/quillmoor/<tier>/results/` (not committed); the complete regenerated tables live in
[`RESULTS-TABLES.md`](RESULTS-TABLES.md). Corpus and question types: [README.md](README.md). Write-up for readers:
[`docs/benchmarks/quillmoor-2026-10.md`](../../../docs/benchmarks/quillmoor-2026-10.md).

## 0. Setup of record (2026-10-29)

| | |
|---|---|
| App | `dev` at #38 → #42 during the day (fixes below landed mid-run; runs before a fix are marked) |
| Corpus | seed 20261029; S 415 files / 4,306 facts; M 1,927 / 25,112; L 10,407 / 157,614 |
| Questions (S) | 757 generated; answer sets use a fixed 376-question subset: all 50 two-step, 40 by-value, 26 changed-value, 60 unanswerable + 140 re-worded and 60 verbatim |
| Judge | `testing/judge.py` prompt v4, API connection `deepseek-v4.1-flash` (the app's Test-judge job), 8 parallel |
| Judge check | **50 / 50** hand-labelled Korvane calibration cases matched (11 pass, 19 partial, 20 fail; `scripts/judge_eval.py --provider api-helper`). An earlier 22-case Quillmoor spot-check also matched; expanded SFT spot-checks found only 12/19 agreement across deliberately stratified cases, including judge contradictions and over-crediting. That sample is not representative and the AI judge is not ground truth. |
| Answer models | GGUF via the app's llama.cpp build (`llama-server`, 4 slots, reasoning off, temperature 0 closed book / 0.2 RAG as the package's `/ask` uses), q8_0 for ≤ 4B, q4_k_m for 7-9B; API model through the app's connection |
| Training recipe (all families) | SFT LoRA r32 / α64, lr 2e-4, 4 epochs, batch 8 × 2, max 256 tokens, merged + GGUF in the same job; the app holds out 10 % of rows (seeded) |

## 1. Files & Knowledge (ingest)

| Tier | Files | MB | Upload | Parse done after upload | Parse failures | Marker substrings present | non-picture marker hits | picture / scan marker hits |
|---|---|---|---|---|---|---|---|---|
| S (before #39) | 415 | 32.9 | 14 s | 317 s | 0 | 4,445 / 4,493 = 98.93 % | 100 % ¹ | 127 / 172 = 73.8 % |
| S | 415 | 32.9 | 26 s | 30 s | 0 | 4,485 / 4,493 = 99.82 % | 4,321 / 4,321 = 100 % | 164 / 172 = 95.3 % |
| M | 1,927 | 146.0 | 137 s | 62 s | 0 | 25,921 / 25,957 = 99.86 % | 25,111 / 25,111 = 100 % | 810 / 846 = 95.7 % |
| L | 10,407 | 773.1 | 762 s | 202 s | 0 | 161,868 / 162,050 = 99.89 % | 157,397 / 157,397 = 100 % | 4,471 / 4,653 = 96.1 % |

¹ The first S run counted 3 `.xml` values lost; they were present (`port="21157" />` re-serialised with a space) and
the matcher was too strict. The corrected S/M OCR totals are 164/172 and 810/846: S missed eight superscript units in
`m³/h`; M missed 35 such equipment ratings and one call sign. The Files page of the L project (10,407 files) loads in
2.6 s and filters by name in under 1.5 s. The generated corpus exercises 42 extensions; the per-extension substring
and identifier-boundary counts are in the generated [`RESULTS-TABLES.md`](RESULTS-TABLES.md). These synthetic fixtures
do not establish coverage for every real producer or layout. The registry contains 43 extensions, but `.msg` is
currently routed through the RFC822 parser and has not been shown to parse Outlook's binary MSG format.

## 2. Bugs found by this benchmark (all fixed, with regression tests)

| PR | What the benchmark hit | Fix |
|---|---|---|
| #39 | Parallel parses each started a Tesseract with one OpenMP thread per core: one 1200×500 nameplate photo took **137 s** instead of **1.7 s**, near the 180 s limit after which a picture silently parses to "(no text found)" — 14 of 46 pictures and scans of S lost their text | OCR runs single-threaded (parallelism comes from parsing files in parallel) |
| #40 | After **Re-embed only** with another embedder, export with models refused (409 "does not reference a shared embedder"); a reranker chosen in Search settings (`bge-reranker-v2-m3`, offered by the page) was never bundled — the package ran without reranking and said nothing; every rebuild reset the reranker to the default | re-embed and build register both models in the shared store; export fetches a reranker named by its HF id or says why it cannot |
| #41 | Qwen3.5-0.8B SFT healthy to step 810 (loss 0.316), gradient NaN from step 820, loss logged as 0.0 for the last 150 steps — the run finished **done** and merged / exported broken weights | the trainer stops at the first non-finite loss or gradient; the run fails with the step and the last checkpoint before it; nothing is saved, merged or exported |
| #42 | 4 clients searching an exported package crashed its server after 165 searches ("CUDA error: an illegal memory access"); concurrent first requests loaded a model twice | one lock for loading and running the bundled models (the studio's own search already serialises each model) |
| Report | The summary matrix omits per-query rows, so the report showed latency `n=0` despite non-zero percentiles and 697 rows in each per-configuration artifact | report now reads the separate run artifact; regression fixture matches the persisted matrix schema |

## 3. The card itself

Six NVIDIA Xid events were logged on the RTX 3090 during this day, all while 4-5 processes shared it at its 350 W
power cap and 76-77 °C: Xid 13 + 43 (SM warp exception, out-of-range address) in llama-server twice, Xid 31 (MMU
fault) in a package server twice. Every one reports the same unit (GPC 2), across llama.cpp and PyTorch processes;
the kernel log shows none since 2026-10-20 before this load, and the Korvane log recorded one such abort on
2026-10-08. #42 is a real thread-safety fix, but the faults also hit a single-threaded, single-model server, so the
card looks marginal under sustained full multi-process load. Runs hit by a fault were repeated; the drivers now
restart a server that dies.

## 4. Per-extension coverage

The table below reports substring marker counts before rounding: an NFKC-normalized, case/whitespace-folded spelling
appearing anywhere in parsed text. That is a parser-output presence signal, not exact field preservation. The same parsed
files were re-read to compute a separate identifier-boundary count; hyphens are treated as identifier characters, but
the check is still not field-aware. Re-reading used `--skip-upload`, so it did not remeasure upload or parser timing.
The generated [`RESULTS-TABLES.md`](RESULTS-TABLES.md) contains both counts by extension and archetype. Aggregate
identifier-boundary results are S 4,485/4,493 (99.82%), M 25,921/25,957 (99.86%), and L 161,868/162,050 (99.89%),
identical to substring totals; the original full-ingest runs had zero parse failures. `.pdf(scan)` is the generated OCR
subtype of `.pdf`; the corpus contains 42 generated filename extensions. These synthetic fixtures are not a claim that
every real-world producer or layout is supported.

| Type | S files | S values | S found | S found % | M files | M values | M found | M found % | L files | L values | L found | L found % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `.bmp` | 6 | 21 | 19 | 90.5 | 26 | 89 | 85 | 95.5 | 142 | 508 | 484 | 95.3 |
| `.cfg` | 5 | 12 | 12 | 100.0 | 15 | 37 | 37 | 100.0 | 58 | 142 | 142 | 100.0 |
| `.conf` | 5 | 12 | 12 | 100.0 | 15 | 37 | 37 | 100.0 | 58 | 142 | 142 | 100.0 |
| `.css` | 14 | 14 | 14 | 100.0 | 48 | 48 | 48 | 100.0 | 180 | 180 | 180 | 100.0 |
| `.csv` | 5 | 200 | 200 | 100.0 | 23 | 1618 | 1618 | 100.0 | 115 | 10054 | 10054 | 100.0 |
| `.doc` | 15 | 118 | 118 | 100.0 | 78 | 587 | 587 | 100.0 | 449 | 3348 | 3348 | 100.0 |
| `.docx` | 15 | 80 | 80 | 100.0 | 77 | 463 | 463 | 100.0 | 449 | 2776 | 2776 | 100.0 |
| `.eml` | 15 | 96 | 96 | 100.0 | 76 | 466 | 466 | 100.0 | 448 | 2876 | 2876 | 100.0 |
| `.epub` | 15 | 107 | 107 | 100.0 | 78 | 597 | 597 | 100.0 | 449 | 3350 | 3350 | 100.0 |
| `.gif` | 5 | 14 | 14 | 100.0 | 26 | 89 | 84 | 94.4 | 142 | 508 | 488 | 96.1 |
| `.htm` | 15 | 76 | 76 | 100.0 | 77 | 450 | 450 | 100.0 | 449 | 2925 | 2925 | 100.0 |
| `.html` | 16 | 110 | 110 | 100.0 | 77 | 608 | 608 | 100.0 | 449 | 3524 | 3524 | 100.0 |
| `.ini` | 5 | 13 | 13 | 100.0 | 15 | 37 | 37 | 100.0 | 58 | 142 | 142 | 100.0 |
| `.jpeg` | 6 | 18 | 15 | 83.3 | 26 | 89 | 83 | 93.3 | 141 | 504 | 480 | 95.2 |
| `.jpg` | 14 | 21 | 20 | 95.2 | 66 | 89 | 85 | 95.5 | 352 | 508 | 485 | 95.5 |
| `.js` | 5 | 18 | 18 | 100.0 | 16 | 61 | 61 | 100.0 | 60 | 227 | 227 | 100.0 |
| `.json` | 6 | 10 | 10 | 100.0 | 23 | 444 | 444 | 100.0 | 116 | 5150 | 5150 | 100.0 |
| `.jsonl` | 6 | 502 | 502 | 100.0 | 23 | 2506 | 2506 | 100.0 | 116 | 15358 | 15358 | 100.0 |
| `.jsx` | 7 | 7 | 7 | 100.0 | 24 | 24 | 24 | 100.0 | 90 | 90 | 90 | 100.0 |
| `.log` | 6 | 26 | 26 | 100.0 | 20 | 194 | 194 | 100.0 | 80 | 1357 | 1357 | 100.0 |
| `.markdown` | 15 | 80 | 80 | 100.0 | 77 | 447 | 447 | 100.0 | 449 | 2911 | 2911 | 100.0 |
| `.md` | 15 | 70 | 70 | 100.0 | 77 | 454 | 454 | 100.0 | 449 | 2767 | 2767 | 100.0 |
| `.odp` | 15 | 134 | 134 | 100.0 | 78 | 996 | 996 | 100.0 | 449 | 6536 | 6536 | 100.0 |
| `.ods` | 6 | 524 | 524 | 100.0 | 24 | 2682 | 2682 | 100.0 | 116 | 15538 | 15538 | 100.0 |
| `.odt` | 16 | 104 | 104 | 100.0 | 78 | 554 | 554 | 100.0 | 449 | 3234 | 3234 | 100.0 |
| `.pdf` | 15 | 82 | 82 | 100.0 | 77 | 467 | 467 | 100.0 | 449 | 2776 | 2776 | 100.0 |
| `.pdf(scan)` | 12 | 28 | 28 | 100.0 | 74 | 134 | 133 | 99.3 | 377 | 597 | 593 | 99.3 |
| `.png` | 15 | 18 | 17 | 94.4 | 66 | 89 | 85 | 95.5 | 352 | 508 | 484 | 95.3 |
| `.pptx` | 15 | 172 | 172 | 100.0 | 77 | 892 | 892 | 100.0 | 449 | 6514 | 6514 | 100.0 |
| `.py` | 4 | 17 | 17 | 100.0 | 16 | 71 | 71 | 100.0 | 60 | 247 | 247 | 100.0 |
| `.rtf` | 15 | 100 | 100 | 100.0 | 78 | 564 | 564 | 100.0 | 450 | 3248 | 3248 | 100.0 |
| `.tif` | 5 | 17 | 17 | 100.0 | 26 | 89 | 85 | 95.5 | 142 | 508 | 486 | 95.7 |
| `.tiff` | 6 | 18 | 17 | 94.4 | 26 | 89 | 85 | 95.5 | 141 | 504 | 483 | 95.8 |
| `.ts` | 5 | 20 | 20 | 100.0 | 16 | 63 | 63 | 100.0 | 60 | 234 | 234 | 100.0 |
| `.tsv` | 6 | 418 | 418 | 100.0 | 23 | 2472 | 2472 | 100.0 | 116 | 15538 | 15538 | 100.0 |
| `.tsx` | 7 | 7 | 7 | 100.0 | 24 | 24 | 24 | 100.0 | 90 | 90 | 90 | 100.0 |
| `.txt` | 16 | 109 | 109 | 100.0 | 77 | 585 | 585 | 100.0 | 449 | 3877 | 3877 | 100.0 |
| `.webp` | 5 | 17 | 17 | 100.0 | 26 | 89 | 85 | 95.5 | 142 | 508 | 488 | 96.1 |
| `.xls` | 5 | 404 | 404 | 100.0 | 23 | 2542 | 2542 | 100.0 | 116 | 15558 | 15558 | 100.0 |
| `.xlsx` | 14 | 120 | 120 | 100.0 | 66 | 968 | 968 | 100.0 | 353 | 4944 | 4944 | 100.0 |
| `.xml` | 6 | 159 | 159 | 100.0 | 23 | 1156 | 1156 | 100.0 | 116 | 7340 | 7340 | 100.0 |
| `.yaml` | 6 | 270 | 270 | 100.0 | 23 | 900 | 900 | 100.0 | 116 | 7192 | 7192 | 100.0 |
| `.yml` | 5 | 130 | 130 | 100.0 | 23 | 1097 | 1097 | 100.0 | 116 | 7212 | 7212 | 100.0 |
| **all** | **415** | **4493** | **4485** | **99.8** | **1927** | **25957** | **25921** | **99.9** | **10407** | **162050** | **161868** | **99.9** |

- S: 415 files, 32.9 MB, upload 26 s, parse finished 30 s after the upload; files that failed to parse: 0
- M: 1927 files, 146.0 MB, upload 137 s, parse finished 62 s after the upload; files that failed to parse: 0
- L: 10407 files, 773.1 MB, upload 762 s, parse finished 202 s after the upload; files that failed to parse: 0

## 5. Smart Brain retrieval

All **53/53** retrieval cells completed through exported standalone packages: S 45/45 across seven embedders, three
reranker settings, semantic/hybrid search, and one keyword-only cell per reranker; M and L four cells each. The score
denominator excludes verbatim probes; latency includes every answerable query.

An independent audit compared every saved result row-ID set with the expected answerable question IDs: all 53 artifacts
matched exactly (S: 697 IDs × 45, M: 1,197 × 4, L: 1,918 × 4; no missing, extra, or duplicate IDs). The resume guard now
enforces exact ID-set equality rather than accepting row counts alone.

| Tier | Setting | quality / latency queries | hit@5 | MRR@10 | p50 ms | clients |
|---|---|---:|---:|---:|---:|---:|
| S | E5-large + hybrid + BGE reranker v2 M3 | 547 / 697 | **90.1%** | 0.841 | 4,335.7 | 4 |
| M | E5-large + keyword, no reranker | 1,047 / 1,197 | 71.1% | 0.627 | 18.3 | 4 |
| M | E5-large + hybrid, no reranker | 1,047 / 1,197 | 74.5% | 0.617 | 119.1 | 4 |
| M | E5-large + keyword + MiniLM reranker | 1,047 / 1,197 | **83.0%** | 0.728 | 267.5 | 4 |
| M | E5-large + hybrid + MiniLM reranker | 1,047 / 1,197 | 79.5% | 0.686 | 368.9 | 4 |
| L | E5-large + keyword, no reranker | 1,768 / 1,918 | 74.2% | 0.649 | 50.5 | 4 |
| L | E5-large + hybrid, no reranker | 1,768 / 1,918 | 73.3% | 0.619 | 240.9 | 4 |
| L | E5-large + keyword + MiniLM reranker | 1,768 / 1,918 | **81.2%** | 0.710 | 276.6 | 4 |
| L | E5-large + hybrid + MiniLM reranker | 1,768 / 1,918 | 77.6% | 0.661 | 136.2 | 1 |

Each tier contains 60 unanswerable questions; these are excluded from retrieval metrics. The recorded L hybrid+MiniLM
result is a one-client run. Earlier local runner logs show three attempts with the default four clients ending in CUDA
illegal-memory-access errors, but failed attempts are not stored in the per-cell JSON; the count is a log-derived
observation, not a matrix result. The GPU was shared, so this does not establish concurrency as the root cause or prove
high-concurrency reliability. Serial latency is measured and is not comparable to four-client rows. The generated
tables mark inferred legacy worker counts as `4*`; the one-client rerun records its worker count directly.

Hybrid was not a universal win: with the MiniLM reranker, keyword-only beat hybrid at both M and L. Without reranking,
hybrid gained at M but lost slightly at L. The complete per-cell/per-type data and p95 are in
[`RESULTS-TABLES.md`](RESULTS-TABLES.md).

## 6. Answer models and fine-tuning

The fixed answer evaluation uses 376 questions and the app's configured DeepSeek v4.1 Flash AI judge (pass = 1,
partial = 0.5, fail = 0). The API baseline scored 16.0% closed-book and **81.1%** through the live default Smart Brain
package's `/ask` route (376/376 judged). Local models used that same exported package's `/search` results and exact
`/ask` prompt builder, then answered through local llama-server; the cross-family RAG runs do not call `/ask` for each
answer. Their current results are Gemma 3 4B 70.35%, Qwen 3.5 4B **80.19%**, Llama 3.2 3B 71.28%, and Qwen 3.5 9B
79.12%, each with 376/376 answers and judgments. Qwen 3.5 0.8B is partial: 191/376 answered and judged (50.8%
coverage), scoring 60.21% only on that surviving subset after one bounded retry hit a local CUDA illegal-memory-access
error; the rest were marked unavailable. Full family, per-type, latency and coverage rows are generated in
[`RESULTS-TABLES.md`](RESULTS-TABLES.md). Multi-hop remains a major weakness, and all scores are AI-judge results, not
human-ground-truth labels. The judge passed a 50/50 Korvane calibration and 22 Quillmoor verdicts were manually
checked; the full answer matrix was not human-graded.

An earlier, separate **17-question** held-out slice scored 0/17 for each of four base models and 1/17 for each tuned
model. An agent manually checked the 68 tuned answers and agreed with that one correct answer per family. That sample
was too small to establish meaningful generalization; its local per-case review is excluded from the public report.

The larger matched closed-book SFT evaluation uses **431 trained facts + 431 app-held-out facts per run**, one
question per fact: the fact's training question, asked exactly as trained for seen facts (**862 expected answers**). Seven base/tuned family pairs completed, one Qwen 3.5 9B pair is
unscored after runtime failures, and one grouped Llama 3.2 3B x3 run completed. Every scored evaluation has 862/862
answers and AI judgments, zero split overlap, and zero judge errors. The table shows exact-pass rates; full weighted
scores, latency, contexts and coverage are in
[`RESULTS-TABLES.md`](RESULTS-TABLES.md).

| Model family | Base: seen-fact pass | Base: held-out correct abstention | Tuned: seen-fact pass | Tuned: held-out correct abstention |
|---|---:|---:|---:|---:|
| Granite 3.3 2B | 0.23% | 94.2% | 9.51% | 0.0% |
| OLMo 2 1B | 0.70% | 46.64% | 11.37% | 0.0% |
| Qwen 3 1.7B | 0.46% | 68.68% | 7.42% | 0.0% |
| Qwen 3.5 4B | 0.23% | 92.58% | 10.67% | 0.0% |
| Qwen 3.5 9B | — | — | — | — |
| Llama 3.2 3B | 0.0% | 99.77% | 12.53% | 0.0% |
| Llama 3.1 8B | 0.0% | 99.54% | 15.78% | 0.0% |
| Mistral 7B | 0.46% | 97.22% | 12.06% | 0.0% |

Across the seven pairs, tuning increased seen-fact recall but every tuned model scored **0%** on correct abstention for
app-held-out facts. The grouped Llama x3 run reached 99.77% on trained facts and 0% on held-out abstention—a direct
memorization signal, not generalization. Qwen 3.5 9B base and tuned runs are **incomplete and unscored**: the initial
6144-context evaluations hit CUDA illegal-memory-access failures after 3/862 and 1/862 answers. A separate full base
retry at context 8192 failed on its first request (0/862 answers, one connection error). In contrast, a serial four-
prompt diagnostic using the same prompts passed at both 6144 and 8192; that small pilot does not reproduce the full-run
fault or establish stability. A previous 9B RAG run at slot context 8192 completed. The closed-book CUDA cause remains
unresolved; no context-size fix is claimed.

These percentages are the app's DeepSeek v4.1 Flash Test-judge outputs, not ground truth. A deliberately outcome-stratified
19-case human/agent audit agreed with the judge on 12 cases and found contradictory reasoning and over-crediting; it is
not representative and does not support an error-rate estimate. Taken together, the results show a measurable increase
in recall accompanied by catastrophic loss of held-out abstention for this recipe and synthetic task. They do not prove
SFT is ineffective in general, but this result is not a releasable quality claim.

## 7. Scope and limitations

These scores apply to the deterministic synthetic Quillmoor files, fixed prompts, tested model versions, and this
shared RTX 3090 host. The generated formats and parser survival checks do not prove every real producer/layout; the
answer judge is automated; the held-out quality slice has only 17 facts; and no browser-level feedback walkthrough is
included. Do not present these as general customer-data accuracy or a completed beta gate.

## 8. Answer-quality changes (2026-10-30)

Two changes to the Smart Brain defaults, measured on tier S before they shipped. The answer model is the app's API
connection (DeepSeek v4.1 Flash), judged by the same Test-judge as §6 on the same fixed 376 questions; every row has
376/376 answers and judgments.

**Where the misses came from.** A per-stage trace on the S index (`bench/retrieval_stages.py`) showed
reverse look-ups ("which asset has serial SN…") failing in the *reranker*, not in retrieval: keyword search had the
right passage in its top 5 for 37/40, and the MiniLM reranker pushed it out. Two-step questions failed because the
second hop is never searched: "who manages the site where UVR-4655-R is installed?" finds the asset's row (it names the
site) but not the site profile, whose name is not in the question. Both hops reached the top 10 for 17/50.

**Reranker.** Same hybrid candidates (top 50), one client, RTX 3090, 547 non-verbatim questions (`bench/rerank_ab.py`):

| Reranker | hit@5 | reverse@5 | multi-hop@5 | p50 | p95 | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| ms-marco-MiniLM-L-6-v2 (old default) | 78.6% | 24/40 | 3/50 | 87 ms | 101 ms | 0.9 GB |
| bge-reranker-v2-m3, fp32 | 89.2% | 40/40 | 10/50 | 731 ms | 906 ms | 3.3 GB |
| bge-reranker-v2-m3, fp16 | 89.2% | 40/40 | 10/50 | 190 ms | 241 ms | 1.9 GB |

**Follow-up search.** The answer model may reply `SEARCH: <query>` instead of answering; the server runs that search,
interleaves its passages with the first ones and asks again without the option (`answer_with_follow_ups` in
`smart_brain/standalone_server.py`, used by the package's `/ask` and the studio's Smart Brain chat). One follow-up.

| Exported package (E5-large + hybrid) | score | paraphrase | verbatim | superseded | reverse | multi-hop | unanswerable | p50 | p95 | follow-ups |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniLM reranker (the §6 baseline) | 81.1% | 92.1 | 96.7 | 96.2 | 70.0 | 10.0 | 100.0 | 1.78 s | 3.79 s | — |
| MiniLM + follow-up (first parser) ¹ | 85.8% | 92.5 | 96.7 | 92.3 | 70.0 | 46.0 | 100.0 | 2.03 s | 5.12 s | 23 |
| bge-reranker-v2-m3 | 88.4% | 95.0 | 96.7 | 98.1 | 100.0 | 32.0 | 100.0 | 1.78 s | 4.36 s | — |
| **bge-reranker-v2-m3 + follow-up** | **96.1%** | 96.1 | 98.3 | 100.0 | 100.0 | 84.0 | 100.0 | 1.58 s | 6.46 s | 87 |

¹ The first parser only accepted a reply that *was* the search line; the model often explained first and put
`SEARCH:` on its last line (12 of the 23 unaided two-step failures). The shipped parser takes the last `SEARCH:` line
anywhere in the reply. The bge rows use it.

The package's `/search` ran the old copy of `server.py` (fp32 reranker, no length cap); retrieval results do not depend
on the precision (hit@5 identical above), latency does. The follow-up costs a second model call when it is used: 87 of
376 answers here, 38 of them on unanswerable questions (the model looked once more, then said the documents do not
contain it: unanswerable stayed 100 %). The remaining failures are mostly records cut in two by the passage chunker
(an XML catalogue entry's SKU at the end of one passage, its supplier in the next) and judge strictness (a correct
"B" marked fail).
