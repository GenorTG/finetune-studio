# Choosing a Smart Brain configuration

This is a decision guide, not a promise that one mode wins on every library. Retrieval quality depends on the
documents, question mix, embedder, chunking, reranker, and answer model. The Quillmoor report records measured
results for one deterministic, synthetic company corpus; its results do not establish performance on arbitrary
customer data.

## The four independent controls

| Control | On | Off | What the benchmark should reveal |
|---|---|---|---|
| Meaning search | Vector similarity over embedded passages | No vector retrieval | Whether semantic similarity helps re-worded questions and related phrasing |
| Word search | BM25 lexical search | No lexical retrieval | Whether exact identifiers, codes, names, and numbers need literal matching |
| Reranker | Cross-encoder reorders retrieved candidates | Keep the initial ranking | Whether a more expensive second-stage model improves top-rank relevance |
| Answer generation | /ask sends retrieved context to a configured chat model | Search-only package; /ask must decline or return a clear unavailable response | Whether generation uses useful context and abstains when evidence is absent |

When both search layers are on, the package fuses the ranked lists with Reciprocal Rank Fusion. The reranker can reorder
only passages already retrieved; it cannot recover evidence that is absent from its candidate set. Answer generation
is downstream of retrieval: a fluent answer does not prove the right passage was found.

## Retrieval combinations: hypotheses to measure

The table describes expected trade-offs to test, not observed scores.

| Meaning | Words | Reranker | Hypothesis |
|---|---|---|---|
| off | off | off | Invalid/empty retrieval; the live API rejects it with HTTP 422 because a package needs at least one way to find passages |
| on | off | off | A useful semantic baseline; likely stronger for paraphrases than exact identifiers |
| off | on | off | A lexical baseline; likely strong for exact codes and values, weaker for paraphrases |
| on | on | off | Hybrid retrieval should balance semantic recall and literal matching; RRF may help or dilute either ranking |
| on | off | on | Reranking may improve ordering when semantic retrieval already surfaced the evidence |
| off | on | on | Reranking may improve ordering for lexical candidates but cannot add missing candidates |
| on | on | on | Highest compute/latency cost; compare its quality gain with the simpler hybrid and single-layer modes |

Each valid search combination is benchmarked with answer generation both off and on. The retrieval score is measured
independently of the answer model, so generation cannot conceal a retrieval miss.

## What Quillmoor measured

The generated company corpus does not produce a universal winner. At M and L, keyword-only search plus the MiniLM
cross-encoder scored higher than hybrid plus the same reranker; without reranking, hybrid won narrowly at M and lost
narrowly at L. Tier S's best cell instead used hybrid search with the larger BGE reranker. Full results and query
denominators are in the [Quillmoor report](quillmoor-2026-10.md) and its machine-generated tables.

| Tier | Keyword only | Hybrid | Keyword + MiniLM | Hybrid + MiniLM |
|---|---:|---:|---:|---:|
| M hit@5 / p50 | 71.1% / 18.3 ms | 74.5% / 119.1 ms | **83.0% / 267.5 ms** | 79.5% / 368.9 ms |
| L hit@5 / p50 | 74.2% / 50.5 ms | 73.3% / 240.9 ms | **81.2% / 276.6 ms** | 77.6% / 136.2 ms¹ |

¹ This recorded L hybrid + reranker result used one client. Earlier local run logs show three attempts at the driver's
default of four clients ending in CUDA illegal-memory-access errors; those failures are not in the per-cell JSON, so the
count is log-derived. The shared GPU leaves the cause unresolved. Legacy worker counts are inferred from the invocation
and marked `4*` in generated tables; don't compare this one-client latency with those rows as matched load conditions.

Tier S's best of 45 tested combinations was E5-large + hybrid + BGE reranker v2 M3 at 90.1% hit@5 (547 quality
questions, 697 latency queries, 4.34 s p50). This differs from the MiniLM reranker results at M/L: model choice matters,
not just which search toggles are on. These scores describe this synthetic corpus, not arbitrary customer data.

## Answer generation: what to expect

With answer generation disabled, /search remains the product surface and /ask must not silently invoke a model.
With it enabled, /ask formats retrieved passages into a grounded prompt and calls the configured OpenAI-compatible
endpoint. The answer model can still make mistakes, ignore evidence, or fail to abstain; retrieval scores alone do
not certify answer quality.

A meaningful answer report includes, per model and mode:

- number of unique questions, successful answers, and judge verdicts;
- AI-judge score (pass = 1, partial = 0.5, fail = 0), shown separately from answer and judge coverage;
- answer latency and recorded source filenames;
- breakdowns for paraphrase, verbatim, changed-value, reverse/by-value, multi-hop, and unanswerable questions;
- manual review of a stratified sample, because an AI judge is an evaluator rather than ground truth.

The score is over judged answers only. Coverage is always shown beside it; a high score on a small surviving subset is
not a successful benchmark.

### Measured answer results (Quillmoor, fixed 376-question subset)

| Model / route | Answered / expected | Judged | Coverage | Score |
|---|---:|---:|---:|---:|
| DeepSeek v4.1 Flash, closed book | 376 / 376 | 376 | 100% | 16.0% |
| DeepSeek v4.1 Flash, live package `/ask` RAG | 376 / 376 | 376 | 100% | **81.1%** |
| Gemma 3 4B, local RAG | 376 / 376 | 376 | 100% | 70.35% |
| Qwen 3.5 4B, local RAG | 376 / 376 | 376 | 100% | **80.19%** |
| Llama 3.2 3B, local RAG | 376 / 376 | 376 | 100% | 71.28% |
| Qwen 3.5 9B, local RAG | 376 / 376 | 376 | 100% | 79.12% |
| Qwen 3.5 0.8B, local RAG (partial after CUDA fault) | 191 / 376 | 191 | 50.8% | 60.21% of judged answers |

Local comparisons retrieve from the same exported E5 hybrid + MiniLM package and use its exact `/ask` prompt builder,
but answer through each local llama-server; they do not issue `/ask` for every local answer. The API RAG row does use
the live package `/ask` route. Scores use pass = 1, partial = 0.5, fail = 0 with the configured AI judge, not human
labels. Multi-hop was the weakest full-coverage category: 2.0–10.0% across Gemma 3 4B, Qwen 3.5 4B, Qwen 3.5 9B,
Llama 3.2 3B, and the API RAG run. The 0.8B score has only 50.8% coverage and must not be compared as a full benchmark result.
Per-type, latency and run details are in the [Quillmoor tables](../../tests/corpus/quillmoor/RESULTS-TABLES.md).

### After the 2026-10-30 default changes

Two Smart Brain defaults changed after the misses above were traced: new indexes on a machine with a GPU rerank with
**bge-reranker-v2-m3** (half precision) instead of MiniLM, and the answer step may run one **follow-up search** when a
question reaches its answer through another item. Same 376 questions, same API answer model and judge:

| Exported E5-large hybrid package | Score | Reverse | Multi-hop | Unanswerable | p50 / p95 answer time |
|---|---:|---:|---:|---:|---:|
| MiniLM reranker (the row above) | 81.1% | 70.0% | 10.0% | 100.0% | 1.78 s / 3.79 s |
| bge-reranker-v2-m3 | 88.4% | 100.0% | 32.0% | 100.0% | 1.78 s / 4.36 s |
| bge-reranker-v2-m3 + follow-up search | **96.1%** | 100.0% | 84.0% | 100.0% | 1.58 s / 6.46 s |

Local answer models on the same packages (same questions and judge):

| Local answer model | MiniLM | bge-reranker-v2-m3 | bge + follow-up |
|---|---:|---:|---:|
| Qwen 3.5 4B | 80.2% | 88.8% | **91.1%** |
| Llama 3.2 3B | 71.3% | **77.7%** | 77.1% |
| Gemma 3 4B | 70.3% | **76.3%** | 76.1% |

The reranker helped every model. The follow-up search helps models that follow its instruction and does nothing for
the two weakest (Gemma also abstained less with it), so the Smart Brain chat has a switch to turn it off.

The reranker alone lifted retrieval hit@5 from 78.6% to 89.2% at about 0.2 s per search on the test GPU; a CPU-only
machine keeps MiniLM. The follow-up costs a second model call on the answers that use it (87 of 376). Details and the
per-stage trace are in [RESULTS.md §8](../../tests/corpus/quillmoor/RESULTS.md).

## Fine-tuning versus retrieval

Fine-tuning and Smart Brain solve different problems. In the matched Quillmoor closed-book SFT test, seven model-family
pairs each improved exact recall on 431 trained facts after tuning (7.4–15.8%), but all seven tuned models scored **0%**
on correct abstention across 431 held-out facts. A grouped Llama 3.2 3B x3 run scored 99.77% on trained facts and 0% on
held-out abstention. That is memorization, not evidence that the model learned to generalize to new customer documents.
The Qwen 3.5 9B pair did not complete and has no score. The judge is AI-based and a deliberately stratified 19-case
audit found disagreements, so these figures are a warning about this recipe, not a universal verdict on fine-tuning.

For facts that can change or were not in the training set, choose Smart Brain/RAG and keep the source documents in the
retrieval corpus. Do not market the measured SFT recall uplift as reliable knowledge retention; the [full matched table
and protocol](../../tests/corpus/quillmoor/RESULTS-TABLES.md) include coverage, judge identity, weighted scores, and
incomplete runs.

## Packaging choices are orthogonal

Archive format (tar.gz, tar, or zip) and encryption (plain or passphrase-encrypted) should not change search behavior.
The export matrix verifies that each package:

1. includes the selected search layers and optional reranker/answer configuration;
2. starts its own standalone server and reports the expected mode from `/health`;
3. gives the same ranked passages for the same search configuration across archive/encryption variants;
4. serves `/ask` only when the answer layer is enabled; and
5. includes the expected encrypted corpus container when encryption is enabled.

Every package was unpacked, queried and checked with the checker loaded by the matrix process. That run predated the stricter recursive scan. A separate fresh audit with the current checker passed on one 2.87 GB hybrid + reranker + answer encrypted tar package: 34 filesystem entries scanned, no unexpected plaintext or symlink issues. The other 71 variants were not rerun under the stricter checker. Public embedder/reranker weights are intentionally plaintext; document/index payloads belong only in `corpus/corpus.enc`.

## Quillmoor protocol and limits

The end-to-end export matrix covers the full 72-case valid cross-product (three search-layer sets × reranker off/on ×
answer off/on × three archive formats × encryption off/on), plus four live rejection checks for invalid no-search
configurations: 76 checks total. Each valid package uses the fixed 60-question paraphrase sample for parity and one
real `/ask` call when answer generation is enabled. Retrieval queries all 697 answerable rows; aggregate quality uses
547 non-verbatim rows, while type breakdowns include the 150 verbatim probes and latency covers all 697 searches.
Answer-model runs use the fixed 376-question evaluation subset, which intentionally includes verbatim training-phrase probes.

Live validation on 2026-10-29 exercised all four combinations with both search layers off (reranker and answer toggled
independently): all four returned HTTP 422 with the expected plain-language reason. The full 72-case valid cross-product also
completed: **72/72 passed**, with no failed or pending rows. Each case exported a standalone archive, checked `/health`, ran
60 parity searches against the configuration reference, and exercised the expected `/ask` behavior.

A valid package case is counted as passed only when export, package file checks, standalone startup, /health, search
parity, and the expected /ask behavior all succeed. Successful rows are reused only when every layer, packaging, and encryption setting matches exactly. Every other
cross-product case remains pending until it has a completed result row. Invalid no-search cases pass only when the live
API rejects the request with HTTP 422 and the plain-language reason. Failed cases are persisted and retried on the next
invocation.
The persisted result file records **76/76 checks passed** (72 valid packages plus four invalid no-search rejections). The separate strict audit is retained with the local benchmark outputs; its one-package scope does not upgrade the other 71 matrix rows to strict-audit passes.

Quillmoor is generated and mechanically checked against known values. It is deliberately varied across 42 generated
file extensions, 29 document archetypes, entities, near-duplicate identifiers, stale values, multi-hop questions, and
unanswerable questions. The parser registry lists 43 extensions; Quillmoor generates 42, with no `.msg` writer. In
addition, the current `.msg` handler does not parse Outlook's binary MSG container, so that entry is not yet proven
supported. Rendered parser self-tests exercise representative synthetic samples, not every real-world layout,
producer, damaged file, or password-protected document. This is a reproducible engineering benchmark, not a substitute
for independently collected, human-authored customer documents or a blinded user study.
