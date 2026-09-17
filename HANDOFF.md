# HANDOFF — finetune-studio

Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## Mission
Ship held-out quality + WebUI evidence that an outside reviewer can't dispute:
real root-cause fixes, real fan-dragon verification, no fake greens.

## State (verified 2026-09-17 21:10 CEST)
| Area | Status |
|------|--------|
| Source-disjoint gap | Fine-tune alone **0/52** strict — needs retrieval grounding |
| **RAG-grounded API** | Committed `6653d31` — `testing/rag_suite.py` + `POST /api/testing/run-rag-suite` |
| **RAG benchmark persistence** | `8a68755`; live benchmark `9050d067` stores explicit Q8 path, transcript, retrieval provenance; audit **1/1 pass, 0 disagreements** |
| **RAG-grounded UI** | Testing card `#t-rag-card` → **Run with RAG** |
| **Full ingested corpus** | `testing/full_corpus_suite.py` + `discover_suites(pid)` hook; all approved source-grounded QA pairs |
| **Full-corpus live result** | 246/246 judged: **230 pass / 16 partial / 0 fail**, weighted **96.7%** at top_k=5 after question-focused scoring and table arithmetic retry |
| **Retrieval live run** | **245/246 hits (99.59%)**; source-miss fallback expands to top-20 and isolates the matched source context |
| **OCR** | Fresh-box tessdata bootstrap argument fixed; local OCR/parser checks **49 passed, 3 skipped** |
| Quality baselines | v3 held-out 17.4% strict; v4 source-disjoint 11.5% weighted unchanged |

### Full-ingested-corpus (Testing dropdown)
- Shared typed helper builds `projects/<pid>/suites/full-ingested-corpus.json` from approved source-grounded QA pairs (no hardcoded project id).
- `discover_suites(project_id)` regenerates/ensures the file when pairs exist; label `local · full-ingested-corpus (N cases)`, `source=project_qa`.
- Auto / local / synthetic / real discovery unchanged. Empty approved set → suite omitted (stale file removed).
- CLI: `scripts/build_full_qa_suite.py` wraps the same helper.
- Does **not** claim run quality — discovery/generation only.

### Live grounded result
- Fan Dragon Q8 + PortableRAG full-corpus run covered **246 cases across 35 source documents**. Raw response retained all transcripts and hit provenance.
- Final live run on `56f4912`: **230 pass / 16 partial / 0 fail**, weighted **96.7%**, average 750.8 ms; model path is the deployed Q8 export.
- Grounded retries correct table variance mix-ups and recover crowded source misses without changing normal top-k=5 behavior.

### Verify (genorbox1)
```
.venv/bin/python -m pytest tests/test_full_corpus_suite.py \
  tests/test_benchmarks_suite_discovery.py \
  tests/test_testing_run_gate.py tests/test_rag_suite.py -v --tb=short
# → 24 passed

.venv/bin/python -m ruff check \
  src/finetune_studio/testing/full_corpus_suite.py \
  src/finetune_studio/benchmarks/suite_defs.py \
  scripts/build_full_qa_suite.py \
  tests/test_full_corpus_suite.py \
  tests/test_testing_run_gate.py
# → All checks passed!
```

## Next steps
1. Persist the final 246-case RAG run as a benchmark row using the live persistence path; retain its raw transcript and audit URL.
2. Investigate the remaining 16 partial cases against parsed-source evidence; do not turn partials into passes without fact coverage.
3. Close the final retrieval miss by improving query/source matching, then rerun the full suite.
4. Keep browser QA green (70/70).

## Commands
```
.venv/bin/python -m pytest tests/test_full_corpus_suite.py \
  tests/test_benchmarks_suite_discovery.py \
  tests/test_testing_run_gate.py tests/test_rag_suite.py -v --tb=short
.venv/bin/python -m ruff check \
  src/finetune_studio/testing/full_corpus_suite.py \
  src/finetune_studio/benchmarks/suite_defs.py \
  scripts/build_full_qa_suite.py \
  tests/test_full_corpus_suite.py \
  tests/test_testing_run_gate.py
```

## Blockers
- Perfect answers are not achieved yet: current grounded result is 230/246 strict passes, 16 partials, and one retrieval miss.
- Browser upload MiniMax quirk unchanged; use API upload fallback.
