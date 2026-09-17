# HANDOFF — finetune-studio

Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## Mission
Ship held-out quality + WebUI evidence that an outside reviewer can't dispute:
real root-cause fixes, real fan-dragon verification, no fake greens.

## State (verified 2026-09-17 18:05 CEST)
| Area | Status |
|------|--------|
| Source-disjoint gap | Fine-tune alone **0/52** strict — needs retrieval grounding |
| **RAG-grounded API** | Committed `6653d31` — `testing/rag_suite.py` + `POST /api/testing/run-rag-suite` |
| **RAG-grounded UI** | Testing card `#t-rag-card` → **Run with RAG** |
| **Full ingested corpus** | `testing/full_corpus_suite.py` + `discover_suites(pid)` hook; all approved source-grounded QA pairs |
| **Full-corpus live result** | 246/246 judged: **143 pass / 79 partial / 24 fail**, weighted **74.2%** at top_k=5; top_k=10: 147/73/26, weighted **74.6%** |
| **Retrieval live run** | top_k=5: **243/246 hits (98.78%)**; top_k=10: **244/246 (99.19%)**; top_k=5 remains the UI default because extra context adds distractors |
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
- Retrieval found the declared source for 243/246 cases. The remaining answer failures are real or legacy over-broad QA targets; scoring was not weakened.

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
1. Commit and deploy current UI/full-corpus/OCR changes; verify `full-ingested-corpus (246 cases)` in Fan Dragon Testing.
2. Re-run OCR upload → promote-to-source after deployment using the API probe (verified: parsed marker present in both probes).
3. Correct the 24 full-corpus failures only from parsed-source evidence; rerun the full grounded suite.
4. Persist full-corpus RAG benchmark runs with artifact paths and raw transcripts.
5. Keep browser QA green (70/70).

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
- Perfect answers are not achieved yet: current grounded result is 143/246 strict passes and retrieval misses 3 cases.
- Browser upload MiniMax quirk unchanged; use API upload fallback.
