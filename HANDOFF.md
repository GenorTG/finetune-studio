# Handoff — finetune-studio

## Mission

Local fine-tune, data-prep and RAG WebUI. GPU-first; never silently fall back.
Commits, pushes, local service restarts and real-hardware tests are authorized.

## State (verified 2026-10-08)

| Area | State |
|---|---|
| Git | `main` pushed: fast Pairs review workspace, Testing-page quiz import + expected-abstention scoring, UI-only Track A/B driver, golden reviewed pairs of both runs, results of both runs. |
| Tests | Full suite on the final code in 5 chunks: see the last commit message / CI. Lint `ruff check src/ scripts/` (the CI scope) clean. |
| Service | genorbox1 :7860, helper seat = local Gemma 4 12B. OpenCode Go key still has no funds (`402`). |
| Corpus | `tests/corpus/korvane` (19 core files, 1,130 facts), quiz `eval/korvane_quiz_core.jsonl` (102 + 20 unanswerable). Golden pairs: run 1 `golden/approved_*`, `rejected_*` (2,491 / 1,964); run 2 `golden/ui_run_*` (2,378 / 1,497); replayable verdicts `results/2026-10-08-ui/`. |
| Last full run | Run 2 = everything through the web UI incl. review and scoring (`tests/corpus/korvane/RESULTS.md` section 7). Project, runs and exports deleted via the UI afterwards; base model kept. |

## Result that matters (run 2, Testing page, q4_k_m)

Quiz 20.6 % (20 partial), unanswerable 0/20; **exact training questions 67 %**, held-out 25 %; tuned model + RAG 78.4 % (2/20 abstain); retrieval@5 90 %.
First run: untrained 2 % (3/20), SFT 17.6 % (0/20), base + RAG 75.5 % (20/20). SFT teaches the training questions, not paraphrase-robust facts, and removes
"not in the documents". Use RAG for facts.

## RAG diagnosis (2026-10-08, RESULTS.md section 8)

Untrained Qwen3.5-9B q4_k_m as RAG reader, cap 16000: 87/102, 20/20 declined (tuned model + RAG: 80/102, 2/20). Old 5000-char context cap fed ~2 of 5 chunks. 11 questions are never retrieved:
the default hybrid + ms-marco rerank gives recall@5 = 90 vs BM25-only 96, hybrid 94, hybrid + RRF-fused rerank 94. Next RAG work: default `max_context_chars` (rag_suite uses 5000; check the RAG chat route too), reranker as an RRF vote or a multilingual one, table-aware chunking, then rerun `scripts/rag_reader_compare.py`. Project `korvane-ragtrace` (id e9f951f8, index + imported quiz) is kept on purpose; delete it via the UI when done.

## Next steps (in order)

1. **Export gate** auto-approves extractive `coverage_fill` pairs for chunks without an approved pair, at every export (unreviewed pairs reach training). Make it opt-in or leave them pending; test it (`data/prep/dataset_build.coverage_gate`, `coverage_fill`). The `reviewed_at` stamp now makes the leak visible.
2. **Testing page**: offer the untrained base model (baseline from the UI); make the merged-bf16 run usable (> 40 min for 122 questions: HF `generate` on Qwen3.5 hybrid layers, 512 max tokens) or warn and default to the GGUF.
3. **Abstain pair builder** (`data/prep/preference.py`): 112/150 abstain questions were answerable from other files; check each against the whole project with RAG. DPO gave no abstention in run 1 and was not re-run.
4. **Paraphrase augmentation** (`scripts/corpus_paraphrase.py`, needs a funded API key or ~2 h local): 3 re-worded questions per pair, train, compare on the Testing-page quiz.
5. Evaluate the adapter on the 4-bit base without merging (bf16 merge beat q4_k_m by 8 points in run 1).
6. Helper load at `n_ctx 32768` OOMs next to the resident RAG embedder (falls back to 16 layers); the CUDA abort at `n_ctx 16384` is still unexplained. Guide still says "SFT or DPO" after a run exists.

## Commands

- Whole UI-only flow: `tests/E2E_MANUAL_GUIDE.md` "Current standard" (phases `a_review_ui`, `a_adds_ui`, `a_test_ui`, `a_eval_ui`, `b_rag_quiz_ui`); driver `FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_track_a.py --list`.
- Review tooling: `scripts/corpus_review.py plan|dump|apply|add|status [--ui]`; gates `corpus_parse_check.py`, `corpus_coverage.py --status approved`; page latency `scripts/bench_review_page.py`.
- Focused tests: `.venv/bin/python -m pytest tests/<file>.py -q -p no:cacheprovider`; full suite = 5 chunks (AGENTS gotcha); lint `.venv/bin/ruff check src/ scripts/`.
- API helper UI E2E: `set -a; . ~/.openclaw/workspace/.env_opcgo; set +a; FTS_ALLOW_LIVE_E2E=1 FTS_E2E_API_KEY="$OPENCODE_GO_API_KEY" .venv/bin/python tests/e2e_helper_settings.py --model deepseek-v4-flash`.

## Known issues

- spa.js re-entry of full-loaded pages falls back to a full reload (AGENTS gotcha); it would cut an in-flight Guide stream.
- Kept on purpose: `data/benchmarks/hf_cache` (228 MB).
