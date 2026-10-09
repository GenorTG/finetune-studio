# Handoff — finetune-studio

## Mission

Local fine-tune, data-prep and RAG WebUI. GPU-first; never silently fall back.
Commits, pushes, local service restarts and real-hardware tests are authorized.

## State (verified 2026-10-09)

| Area | State |
|---|---|
| Git | `main`: run-then-judge rebuild of the whole project test suite (this session); see the commit message. |
| Tests | Full suite in 5 chunks green on the final code (chunk 2 has `test_repo_hygiene` red until the new files are committed). `ruff check src/ scripts/` clean. |
| Service | genorbox1 :7860 restarted on the new code (schema migrated, 8 legacy runs rescored). Helper seat = local Gemma 4 12B. OpenCode Go key still has no funds (`402`). |
| Corpus | `tests/corpus/korvane` (19 core files, 1,130 facts), quiz `eval/korvane_quiz_core.jsonl` (102 + 20 unanswerable); hand-labelled judge gold `eval/judge_gold.json` (50 cases). |
| Kept on purpose | project `korvane-ragtrace` (id e9f951f8: index, quiz, 2 runs of the new flow incl. `quiz_live25`); `data/benchmarks/hf_cache` (228 MB). Delete via the UI when done. |

## What changed: run, then judge (Genor 2026-10-09; `docs/judging/RUN-THEN-JUDGE.md`)

The suite used to score answers inline with substring/keyword matchers. Now a test RUN only saves raw transcripts (question, model answer, answer key; RAG: retrieved
context) with no verdict. Judging is a separate step: an AI judge on ANY provider row (default = helper seat; Settings -> Test judge) or the human in the Testing page
(keys 1/2/3/0). Verdicts are appended to `case_judgements` (human > ai > exact > scripted), judge-vs-human agreement is shown, `pass_rate` is `None` until something is
judged, a failed judge call leaves the case unjudged. Auto-judge is a saved setting, default OFF. Public benchmarks (MMLU/GSM8K/HellaSwag, built-in MCQ) keep exact match
(`suite_defs.scoring_mode_for_suite`) and live only on the Benchmarks page; a project quiz sent there is a 400. Code: `testing/{judge,judging,run_store,scoring,audit}.py`,
`db/judgements.py`, `webui/testing_jobs.py` (jobs hold `ENGINE_LOCK`; local judge unloads the tested model), `webui/routes/testing.py`, `templates/project_testing.html`.
Judge quality: checklist prompt v4 = 48/50 on the gold cases at first run (2 gold labels were wrong; 50/50 after fixing them), v3 verdict-only 44/50, all errors too lenient. The set is small and partly tuned against: a sanity check, not a benchmark. Run `scripts/judge_eval.py --provider <id>` before trusting a new judge.

## Result that matters (re-judged, RESULTS.md section 9; Qwen3.5-9B, Korvane, quiz 102 + 20 unanswerable)

SFT run 2 q4_k_m on the Testing page: pass 24 (+20 partial) of 102, unanswerable 0/20 (old matcher: 21 + 20). Base + RAG: 76-97 pass depending on reader/top-k, 20/20
declined; best = Qwen3.5-9B base top-20: 97/102, 20/20. Tuned model + RAG: 86/102 but 2/20 declined. SFT teaches the training questions, not paraphrase-robust facts, and removes
"not in the documents". Use RAG for facts. Retrieval is the remaining RAG loss (11 questions never retrieved; hybrid + ms-marco rerank recall@5 = 90 vs BM25 96).

## Next steps (in order)

1. **Export gate** auto-approves extractive `coverage_fill` pairs for chunks without an approved pair, at every export (unreviewed pairs reach training). Make it opt-in or leave
   them pending; test it (`data/prep/dataset_build.coverage_gate`, `coverage_fill`).
2. Judge the live kept project's runs again once a stronger judge is connected (API row or the 30B MoE: `scripts/judge_eval.py`), then compare against Gemma; keep the better as default.
3. RAG work: default `max_context_chars`, reranker as an RRF vote or multilingual, table-aware row-preserving chunking, default top-k 10-20; rerun `scripts/rag_reader_compare.py`.
4. **Abstain pair builder** (`data/prep/preference.py`): 112/150 abstain questions were answerable from other files; check each against the whole project with RAG. DPO gave no abstention.
5. **Paraphrase augmentation** (`scripts/corpus_paraphrase.py`, funded API key or ~2 h local): 3 re-worded questions per pair, train, compare on the Testing page.
6. Evaluate the adapter on the 4-bit base without merging (bf16 merge beat q4_k_m by 8 points in run 1). Merged-bf16 test needs > 40 min for 122 questions: default to the GGUF.
7. Follow-up cards filed 2026-10-09: Compare tab still keyword-scores (move to run-then-judge); audit MMLU/GSM8K/HellaSwag against official protocols; drop the corpus-specific
   table-arithmetic retry in `rag_suite.run_rag_case`.

## Commands

- Whole UI-only flow: `tests/E2E_MANUAL_GUIDE.md` "Current standard"; driver `FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_track_a.py --list` (phases now run -> judge -> review).
- Judge a saved report offline: `.venv/bin/python scripts/rejudge_reports.py <report.json> --provider <id>`; CLI `fts suite MODEL SUITE --judge [PROVIDER] --out FILE`.
- Review tooling: `scripts/corpus_review.py plan|dump|apply|add|status [--ui]`; gates `corpus_parse_check.py`, `corpus_coverage.py --status approved`.
- Focused tests: `.venv/bin/python -m pytest tests/<file>.py -q -p no:cacheprovider`; full suite = 5 chunks (AGENTS gotcha; chunk 2 and 3 take ~8-10 min under load: run them with nohup); lint `.venv/bin/ruff check src/ scripts/`.
- UI check without a GPU: `.tmp/sbx/run_sandbox.py <port> <dir>` (fake engine + fake judge, redirected HOME, temp cwd).

## Known issues

- spa.js re-entry of full-loaded pages falls back to a full reload (AGENTS gotcha); it would cut an in-flight Guide stream.
- Legacy runs (before 2026-10-09) show "old matcher" verdicts; judge them again from the Testing page.
- The 2026-10-08 `heldout` / `memorization` reports in `results/2026-10-08-ui/` carry no answer keys, so they could not be re-judged.
