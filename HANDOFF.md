# Handoff — finetune-studio

## Mission

Local fine-tune, data-prep and RAG WebUI. GPU-first; never silently fall back.
Commits, pushes, local service restarts and real-hardware tests are authorized.

## State (verified 2026-10-09)

| Area | State |
|---|---|
| Git | `main` = run-then-judge rebuild + the 2026-10-09 pass: export gate opt-in, Compare tab on run-then-judge, row-preserving RAG chunking + top-k 10, missing-model 400, script fix. |
| Tests | Full suite in 5 CI shards green on the merged code (`scripts/ci_shard.py i 5`); `ruff check src/ scripts/` clean. |
| Live browser E2E (this pass) | Testing page: run (live progress), AI judge (Gemma helper, 26/26), human keys 1/3, key 0 restores the AI verdict, agreement chip. Drivers vs live app: `a_review_ui_judge` 12/12, `b_rag_quiz_ui` 10/10, `a_test_ui` 8/8. Scripts live: `rejudge_reports.py` ok, `rag_reader_compare.py` ok after the JSONL-import fix. Export page: opt-in checkbox off by default, pill "would add 415 pairs from 154 passages". Compare page renders. |
| Not browser-verified | Starting a comparison from the Compare form (the project has one model only: needs 2); the Export checkbox actually changing an export; old indexes still use word chunks until rebuilt. |
| Service | genorbox1 :7860 on the merged code. Helper seat = local Gemma 4 12B. OpenCode Go key still has no funds (`402`). |
| Corpus | `tests/corpus/korvane` (19 core files, 1,130 facts), quiz `eval/korvane_quiz_core.jsonl` (102 + 20 unanswerable); hand-labelled judge gold `eval/judge_gold.json` (50 cases). |
| Kept on purpose | project `korvane-ragtrace` (id e9f951f8: index, quiz; its `base_model` now points at the helper Gemma GGUF because the old Qwen base GGUF was deleted); `data/benchmarks/hf_cache` (228 MB). |

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

1. **Measure the new chunker:** rebuild the project index (RAG page), rerun `scripts/rag_reader_compare.py --pid e9f951f8 --reader <name>=<gguf> --top-k 10` (live, before the rebuild: Gemma 12B top-k 10 = 92/102 quiz,
   20/20 declined, 4 not retrieved, 6 reader misses) and compare. Then decide the reranker (RRF vote or multilingual).
2. Judge the kept project's runs again with a stronger judge (API row or the 30B MoE: `scripts/judge_eval.py`), compare against Gemma, keep the better as default.
3. Compare tab leftovers: move `_target_model` (lazily imported by `routes/comparison.py`) into `testing_models.py`; list `kind=compare` runs as a group on the Testing page; click through a real 2-model comparison.
4. **Abstain pair builder** (`data/prep/preference.py`): 112/150 abstain questions were answerable from other files; check each against the whole project with RAG. DPO gave no abstention.
5. **Paraphrase augmentation** (`scripts/corpus_paraphrase.py`, funded API key or ~2 h local): 3 re-worded questions per pair, train, compare on the Testing page.
6. Evaluate the adapter on the 4-bit base without merging (bf16 merge beat q4_k_m by 8 points in run 1). Merged-bf16 test needs > 40 min for 122 questions: default to the GGUF.
7. Follow-ups: audit MMLU/GSM8K/HellaSwag against official protocols; `rag_suite.run_rag_case` still widens retrieval using the case's gold `source_id` (inflates recall: drop it or report it separately).

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
