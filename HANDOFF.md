# Handoff — finetune-studio

## Mission

Local fine-tune, data-prep and RAG WebUI. GPU-first; never silently fall back.
Commits, pushes, local service restarts and real-hardware tests are authorized.

## State (verified 2026-10-09, evening pass)

| Area | State |
|---|---|
| Git | `main` = run-then-judge rebuild + both 2026-10-09 passes; every leftover of the morning pass is done and browser-verified (below). |
| Tests | Full suite green in 5 CI shards on the merged code; `ruff check src/ scripts/` clean. |
| Browser-verified live (this pass) | Export page: opt-in checkbox off by default, pill count, blocked-export confirm, forced export of 415 unreviewed pairs with the right wording. Testing page: Dataset check run (driver `a_eval_ui` 7/7), comparison runs grouped under one header with a working deep link. Compare page: 2-model run (project base + a library model), group judge, human override ("you"), undo restoring the AI verdict. RAG page: rebuild from scratch on the new chunker (19 docs, 91 -> 143 chunks). Judge picker: a judge whose GGUF is gone is disabled with "file missing on disk". |
| Bugs found live and fixed | Export gate approved 415 unreviewed pairs BEFORE answering 409 (declining "export anyway" kept them approved; the forced retry said "approved only"). SPA pushed the new URL AFTER re-running page scripts, so `/compare?group=` and `/testing?run=` reached through an in-app link ignored the query. A deleted judge GGUF failed with a bare path. "not retrieved" shown for quizzes that name no source. Queued comparison model shown as "answering 0/26". |
| Qwen3.5-9B tested with RAG (Genor 2026-10-09 evening; run `a0260f53` kept) | Qwen3.5-9B (HF, 4-bit) reader, new row chunks, top-k 10: judged by Gemma 12B **97/102** + 20/20 declined (95.9%); judged by a temporary Qwen3.5-9B Q4_K_M **92/102** + 20/20 (91.8%). Judges agree on 116/122; of the 6 splits, Gemma was wrong twice (e039, e044: "I don't know" judged pass — e039's reasoning is a raw JSON blob, verdict taken as pass although its checklist says missing) and Qwen wrong twice (e038, e095: exact key in the answer judged fail); e032/e043 are ambiguous. Corrected: ~95/102. The Qwen GGUF + provider were deleted again (5.8 GB). |
| Chunker measurement | Gemma 12B reader, top-k 10, judge Gemma: old word chunks 92/102 quiz + 20/20 declined (4 not retrieved, 6 reader misses); row-preserving token chunks 91/102 + 20/20 (4 not retrieved, 7 reader misses; key values retrieved 94/102). Same within judge noise; the two row-mixing cases moved: e097 fail -> pass, e077 half -> partial. |
| Not done | Stronger-judge comparison: the Qwen3-30B-A3B GGUF behind provider `local-qwen30b-a3b` is gone from disk and the API helper key has no funds (`402`), so no second judge is available; `rag_suite` gold-rank trace (`gold_rank_wide`) is unit-tested only (the Korvane quiz names no `source_id`). |
| Service | genorbox1 :7860 on `main`. Helper seat = local Gemma 4 12B. 2026-10-10: GPU mask now comes ONLY from the Settings Compute device choice (RTX 3090 saved); `gpu-mask.conf` drop-in removed, stale `CUDA_VISIBLE_DEVICES`/`FTS_MODEL_DIRS*` cleared from the systemd user manager; live `GET /api/system/compute-device` = `source: saved`, no env overrides, GTX 1070 masked. `accel.env.vendor_tool` finds `nvidia-smi` off PATH. Architecture as built + supervisor idea: `docs/TOPOLOGY.md`. |
| Corpus | `tests/corpus/korvane` (19 core files, 1,130 facts), quiz `eval/korvane_quiz_core.jsonl` (102 + 20 unanswerable); judge gold `eval/judge_gold.json` (50 cases). |
| Kept on purpose | project `korvane-ragtrace` (id e9f951f8: files, rebuilt index, quizzes; its 415 extractive fill pairs were reset to pending and this pass's runs/dataset deleted); `data/benchmarks/hf_cache` (228 MB). |

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

1. **Second judge:** re-download a strong local judge (Qwen3-30B-A3B GGUF, or fund the API key), run `scripts/judge_eval.py --provider <id>` on the 50 gold cases, then "Re-judge all" one Korvane RAG run and compare against Gemma; keep the better as default judge.
2. **RAG reranker:** decide RRF vote vs multilingual cross-encoder (ms-marco MiniLM is English-only; hybrid + rerank recall@5 = 90 vs BM25 96 on the old chunks); measure with `scripts/rag_reader_compare.py --pid e9f951f8 --reader gemma12=<gguf> --top-k 10` (baseline above).
3. **Quiz provenance:** add `source_id` to the Korvane quiz cases so `retrieval_hits` / `misses_found_wider` / `gold_rank_wide` report on the live runs, not only in unit tests.
4. **Abstain pair builder** (`data/prep/preference.py`): 112/150 abstain questions were answerable from other files; check each against the whole project with RAG. DPO gave no abstention.
5. **Paraphrase augmentation** (`scripts/corpus_paraphrase.py`, funded API key or ~2 h local): 3 re-worded questions per pair, train, compare on the Testing page.
6. Evaluate the adapter on the 4-bit base without merging (bf16 merge beat q4_k_m by 8 points in run 1). Merged-bf16 test needs > 40 min for 122 questions: default to the GGUF.
7. Follow-ups: audit MMLU/GSM8K/HellaSwag against official protocols; the Dataset-check run writes `full-ingested-corpus*.json` suites that then show up in the quiz pickers (two entries) — decide whether they belong there.

## Commands

- Whole UI-only flow: `tests/E2E_MANUAL_GUIDE.md` "Current standard"; driver `FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_track_a.py --list` (phases now run -> judge -> review).
- Judge a saved report offline: `.venv/bin/python scripts/rejudge_reports.py <report.json> --provider <id>`; CLI `fts suite MODEL SUITE --judge [PROVIDER] --out FILE`.
- Review tooling: `scripts/corpus_review.py plan|dump|apply|add|status [--ui]`; gates `corpus_parse_check.py`, `corpus_coverage.py --status approved`.
- Focused tests: `.venv/bin/python -m pytest tests/<file>.py -q -p no:cacheprovider`; full suite = 5 chunks (AGENTS gotcha; chunk 2 and 3 take ~8-10 min under load: run them with nohup); lint `.venv/bin/ruff check src/ scripts/`.
- UI check without a GPU: `.tmp/sbx/run_sandbox.py <port> <dir>` (fake engine + fake judge, redirected HOME, temp cwd).

## Known issues

- spa.js re-entry of full-loaded pages falls back to a full reload (AGENTS gotcha); it would cut an in-flight Guide stream.
- Legacy runs (before 2026-10-09) show "old matcher" verdicts; judge them again from the Testing page.
- llama-cpp-python 0.3.36 aborted the whole server (CUDA illegal memory access, SIGABRT, systemd restarted it, the judge job was reconciled as failed at 80/122) while a Qwen3.5-9B Q4_K_M judged with the auto 262k window + q8 KV + flash-attn; the same GGUF judged all 122 fine at `n_ctx` 32768, f16 KV, flash-attn off. A native abort in the judge is not caught in-process.
- Pairs approved by an earlier opt-in export stay approved (`corpus_review.py status` shows them); the option only adds, never un-approves.
- The 2026-10-08 `heldout` / `memorization` reports in `results/2026-10-08-ui/` carry no answer keys, so they could not be re-judged.
