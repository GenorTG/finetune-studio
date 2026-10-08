# Handoff — finetune-studio

## Mission

Local fine-tune, data-prep and RAG WebUI. GPU-first; never silently fall back.
Commits, pushes, local service restarts and real-hardware tests are authorized.

## State (verified 2026-10-08)

| Area | State |
|---|---|
| Git | `main` pushed: table-header carry fix, Qwen3.5 GGUF `--no-mtp` fix, review/eval tooling, Track A/B browser phases, archived results. |
| Tests | Full suite on the final code, 5 chunks: 2,474 passed; the one failure (`test_repo_hygiene`, new Korvane fixtures not allow-listed) fixed and re-run green; `ruff check src/ scripts/` (the CI scope) clean. |
| Service | genorbox1 :7860; helper seat = **local** Gemma 4 12B. API row `api-helper` exists, but OpenCode Go returned `402 Insufficient account funds` on 2026-10-08: API paraphrasing/benchmarks are blocked until it is topped up. |
| Corpus | `tests/corpus/korvane` (19 core files, 1,130 facts), reusable for every future test. Golden reviewed pairs `golden/approved_pairs_core_2026-10-08.jsonl.gz` (2,491); what the miner got wrong `golden/rejected_with_reasons_2026-10-08.jsonl.gz` (1,964). |
| Last full run | Track A + B, Qwen3.5-9B, fresh DB, browser-led, every pair read by hand: `tests/corpus/korvane/RESULTS.md`. Project, runs and exports deleted via the UI afterwards; base model kept. |

## The result that matters

| Model | Paraphrase recall (102 Q) | Abstains on unknown (20 Q) |
|---|---|---|
| base, untrained | 2 % | 3/20 |
| SFT 6 epochs, q4_k_m / merged bf16 | 17.6 % / 25.5 % | 0/20 |
| DPO on top (188 reviewed pairs) | 15.7 % | 0/20 |
| **base + RAG (Track B)** | **75.5 %** (retrieval@5 90 %) | **20/20** |

SFT on ~2.4 reviewed pairs per fact teaches style, not paraphrase-robust facts, and removes the model's ability to say "not in the context". Use RAG for facts. The gate for
knowledge training is paraphrase recall (`scripts/corpus_eval.py`), not eval loss (its minimum sits at epoch ~2 while recall keeps rising).

## Next steps (in order)

1. **Export gate** auto-approves extractive `coverage_fill` pairs for chunks with no approved pair, at every export, so unreviewed pairs reach training. Make it opt-in or leave them pending; add a test
   (`data/prep/dataset_build.coverage_gate`, `coverage_fill.fill_all_project_gaps`).
2. **Abstain pair builder** (`data/prep/preference.py`): 112/150 abstain questions were answerable from other files. Check each candidate with a RAG search over the whole project
   and drop answerable ones; same for "hallucination" pairs whose rejected answer is a generic non-answer.
3. **Paraphrase-augmentation experiment** (`scripts/corpus_paraphrase.py`; needs a funded API key or ~2 h of local helper): 3 re-worded questions per reviewed pair, answers untouched,
   train, compare recall. If recall rises materially, add "Paraphrase questions" to the pairs page.
4. Split "under-training" from "QLoRA-merge loss": evaluate the adapter on the 4-bit base without merging (merged bf16 beat q4_k_m by 8 points).
5. One whole-service CUDA abort (`illegal memory access`) while building preference pairs with the helper at `n_ctx 16384`; not reproduced at 32768. Look at `models/gguf_fit.py` n_ctx/ubatch handling.
6. Guide still pushes "SFT or DPO" for a project that already has runs; tune `guide/kb/training*.md`. Backlog: delegate/background audit of every long operation, extended-tier runs,
   API-helper benchmark rerun, concurrency for API helpers (`ModelManager._invoke_lock`), fan-dragon deploy (deferred), `main` protection vs `ci-ok`.

## Commands

- Track A/B: `tests/E2E_MANUAL_GUIDE.md` ("Current standard") lists every phase; driver `FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_track_a.py --list`.
  Review tooling: `scripts/corpus_review.py dump|apply|add|status`; gates `corpus_parse_check.py`, `corpus_coverage.py --status approved`, `corpus_eval.py`, `corpus_eval_rag.py --pid <p>`.
- Old synthetic walkthrough: `FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_user_walkthrough.py --list`.
- API helper UI E2E: `set -a; . ~/.openclaw/workspace/.env_opcgo; set +a; FTS_ALLOW_LIVE_E2E=1 FTS_E2E_API_KEY="$OPENCODE_GO_API_KEY" .venv/bin/python tests/e2e_helper_settings.py --model deepseek-v4-flash` (`--local` re-seats the GGUF).
- Focused tests: `.venv/bin/python -m pytest tests/<file>.py -q -p no:cacheprovider`; full suite = 5 parallel chunks (AGENTS gotcha), ~18 min; lint `.venv/bin/ruff check src/ scripts/`.

## API helper (measured 2026-10-07, 14-doc corpus)

deepseek-v4-flash with `reasoning_effort: none` mines 3.7x faster than the local helper at equal fact coverage (30 s vs 111 s); local wins on short sequential calls (preference pairs);
mimo-v2.5 only when answer prose matters; deepseek with default thinking returned empty answers for pairs. Settings → Helper model; the document text goes to the provider while the API seat is active.

## Known issues

- DPO evidence is thin (Qwen3-0.6B and Qwen3.5-9B, tiny n): it shifts style, did not add abstention or recall here. ORPO not wired.
- spa.js re-entry of full-loaded pages falls back to a full reload (AGENTS gotcha); it would cut an in-flight Guide stream.
- Kept on purpose: `data/benchmarks/hf_cache` (228 MB).
