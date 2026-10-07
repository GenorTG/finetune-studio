# Handoff — finetune-studio

## Mission

Local fine-tune, data-prep and RAG WebUI. GPU-first; never silently fall back.
Commits, pushes, local service restarts and real-hardware tests are authorized.

## State (verified 2026-10-07)

| Area | State |
|---|---|
| Git | `main` = three preference/guide lanes + `054c3e9` 4-bit fix + helper seat (API provider) + `feat/async-jobs` merge. Branch tips still on origin; worktree `../finetune-studio-wt/async-jobs` may exist (remove after confirming merged). |
| Tests | Full suite on merged tree, 5 chunks: all pass except one (`test_gguf_layers::test_manager_translates_legacy_99`, fixed afterwards, file re-run green); ruff + codemap clean. |
| Service | genorbox1 :7860 on the merged build; helper seat = **local** (API row configured: OpenCode Go, `deepseek-v4-flash`, effort `none`; key stored in `~/.finetune-studio/fts.db`, never in the repo). |
| Walkthrough | All phases re-run green on the merged build; the gguf phase now waits for export jobs to reach `done`. |
| Async exports | Real GGUF conversion (f16+q8_0, 15 s) with 73 concurrent `/api/projects` + export-page requests: slowest 0.01 s. Export page shows live phase/elapsed/progress + Cancel. Sandbox before/after: 19.98 s freeze → 0.07 s. |
| Cleanup | ~4.6 GB of old `.tmp` experiments, stray DBs/logs, stale test project dirs under `~/.finetune-studio/projects`, `data/aethermoor` deleted; throwaway project + base model removed after tests. |

## API helper — what was built and measured

Settings → **Helper model**: choose the local GGUF or an API provider (preset OpenCode Go / OpenAI / OpenRouter / custom; Fetch model list, Test connection, reasoning effort; key write-only). The helper does mining, suite generation, preference pairs and the Guide ("Guide" = the in-app panel; the **helper** is the model behind it). The document text goes to the provider when the API seat is active (warned on the card).
Same 14-document corpus, same RTX 3090 box, fresh project per row (`scripts/helper_bench.py`, `tests/e2e_user_walkthrough.py`):

| Helper | Mining 14 docs (12/12 facts in all) | Guide 5 questions (median / total) | 20 preference pairs |
|---|---|---|---|
| Local Gemma-4 12B Q4 | 111 s, 35 pairs | 2.3 s / 19 s (terse; answers often just the readiness line) | 45 s |
| deepseek-v4-flash, effort none | **30 s**, 37 pairs | 5.3 s / 30 s with native tools (grounded, notices real state) | 92 s (17 "refusal as rejected" drops) |
| deepseek-v4.1-flash, none | 30 s, 42 pairs | 8.2 s / 40 s (grounded) | 94 s |
| mimo-v2.5, none | 70 s, 40 pairs | 12.3 s / 54 s (best explanations) | 251 s |
| deepseek-v4-flash, default thinking | 70 s, 1 source got no pair | 6.6 s / 31 s | **failed** (empty answers 3×) |

Verdict: for mining the API is 3.7× faster at equal fact coverage; local wins on short sequential calls (pairs); deepseek with `reasoning_effort: none` is the cheap pick, mimo-v2.5 only if answer prose matters. Before native tool declaration, deepseek-none returned raw `<｜｜DSML｜｜` markup as 3 of 5 Guide answers — fixed in `OpenAICompatProvider`.

## Next steps

1. Concurrency for API helpers: `ModelManager._invoke_lock` serialises every call, so mining/pair authoring pay full network latency per call; allow N parallel requests when the seat is remote (pairs 92 s → ~15 s likely).
2. Guide quality: it still recommends DPO for a project with approved pairs and no SFT run; answers quoting only the readiness line are terse. Tune `guide/prompt.py` / `guide/kb/training*.md`.
3. Preference quality at scale: run `build-preference` + DPO on the SFT-merged run and measure (`scripts/pref_quality_eval.py`); add a length-ratio warning for chosen ≪ rejected (observed 0.13).
4. Guide helper cold-load (~80 s local) shows no progress until the stream starts — confirm the "loading helper" status renders at once.
5. Diagnose the official GSM8K UI run with bounded samples; decide whether `main` protection should require `ci-ok`; fan-dragon deploy stays deferred.

## Commands

- Walkthrough: `FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_user_walkthrough.py --list` (create upload prep review rag export model train gguf chat test bench cleanup); manual guide: `tests/E2E_MANUAL_GUIDE.md`.
- API helper UI E2E: `set -a; . ~/.openclaw/workspace/.env_opcgo; set +a; FTS_ALLOW_LIVE_E2E=1 FTS_E2E_API_KEY="$OPENCODE_GO_API_KEY" .venv/bin/python tests/e2e_helper_settings.py --model deepseek-v4-flash [--effort none]` (`--local` seats the GGUF again).
- Benchmark current seat: `.venv/bin/python scripts/helper_bench.py --label <name>` (needs a project with approved pairs).
- Focused tests: `.venv/bin/python -m pytest tests/<file>.py -q -p no:cacheprovider`; full suite = 5 parallel chunks (AGENTS gotcha), ~18 min; lint `.venv/bin/ruff check src/ scripts/`.

## Known issues

- Preference tuning evidence is Qwen3-0.6B at tiny n (DPO + keep-chosen shifts behaviour but costs facts; KTO did not help): treat as opt-in. ORPO not wired.
- Stale, not from this work: `.tmp/db-backup/` (8 MB) kept on purpose; `data/benchmarks/hf_cache` (228 MB benchmark dataset cache) kept.
- spa.js re-entry of full-loaded pages falls back to a full reload (AGENTS gotcha); it would cut an in-flight Guide stream.
