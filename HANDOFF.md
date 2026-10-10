# Handoff — finetune-studio

## Mission

Local fine-tune, data-prep and RAG WebUI. GPU-first; never silently fall back.
Now being re-architected around a small supervisor that owns every component (`docs/TOPOLOGY.md`). Commits, pushes, local service restarts and real-hardware tests are authorized.

## State (verified 2026-10-10)

| Area | State |
|---|---|
| Git | `main` = supervisor P1 (`ad04dd3` + env fix + this handoff commit), on top of the run-then-judge rebuild. |
| Tests | Full suite green in 5 chunks on the merged code (458 + 488 + 651 + 537 + 597 passed, 1 skipped); `ruff check src/ scripts/` clean. |
| Supervisor P1 (live on genorbox1) | systemd runs `python -m finetune_studio.supervisor`: it owns :7860 and hands the fd to the web child (`uvicorn --fd`), probes `/api/health`, restarts with backoff (1..30 s, FAILED after 8 early deaths), logs events (`run/events.jsonl`), serves a control API on `$FTS_ROOT/run/supervisor.sock`. CLI: `fts status\|start\|stop\|restart\|logs\|events\|up\|doctor`. Live proof: `kill -SEGV` the web child, a request sent during the crash answered 200 after 2.9 s, event log `exited SIGSEGV -> backoff -> spawned -> ready`; `systemctl --user restart` leaves no orphan. Compute device after restart: `source: saved`, `env_overrides: []`, 3090 active, 1070 masked. Old unit saved in `.tmp/unit-backup/`. |
| Genor's decisions (2026-10-10) | 1) supervisor first; 2) ONE inference worker, one model resident, heavy GPU runs (training) evict it first (today's `unload_all_models` at `routes/training.py:~695` becomes a supervisor lease rule); 3) after a crash: report, never auto-reload (an auto-recovery system may come later). |
| fan-dragon | Not yet switched: `git pull` + `update.sh` works (standalone web, `managed: false`), then run `bash install-service.sh` once on that host to get the supervisor unit (`update.sh` warns until then). |
| Run-then-judge (2026-10-09) | A test RUN saves raw transcripts; judging is a separate step (AI judge on any provider row, or the human on the Testing page). `docs/judging/RUN-THEN-JUDGE.md`; code `testing/{judge,judging,run_store,scoring,audit}.py`, `db/judgements.py`, `webui/testing_jobs.py`. Judge prompt v4 = 48/50 on `eval/judge_gold.json` (small, partly tuned: a sanity check). |
| Result that matters (Qwen3.5-9B, Korvane, 102 + 20 unanswerable) | SFT run 2 q4_k_m: pass 24 (+20 partial), unanswerable 0/20. Base + RAG: 76-97 pass, 20/20 declined; best Qwen3.5-9B top-20 = 97/102. Tuned + RAG: 86/102, 2/20 declined. Use RAG for facts; SFT teaches the training questions only. Retrieval is the remaining RAG loss (11 questions never retrieved). |
| RAG judge comparison | Qwen3.5-9B RAG run `a0260f53`: Gemma 12B judge 97/102 + 20/20, Qwen3.5-9B judge 92/102 + 20/20; judges agree on 116/122, corrected ~95/102. No second strong judge available yet (Qwen3-30B-A3B GGUF gone, API helper key `402`). |
| Service config | Helper seat = local Gemma 4 12B. GPU mask comes ONLY from Settings -> Compute device (RTX 3090 saved); only unit drop-in left is `path.conf`. `accel.env.vendor_tool` finds `nvidia-smi` off PATH. |
| Kept on purpose | project `korvane-ragtrace` (id e9f951f8); `data/benchmarks/hf_cache` (228 MB); corpus `tests/corpus/korvane`, quiz `eval/korvane_quiz_core.jsonl`. |

## In flight

Nothing running. Next architectural step is supervisor P2 (below); no code for it exists yet.

## Next steps (in order)

1. **Supervisor P2 — inference service worker** (`docs/TOPOLOGY.md` §7): move the resident `InferenceEngine` into a supervised child (one model max) behind a `RemoteEngine` with the same attributes (`model`, `model_path`, `n_ctx`, `offload`, `is_gguf`, `vision`, `_busy`, `_last_used`, `load`, `unload`, `generate`) plus a `chat_completion` op for `webui/routes/data_prep_chat.py:380` (the only caller that touches `engine.model` directly). Crash = typed `EngineCrashed`, real 5xx, model left unloaded and reported. Replace `ENGINE_LOCK` with a supervisor GPU lease; training start asks the supervisor to evict the model and refuses if VRAM is not free. Test: fake-transport unit tests, real-subprocess SIGSEGV test, live 3090 run (load, generate, kill worker, UI survives, reload works, `nvidia-smi` shows VRAM freed).
2. Supervisor P3: training worker and GGUF export/convert as supervised children (a web restart must stop killing training); P4: `fts test|train|suite|compare|benchmark|rag-test` go through the supervisor, API gateway child for external providers.
3. Small P1 follow-ups: a Settings/Models card for `GET /api/system/supervisor`; show "restart" instead of `SIGTERM` as last exit for operator restarts in `fts status`.
4. **Second judge:** re-download a strong local judge (Qwen3-30B-A3B GGUF, or fund the API key), `scripts/judge_eval.py --provider <id>`, "Re-judge all" one Korvane RAG run, keep the better as default.
5. **RAG reranker:** RRF vote vs multilingual cross-encoder (ms-marco MiniLM is English-only); measure with `scripts/rag_reader_compare.py --pid e9f951f8 --reader gemma12=<gguf> --top-k 10`.
6. **Quiz provenance + abstain pairs:** `source_id` on the Korvane quiz cases; abstain pair builder must check each question against the whole corpus (112/150 were answerable elsewhere).
7. Paraphrase augmentation (`scripts/corpus_paraphrase.py`), evaluate the adapter on the 4-bit base without merging, audit MMLU/GSM8K/HellaSwag protocols.

## Known issues

- Until P2, a llama.cpp CUDA abort still takes the web process down (now restarted in ~3 s with the signal in `fts events`; in-flight jobs and the resident model are lost). Qwen3.5 GGUF at the auto 262k window + q8 KV + flash-attn aborts on llama-cpp-python 0.3.36; load it with `n_ctx` 32768, f16 KV, flash-attn off.
- A web restart (crash or `fts restart web`) kills a running training worker (same process group, as before the supervisor).
- `fts status`/`fts doctor` import the package, which runs the GPU policy (one `nvidia-smi` call); harmless but not free.
- spa.js re-entry of full-loaded pages falls back to a full reload; it would cut an in-flight Guide stream.
- Legacy runs (before 2026-10-09) show "old matcher" verdicts; judge them again from the Testing page.
- Pairs approved by an earlier opt-in export stay approved (`corpus_review.py status`); the 2026-10-08 `heldout`/`memorization` reports carry no answer keys.

## Commands

- Service: `fts status`, `fts restart web`, `fts logs web -n 100`, `fts events -f`, `fts doctor`, `fts up`; whole tree `systemctl --user restart finetune-studio`; reinstall unit `bash install-service.sh`.
- Tests: `.venv/bin/python -m pytest tests/test_supervisor_core.py tests/test_supervisor_control.py tests/test_supervisor_env.py tests/test_supervisor_web_routes.py tests/test_supervisor_unit_contract.py -q -p no:cacheprovider`; full suite = 5 chunks (AGENTS gotcha), run with nohup; lint `.venv/bin/ruff check src/ scripts/`.
- UI flow: `tests/E2E_MANUAL_GUIDE.md` "Current standard"; `FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_track_a.py --list`. Judge offline: `scripts/rejudge_reports.py <report.json> --provider <id>`; CLI `fts suite MODEL SUITE --judge [PROVIDER] --out FILE`.
- UI check without a GPU: `.tmp/sbx/run_sandbox.py <port> <dir>` (fake engine + fake judge, redirected HOME, temp cwd). A supervisor sandbox needs temp cwd, redirected `HOME`, `FTS_ROOT`, `FTS_DB`.

## Blockers

None. Open question for Genor only if he wants it changed: the supervisor is the systemd unit's single process (systemd -> supervisor -> children) with HTTP over a unix socket for control, per `docs/TOPOLOGY.md` §7.
