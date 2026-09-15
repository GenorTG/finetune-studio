# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon pulls and runs `systemctl --user` unit `finetune-studio` on :7860.

## State (verified 2026-09-15 ~11:10 genorbox1)
| Area | Status |
|------|--------|
| Git | Uncommitted E2E-40 work on `main` (no commit per session rule) |
| E2E-40 Unsloth isolation | ✅ Training via `multiprocessing` **spawn** child (`training/worker.py`); uvicorn never imports unsloth |
| InferenceEngine | ✅ Plain transformers only; bf16 preferred; BitsAndBytes 4-bit on request/OOM; strips bare `</think>` |
| vram/profile.py | ✅ qlora path uses bitsandbytes+PEFT (no unsloth) |
| Tests | ✅ 25 passed: worker protocol, server-never-imports-unsloth, inference load, stop/merge/reconcile/monitor suite |
| Ruff | New modules clean for F821/F401; engine.py still has pre-existing I001 noise |

## Next steps
1. Commit + push E2E-40 when Genor asks (do not include other agents’ data-prep/app.js/chat files).
   `git status` then stage only training/inference/vram/tests for this change.
2. On fan-dragon: pull, `systemctl --user restart finetune-studio`, confirm cgroup.
   `ssh fan-dragon 'bash -lc "cd ~/finetune-studio && git pull --ff-only && systemctl --user restart finetune-studio"'`
3. GPU truth check: `/api/inference/chat` “capital of France” → Paris (not `</think>`); stock Qwen3-4B bf16; Unsloth-trained merge recall still works.
4. Start a short train + Stop during “Loading model…” — child must die, status `stopped`.
5. Confirm `python -c "import sys; …"` after hitting inference: `"unsloth" not in sys.modules` in the **server** process (journal / debug endpoint optional).

## Commands
- Related tests: `.venv/bin/python -m pytest tests/test_training_worker_protocol.py tests/test_server_never_imports_unsloth.py tests/test_inference_unsloth_load.py tests/test_training_no_fork_pool.py tests/test_training_stop_phases.py tests/test_training_merge_nonfatal.py tests/test_merge_base_resolution.py tests/test_reconcile_stale_runs.py tests/test_training_monitor_template.py -v`
- Lint touched: `.venv/bin/python -m ruff check src/finetune_studio/training/worker.py src/finetune_studio/testing/inference.py src/finetune_studio/training/vram/profile.py --select F821,F401`
- Deploy: push then fan-dragon pull + `systemctl --user restart finetune-studio`

## Blockers
None for code; GPU verification pending commit/deploy.

## Additional verified work (added 11:50, before OpenClaw gateway outage)

### benchmarks-page-fixes (Cursor child 2267498c, settled ok)

| File | Change |
|------|--------|
| `webui/routes/benchmarks.py` | Suite discovery only includes files that exist on disk (Path.is_file() + project auto_suites); suite validation happens BEFORE unload/load (returns 400/404 JSON without touching the engine); 409 if run isn't done+output_path; NEW POST /projects/{pid}/base/run for explicit base-model benchmark (never a silent fallback); _execute_benchmark wrapped in try/finally so the bench engine always unloads; JSON error statuses throughout |
| `webui/templates/benchmarks.html` | Honest subtitle; empty-state copies link to Open Training; RUN button disabled when run isn't done+output_path with error tooltip; model_label column rendered via new Jinja filter; explicit base-model probe row (separate form posting to /base/run); suite labels show auto · <name> (<n> cases) for auto-suites; JS error handling on runLatestBenchmark + runComparison (was silently swallowing errors before) |
| `webui/model_labels.py` (NEW, 45 lines) | HF cache path models--<org>--<repo>/snapshots/<hash> → <org>/<repo> short label; plain org/repo ids kept as-is; otherwise uses last path segment |
| `webui/routes/pages.py` | Registers model_label filter (filters live in pages.py, not app.py); new _discover_suites(pid) helper; hides __base_model__ placeholder runs from the visible table |
| `tests/test_bench_judge.py` | Updated to use status='done' (was 'completed', which never matched) |
| `tests/test_benchmarks_discovery.py` (NEW) | Suite discovery tests (file-exists filter, auto_suites prefix, missing-file 404) |
| `tests/test_benchmarks_run_validation.py` (NEW) | 409 when run isn't benchmarkable; suite validation order (before unload/load) |
| `tests/test_benchmarks_model_label.py` (NEW) | HF cache parsing edge cases; org/repo passthrough |
| node --check on inline JS | exit 0 |
| Ruff | Clean on touched files |
| Tests | 18 passed |

Not committed/pushed — staged in working tree on main HEAD 713aec7.

## OpenClaw gateway outage (started ~11:30)

All gateway-dependent tools failing. Confirmed by trying 17+ exec invocations (different shapes), 2 browser, 1 openclaw, 2 sessions_spawn with visible=true:

| Tool | Error envelope |
|------|---------------|
| exec (default host=auto) | service child cleanup identity lost |
| exec (host=node, node=fan-dragon) | gateway rejected websocket upgrade (HTTP 503): Gateway websocket admission closed |
| openclaw | OpenClaw could not reach working inference. Run openclaw onboard on the machine running OpenClaw to reconnect |
| browser | timed out. Restart the OpenClaw gateway (OpenClaw.app menubar, or openclaw gateway)... |
| sessions_spawn (ACP, visible=true) | Invalid arguments for tool openclaw:core:sessions_spawn: visible: must be boolean (schema rejects even though catalog lists it as boolean) |

Working tools (used to write this HANDOFF entry): process, read, write, apply_patch, tool_call, tool_search. No zombie exec sessions — process(action=list) returns No running or recent sessions.

### Path forward (3 options)
1. Restart the OpenClaw gateway on the gateway host (openclaw gateway from OpenClaw.app menubar). This is admin work, not app-interfacing — falls in Genor's CLI allowed bucket. Unblocks all 4 broken tools. I then resume: python3 /tmp/children_deploy.py on genorbox1 → ships the combined work to fan-dragon → I drive the human-driven E2E via the browser tool exclusively.
2. Run the deploy yourself (one command from /home/genorbox1/work/finetune-studio): python3 /tmp/children_deploy.py. That ships the combined work to fan-dragon + restarts the systemd service. You then drive the E2E via the browser yourself, or I drive it via the browser when the gateway recovers.
3. Wait for next session (when the gateway should be healthy). All work preserved on disk: source material at /home/genorbox1/.openclaw/workspace/.tmp/velmaris_worldbuilding.md, deploy script at /tmp/children_deploy.py, both children's code in the working tree on main HEAD 713aec7.
