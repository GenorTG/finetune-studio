# AGENTS.md — finetune-studio

Local fine-tune + data-prep + RAG WebUI (FastAPI + Jinja, Python 3.12, uv venv `.venv/`), CLI `fts`.
Hosts: edit/commit/push on **genorbox1** (`/home/genorbox1/work/finetune-studio`, RTX 3090 + GTX 1070);
**fan-dragon** (`/home/genortg/finetune-studio`) is deploy/test only — pull there, never SSH-edit files.
Both run the `finetune-studio` systemd **user** unit on :7860.

## Read first
1. `HANDOFF.md` — current state, in-flight work, next steps (≤120 lines; longer = stale, rewrite it).
2. `docs/WORKPLAN.md` — iron rules (binding) + ordered plan (last updated 2026-10-02; HANDOFF has newer state).
3. `docs/PRODUCT-BRIEF.md` — north star / quality bar. `docs/GOTCHAS.md` — full learned-rule log.
4. On demand: `docs/README.md` (doc map), `docs/CODEMAP.md` (symbol map), `RESTART.md` (service ops).

## Commands
- Setup/repair: `bash install.sh` (`--plan` preview GPU plan, `--check`, `--repair`, `--cpu`, `--gpu <vendor>`); dev extras: `uv pip install --python .venv/bin/python -e '.[dev]'`.
- Tests: `.venv/bin/python -m pytest -q -p no:cacheprovider --ignore=tests/test_vram.py` (full suite ~21 min); one file: `.venv/bin/python -m pytest tests/test_x.py -q`. `make test` = verbose full run incl. GPU tests.
- Lint: `.venv/bin/ruff check src/ scripts/` (must stay clean; `tests/` has ~108 legacy findings). `make lint` hides failures (`|| true`) — never trust it.
- Run: `make run` (= `bash run.sh`, uvicorn :7860). Prefer the service: `systemctl --user restart finetune-studio`; logs `journalctl --user -u finetune-studio -f`.
- Accelerator check: `.venv/bin/fts accel` (nonzero only on a GPU host that fell back to CPU).
- Symbols: `make codemap` / `.venv/bin/python scripts/codemap.py --grep NAME`; `make codemap-check` fails on drift.
- E2E browser suite: `tests/run_qa.sh` (see `tests/README_E2E.md`; can mutate live data).
- Deploy: `git push`, then on the host `bash update.sh` (non-interactive: pull ff-only, venv health, deps + torch pin, llama.cpp, DB migrate, restart). (Re)install unit: `bash install-service.sh`. Details: `RESTART.md`, `docs/DEPLOYMENT.md`.
- Verify before "done": the changed module's test file + ruff pass; GPU-path changes need a real GPU run noted in HANDOFF.

## Layout
- `src/finetune_studio/` — `webui/` (app, routes, templates, static), `cli/` (`fts` commands), `training/` (engine, worker, vram, export), `models/` (loaders, providers, registry), `data/` (parsers, prep, fs, `rag_portable/`), `rag/`, `testing/` (inference, judge), `benchmarks/`, `compare/`, `db/`, `accel/` (vendor-neutral GPU layer), `naming.py`, `config.py`.
- `tests/` — pytest, one file per concern; `e2e_*.py` + `run_qa.sh` are live/browser suites.
- `scripts/` — `accel_plan.py` (GPU detect/plan/install), `install_diagnose.py`, `codemap.py`, `git-hooks/`, dataset tools.
- `install*.{sh,ps1,bat,fish,zsh}`, `run*.{sh,bat,fish,zsh}`, `update.sh`, `install-service.sh` — keep shell variants in sync when changing one.
- Runtime (never committed): `data/`, `datasets/`, `models/`, `media/`, `output/`, `projects/`, `.llama.cpp/`, `.tmp/`; app state under `$FTS_ROOT` (default `~/.finetune-studio`).

## Conventions
- New code goes in the module that already owns the concern (check CODEMAP); new feature → new module + new test file, not a bigger file. Run `make codemap` after adding/moving/renaming symbols.
- Type hints on every signature; Pydantic/dataclass models for anything crossing the API boundary.
- HTTP: reuse what the module already uses (`urllib.request`/`httpx`); do not add a new HTTP library.
- Device code goes through `finetune_studio.accel` — never hard-code `cuda`, `{"": 0}`, `bf16=True` or bitsandbytes.
- Errors are honest: real 4xx/5xx with a reason, never 200 `{error}` or a silent fallback.
- Commits: imperative "what + why", one logical change each; the pre-commit hook bumps `VERSION` (run `make hooks` once per clone; skip with `FTS_NO_BUMP=1`).

## Session protocol
- Workspace `AGENTS.md` owns work modes / spawn rules. Here: goal + card, verify with Commands above.
- Cursor only if Genor asks: `ensure-cursor-helper.mjs --cwd /home/genorbox1/work/finetune-studio --label cursor-helper:finetune-studio`.
- End: rewrite `HANDOFF.md`, commit and push (Genor 2026-10-05: commits/pushes to origin are allowed); push only after the touched tests + ruff pass, then check `gh run list --limit 5` for failed Actions.
- HANDOFF sections, in order: `Mission` (2 lines), `State (verified <date>)` table, `In flight`, `Next steps` (≤7, exact commands), `Known issues`, `Commands`, `Blockers`. ≤120 lines; rewrite, never append; archive old copy to `docs/archive/HANDOFF-<date>.md`.

## Gotchas
<!-- One line each. Full context + history: docs/GOTCHAS.md — append there too. -->
- CI shards are packed by `WEIGHTS` in `scripts/ci_shard.py` (one file, `test_rag_audit`, is ~70 s): when a shard's wall time in the job summary drifts, re-measure with `--durations=0` and edit the weights; actions stay SHA-pinned (`# vX.Y.Z` comment), Dependabot bumps them.
- Legacy `.doc` fallback calls `olefile`; keep it in the `parsers` extra, not only in dev/test environments (`tests/test_doc_parser_olefile.py`).
- Dev docs are local-only: `docs/{ARCHITECTURE,CODEMAP,DEVELOPER,GOTCHAS,PRODUCT-BRIEF,README,REFACTOR-SPEC,WORKPLAN}.md` + `docs/{modules,audit,archive,judging}/` are gitignored — never `git add -f`, never link from README/Pages/INSTALL/DEPLOYMENT/TUTORIAL (`tests/test_repo_hygiene.py`).
- Never commit runtime data/corpora (`data/`, `datasets/`, `models/`, `media/`, `.venv`) — copy to fan-dragon by rsync; `.gitignore` re-includes need the three-step form (`data/benchmarks/default.json`).
- Install only via `install.sh`/`update.sh` (uv venv has no pip module; never `python -m pip`); every resolve passes `-c .venv/torch-constraints.txt` or torch silently drifts to another CUDA build.
- GPU stack choice lives only in `scripts/accel_plan.py` (tests forbid hard-coded `cuNNN`/`rocmX.Y` in installers); fake hardware with `FTS_ACCEL_FIXTURE=<json>`.
- abetlen cu13x llama-cpp wheels carry SASS for sm_75+ only (cu125: sm_60+): `ABETLEN_CUDA_TAGS` floors in `accel_plan.py` keep Pascal/Volta off them; nvcc 13 cannot target sm_<75 and Blackwell needs nvcc>=12.8 (`nvcc_problem`), `FTS_GPU_*` numeric tokens are indices, never name substrings.
- uv: GPU llama-cpp wheel needs `--extra-index-url … --index-strategy first-index` (`--index-url` lets PyPI's CPU sdist win); source builds need `--no-cache`.
- Unsloth pins torch<2.13/transformers<=5.5: default `FTS_UNSLOTH=auto` skips it; `FTS_UNSLOTH=1` builds an older torch stack. torchaudio is intentionally not installed.
- genorbox1 GPU 1 is a GTX 1070: never use it — mask with `CUDA_VISIBLE_DEVICES`/`FTS_GPU_EXCLUDE="GTX 1070"` (the service unit does not mask it).
- Never kill foreign GPU/RAM users (games, ComfyUI, other services); no headroom → pause and report. Unload models/helpers you loaded.
- Sandbox instances: temp cwd + redirected `HOME`, `FTS_ROOT`, `FTS_DB` (main DB `data/finetune_studio.db` is cwd-relative); `FTS_ROOT` alone hits live data.
- Test screenshots go only to `.tmp/qa-shots/` (`FTS_QA_SHOTS`), never into docs/README/Pages/media; live E2E needs `FTS_ALLOW_LIVE_E2E=1`.
- Data-prep paths must go through `data.fs.paths.resolve_in_project`/`resolve_within`; no raw `Path(user_str)` in a route (`tests/test_data_prep_path_fence.py`).
- RAG export is AES-256-GCM encrypted by default; passphrase only in POST bodies, never stored/logged; shipped server binds 127.0.0.1 unless a token is set (`tests/test_rag_encrypted_package.py`).
- Studio corpus export is the encrypted `.ftsrag` bundle only (`rag_portable/secure_bundle.py`); never wire plaintext `store.export_bundle` to a route.
- Every project template declares `{% block workspace %}model|rag{% endblock %}` (RAG pages also `workspace_nav`); tab label, `<h1>` and `breadcrumb_tab` use the same word.
- Page headers: flow step + plain-language purpose + next-step link; jargon only in a `text-xs` line below.
- CSS: use only tokens defined in `app.css :root`/light block; keep the global `[hidden]{display:none!important}`; bump `app.css?v=` in `base.html` on every CSS change, and update every test pin of it in the same commit (`grep -rn '?v=' tests/`).
- Visual QA: `window.__audit` probe in dark AND light via the playwright-core runner in `~/.openclaw/workspace/.tmp/qa-sweep/`; on stubborn narrow-width bugs grep every `@media` block for the selector.
- Name artifacts with `naming.display_for_path()`/`resolve_run_path()` (never dir basenames, never `"/output" in path`); DB access only via `finetune_studio.db`.
- Match library files ↔ QA sources by `sha256` (manifests use `files/<sha12>/…`); RAG `document_id` is md5(path).
- Wizard chain: step 2 loads the helper, aborts if nothing was mined, unloads it before training — never let mining errors pass silently.
- fan-dragon login shell is fish: wrap multi-part SSH commands in `bash -c '…'`.
- Ruff `F821` here = a real `NameError` on an error path; fix it as a bug.
- OpenClaw tools: `exec` takes `timeoutSeconds`, not `timeout`; verify via `curl http://<host>:7860` first, browser second (fall back to curl after one timeout); don't narrate memory/runtime meta.
- `preset_advisor.py` step floor ignores `validation_split` and assumes eff. batch 8 (`steps = (pair_count * epochs) // eff_batch`): its `optimizer_steps` run ~2.2× high vs `metrics_json.total_steps` — treat as an upper bound until fixed.
- Models outside `models/` + `output/` → add a `model_dirs_extra` entry, `POST /api/models/refresh`, then reload the training page (its model dropdown is server-rendered at load).
- Training API start takes `model_path` (not `base_model`) + `preset_id` (e.g. `"qlora"`); GPU pinning goes in a `finetune-studio.service.d/*.conf` drop-in, never `systemctl --user set-environment`.
- RAG chat on a small tuned model ignores retrieved context when its training rows had none (Qwen3-0.6B plain rows: 2/5 grounded, with context-grounded rows 4/5 — `.tmp/rag-grounding/RESULTS.md`); the export's `grounded_share` option (`data/prep/grounding.py`, default 40% when a RAG corpus exists) fixes it at the data level. The rag/chat prompt layout lives only in `data/rag_portable/prompt.py` (pinned by `tests/test_rag_chat_prompt.py` + `test_grounded_rows.py`); moving CONTEXT into the user turn did not help — never rewrite it unmeasured.
- RAG embedder/reranker live in `data/rag_portable/model_cache.py` (key = kind, name, resolved device; idle expiry `FTS_IDLE_TIMEOUT`). Anything that needs the GPU (training start, merge, export, model load, unload-all) must call `release_rag_models()` first; `store.get_embedder` / `query.get_reranker` stay patchable aliases of the cached getters — tests patch those names, never `model_cache` internals (conftest releases the cache after every test).
- Never run `ruff --select RUF100 --fix` on its own: with only RUF100 selected, every other rule's `# noqa` looks unused and ~180 get stripped repo-wide. Use `ruff check src/ scripts/` (project config) and fix RUF100 hits by hand.
- `color-mix()` tints under text fool the `window.__audit` contrast probe (reads the mixed colour as 0–255 → bogus 1.38:1); use `rgba()` tints and `--err-text`/`--warn-text` for text on light.
- Coverage-fill questions (`data/prep/coverage_question.py`) must be self-contained: distinctive subject + section/file scope, pronoun-led answers dropped; a chunk with no such question is reported `reason="no_specific_question"` (export gate blocks it), never given a vague pair (`tests/test_coverage_question.py`, 5-doc corpus in `tests/fixtures/coverage_docs/`).
- Tests must not write into the repo cwd: `db.datasets.datasets_dir` derives `<db dir>/projects/<pid>/datasets` from the ORIGINAL `settings` singleton (not the copy `temp_db` swaps into `cfg`), so conftest points that singleton's `db_path` at a private per-test temp dir in place; never rebind modules to a different settings object (path-fence/route tests mutate the shared one) (`tests/test_no_repo_data_leak.py`).
- E2E proof runs go through the LIVE :7860 instance (Genor 2026-10-06; restart the user unit first so it runs current code) with a throwaway project (`e2e-*`/`zz-*`), deleted via the API afterwards with weights/exports; sandbox instances (:7891/:7892) only for CPU/degraded/index variants. Back up `data/finetune_studio.db` to `.tmp/db-backup/` before cleaning it. Trust raw evidence (GPU0-only nvidia-smi CSVs, journal, JSON), not UI screenshots alone.
- Agent shells cap a foreground Bash call at 600 s and deny `run_in_background`: run the 22-min suite as 5 chunks (`ls tests/test_*.py | grep -v test_vram.py | awk -v k=N 'NR%5==k'`, ≤ ~5 min each; chunk 2/3 can need a 585 s timeout); no pytest-xdist installed. Never edit src/tests while a chunk is running.
- fan-dragon (RTX 5080, stub `.git`, node-only) deploy is deferred — Genor 2026-10-06: not now; don't restore its checkout unasked.
- Chosen GPU != index 0 (`CUDA_VISIBLE_DEVICES=1,0`, `FTS_DEVICE=cuda:1`): transformers' `TrainingArguments` hard-codes `cuda:0` + `n_gpu=device_count` (touches the other card, DataParallel), PEFT's `torch_device="cuda"` → safetensors reads onto `cuda:0`, llama.cpp default LAYER split spreads over every visible card and inits a context on each. The training child gets the card as its device 0 (`accel.isolated_env`), loaders call `accel.activate()`, `models.hf_loader.load_peft_adapter` pins the adapter device, `llama_gpu_kwargs` sets `split_mode=NONE`; only a visibility mask keeps a context off other cards in the server process. Live probe: `.tmp/lane-c/` (watchdog kills on first context on the GTX 1070).
- Dataset export (route `data-prep/export` + `fts dataset build`) shares `data/prep/dataset_build.py` (coverage gate, persist+register); it imports `fill_all_project_gaps` lazily so tests patch `data.prep.coverage_fill.fill_all_project_gaps`; `--out` is fenced by `resolve_in_project` (project dirs only) (`tests/test_cli_dataset_build.py`).
- A grounded row (retrieved CONTEXT in its system turn) teaches "answer from the context", not memorisation: live run, plain rows 15/15 recalled, grounded rows 1/9 asked bare, 9/9 with their own context. Suite cases therefore carry `system_prompt` (generate_suite → load_test_suite → run_suite) and the wizard reports "from memory" vs "with context" separately; never score grounded rows bare, and never read the plain-row number as generalisation (`tests/test_suite_grounded_context.py`).
- llama.cpp's CUDA path aborts the WHOLE service (`CUDA error: an illegal memory access`, SIGABRT) once a prompt fills a 512-token micro-batch on some Q8_0 models (Qwen3-0.6B Q8_0: 0/10 ok at ubatch 512, 10/10 at <=384, 42/42 at 256 up to 20k-token prompts; Q6_K/Q4_K_M and Gemma-12B Q4_K_M never failed). It is NOT memory: it reproduces at 2.5 of 24 GB and with `n_gpu_layers=0`. `load_llama_gguf` therefore caps `n_ubatch` at 256 (`FTS_LLAMA_UBATCH` overrides; ~8 % prompt speed on a 12B). Native bug in the installed source-built llama-cpp-python 0.3.36; the abetlen cu13x wheel is untested. Diagnose with the native log (`models/llama_native_log.py`), not the Python exception.
- GGUF loads never shrink `n_ctx` (agentic/tool use needs it): `models/gguf_fit.py` plans how many layers fit in free VRAM and `load_llama_gguf` steps `n_gpu_layers` down on OOM (6 attempts, the last always CPU-only with `offload_kqv=False, op_offload=False` — with 0 layers llama.cpp still allocates ~1 GiB compute buffers on the card). llama-cpp-python raises a bare `ValueError('Failed to load model from file')` for an OOM: the cause (`cudaMalloc failed: out of memory`) exists only in the native log, so the old string-matching retry never fired on real failures. Explicit `n_gpu_layers` is an upper bound; -1/99 = automatic. Verified live under real pressure (torch process holding 17/22 GiB): 12B at 32k ctx loads 14/48 layers, then CPU-only at 0.3 GiB free; the service never restarted.
- `pkill -f <pattern>` inside an agent Bash call matches the call's own command line and kills the shell (exit 144): find a sandbox pid with an anchored `pgrep -f '^/abs/python -m uvicorn.*--port N'` and `kill` it.
- Compute-device choice (Settings card, `$FTS_ROOT/compute_device.json`, `accel/saved_choice.py`) is applied only at process start by `accel.env.apply_device_policy`; explicit `FTS_GPU_*`/`FTS_DEVICE`/`*_VISIBLE_DEVICES` beat it, so the live unit's `FTS_GPU_EXCLUDE` drop-in shows "overridden" until removed; CPU-only must mask every card (llama.cpp offloads to any visible GPU) (`tests/test_compute_device_*.py`).
- Dataset uploads must preserve DPO `{prompt, chosen, rejected}` rows unchanged; generic SFT column conversion drops preference semantics and causes valid DPO uploads to fail (`data/converter.py`, `tests/test_dataset_upload_convert.py`).
- Dataset health checks must use the selected training route's formatter; the SFT-only checker marked valid DPO rows untrainable (`data/dataset_health.py`, `tests/test_dataset_health.py`).
- Parsed-source readiness uses `chunk_count > 0`, not `status == "ready"`: mining transitions source manifests to `generated`/`generated_incomplete` (`data/prep/runner.py`, `webui/routes/data_prep_chat.py`).
