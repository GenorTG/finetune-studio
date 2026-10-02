# webui/routes — core workflow routes (activity, rag, pages, data-prep, data-prep-chat, training, benchmarks, testing)

These eight route modules implement the project workflow: landing/dashboard pages, project-scoped RAG corpora, the data-prep (file → Q&A pair) pipeline and its chat assistant, training-run lifecycle (start/stop/export/auto-suite), benchmark execution/judging, and the Testing-tab inference/suite runner. They are mounted onto `app.py`'s FastAPI app with varying prefixes and together they are what the templates in `webui/templates/` call via `fetch()`.

## `activity.py`

**Purpose:** Aggregates every background-task subsystem (training, data-prep, RAG builds, benchmarks, exports, HF downloads, system updates, inference model load) into one unified activity feed for the drawer UI.

**Key functions:**
- `_norm_status(raw)` / `_progress_for(status)` — collapse each subsystem's own status vocabulary (`"completed"`, `"ready"`, `"succeeded"`, …) into one shared vocabulary: `queued | running | done | error`.
- `_task_key(t)` — builds a dedup identity (`kind:run:<id>` or `kind:id:<id>`) so a live in-memory row and its persisted DB row collapse to a single entry instead of appearing twice.
- `_ProjCache` — memoizes `db.get_project()` lookups for the duration of one snapshot build (each snapshot touches every project referenced by every task).
- `_persistent_tasks(projs)` — reads eight DB tables (`activity_events`, `runs`, `benchmarks`, `exports`, `data_prep`, `rag_builds`, `hf_downloads`, `updates`) and turns each row into a normalized task dict. Every DB read is wrapped in its own `try/except` that appends a `{"kind": "_error", ...}` sentinel on failure rather than crashing the whole snapshot — one subsystem's DB error does not blank the entire activity feed.
- `collect_activity()` — the main entry point. Builds live rows first (training engine state, inference engine loaded/loading model, in-memory data-prep `_RUNS`, in-memory HF `_DOWNLOADS`), tags them `_live=True`, then merges in `_persistent_tasks()` rows that aren't already covered by a live row (`seen` set keyed by `_task_key`). Sorts running/queued first, then by `started_at` desc, caps at 60 rows but always keeps live rows even past the cap. Computes `active_count`/`by_kind`, excluding persisted running/queued rows older than 2 hours (`_STALE_AFTER`) so an interrupted job that never reached a terminal status doesn't spin the badge forever.

**Wiring:** `GET /api/activity` (one-shot) and `GET /api/activity/events` (SSE, 1s poll with a content-fingerprint dedup so idle periods only send keepalive comments) both call `collect_activity()` via `asyncio.to_thread` (it is sync and DB-heavy; calling it inline blocked the event loop every second). Imports `training_engine`/`inference_engine` from `webui.app` and `_RUNS` from `webui.routes.data_prep` and `_DOWNLOADS` from `webui.routes.hf_models` lazily inside the function (avoids import-time cycles).

**Gotcha:** A deleted project must not leave a ghost "running" row — the training-engine live-row branch explicitly checks `db.get_project(pid)` and sets `proj_name=""` plus skips appending if the project is gone (lines ~338-346). Any new live-task source added here needs the same check.

## `rag.py`

**Purpose:** Per-project PortableRAG corpus lifecycle — build, rebuild vectors, search, chat (RAG-augmented), bundle import/export, source management.

**Key routes/functions:**
- `_corpus_dir(pid)` → `~/.finetune-studio/rag_corpora/<pid>/`. `_project_404(pid)` guards every route that needs a real project before touching the filesystem.
- `rag_build` (`POST /{pid}/rag/build`) — sources chunks from the project's **parsed** files (`files/<sha>/*.txt`), not raw uploads. Not a background task (those dropped the systemd `HF_HOME` env var); the build itself runs in `asyncio.to_thread` so the event loop and `/build/progress` stay responsive. 404s on an unknown project.
- `rag_quick` (`POST /{pid}/rag/quick`) — attempts to promote un-parsed library files into QA sources, then builds. Inspect the returned `failed` list: top-level `ok` reflects the build result and can be true despite individual promotion failures.
- `_rag_build_snapshot` / `rag_build_status` / `rag_build_progress` — progress polling: reads `manifest.json` (if present → `phase=done`), else `chunks.parquet` existence (→ `embedding`), else counts files under `sources/` (→ `chunking`), else `queued`. SSE variant streams for up to 10 minutes then emits a `timeout` phase (no `error` phase).
- `rag_rebuild_vectors` (`POST /{pid}/rag/rebuild-vectors`) — re-embeds existing chunks with a (possibly new) embedder. **See Gotchas — this had a silent-drop bug, fixed in this audit.**
- `rag_chat` (`POST /{pid}/rag/chat`) — retrieves top-k via `PortableRAG.search`, builds a context-grounded system prompt, generates via `inference_engine` if a model is loaded there, else falls back to `ModelManager.get_manager().active()`. Strips `<think>` blocks via `webui.thinking.split_thinking` before returning.
- `rag_bundle` / `rag_import` — tar/zip export and import of a full corpus (including, optionally, cached embedder/reranker model weights so it's portable across machines). Import stages the upload to a `NamedTemporaryFile` so it streams instead of buffering the whole archive in memory, and records an `activity_events` row on success so the import shows up in the Activity drawer.

**Wiring:** Mounted at `/api/projects` (`app.py:324`, confirmed: `GET /api/projects/{pid}/rag`, etc.). `rag_chat`/`rag_search` are called from `project_rag.html` / `chat_v2.html` templates. `project_rag_page` in `pages.py` and `data_prep_chat.py`'s tool-calling loop do **not** call into this module — data-prep chat reads sources via `qa_fs`/`project_filesystem` directly, not through PortableRAG.

**Gotchas / invariants:**
- **Fixed in this audit (`rag.py:410-434`):** `rag_rebuild_vectors` previously passed `req.embedder` straight through to `PortableRAG.rebuild_vectors()`. When the request body omitted `embedder` (the documented flow: `POST /rag/settings` to change the embedder, *then* `POST /rag/rebuild-vectors` to apply it — see `rag_patch_settings`'s docstring), `rebuild_vectors(embedder=None)` falls back to `manifest.embedding_model.name` (`rag_portable/store.py:800`) — the **old**, currently-active embedder — not the newly-patched `manifest.rag_settings.embedder`. The settings patch was silently discarded on the very call meant to apply it. Fixed by having `rag_rebuild_vectors` read the manifest's `rag_settings.embedder` as the fallback when the request doesn't specify one, before calling `rebuild_vectors()`. Regression test: `tests/test_webui_routes_workflow_audit.py::test_rebuild_vectors_applies_pending_embedder_from_settings_patch`. Note: the only current frontend caller (`rag.html:saveSettings()`) doesn't send `embedder` in its PATCH body today, so this bug was latent for the shipped UI but live for any other API consumer (scripts, future UI, the documented contract itself).
- `Manifest.embedding_model` (what vectors were actually built with) and `Manifest.rag_settings.embedder` (the "current/requested" setting) are two *different* fields that only stay in sync through a successful `rebuild_vectors()` call — any future route touching embedder settings must respect this split.
- `rag_build`'s `reset=True` path does a real `shutil.rmtree(corpus)` before rebuilding — irreversible.
- A normal rebuild consumes parsed project artifacts; a raw library upload is not included until parsing/promotion succeeds. Quick Index attempts promotion, but its top-level `ok` does not mean every file was promoted.
- `_project_404(pid)` is not called on every route in this module. Verify the route-specific guard before treating corpus settings, sources, search, bundle, or chat operations as project-existence protected.

## `pages.py`

**Purpose:** Server-rendered Jinja2 page routes — the dashboard and all `/projects/{pid}/...` tab pages. No business logic beyond building template context; all mutation happens through the JSON API routes in the other seven files.

**Key functions:**
- `_project_ctx(pid)` — the shared context builder used by most project pages: loads project, rags, runs (with `benchmarks` attached per run), datasets, and scans run output dirs for exported models (`_scan_run_models`).
- `_scan_run_models(runs)` — walks each run's `output_path/{merged,abliterated,gguf}` subdirectories on disk. For `gguf`, lists every `*.gguf` file as a **separate** model entry (unlike `training.py`'s `/runs/{run_id}/exports`, which collapses all GGUF files into one arbitrarily-chosen representative path — see the `training.py` section below).
- `_recent_suite_runs(pid, limit)` — flattens all benchmarks across all runs into one newest-first list for the Testing page's "recent suite runs" panel. Tolerates `scores` stored as either a dict or a JSON string.
- `benchmarks_page` — `ran_at` may be NULL on old rows; both the formatter and the sort treat it as `0` (not `localtime(None)` = now), matching `_recent_suite_runs`.
- `debug_info()` (`GET /api/debug/info`) — Settings-page system info: Python/platform, GPU via `nvidia-smi` subprocess (best-effort, swallows `FileNotFoundError`/`SubprocessError`/`OSError`/`ValueError` into a `gpu_error` field rather than failing the whole endpoint), and installed-package versions probed via `__import__`.

**Wiring:** Every page route lazy-imports its data dependencies (`db`, `discovered_models`, `training_engine`, `inference_engine`) inside the handler body, not at module level — this module is imported very early during router registration, so top-level imports of `webui.app` globals would create an import cycle. `project_testing_page` imports `_discover_suites` from `benchmarks.py` (used, not dead). `export_page` imports `project_export.annotate_runs_for_export`/`runs_by_id` (lane B1, exports.py's sibling helper module).

**Gotchas:**
- `inference_page`: when a model is loaded outside the `discovered_models` list (e.g. loaded via `/api/chat-v2/load`), it synthesizes a `SimpleNamespace` stand-in via `load_model_info()` so the template doesn't crash on a missing object — intentional, not a bug.
- `# mypy: disable-error-code="arg-type,call-arg"` at the top is a known starlette stub mismatch (`TemplateResponse(name, context)` vs the newer `(request, name, context)` signature) — not a real type error, documented inline with a link to the upstream issue.

## `data_prep_chat.py`

**Purpose:** Server-side tool-calling chat for the Data Prep page. Lets the user drive Q&A mining conversationally; the model can call `list_sources` / `read_source` / `list_qa_pairs` / `create_qa_pairs` against the project filesystem.

**Key functions:**
- `TOOLS_CATALOG` / `SYSTEM_PROMPT` — the tool schema and the prompt instructing even small local models to emit `<tool_call>{"name":...,"arguments":{...}}</tool_call>` blocks (used for local GGUF backends; external OpenAI-compatible backends use native `tools`/`tool_choice` instead, synthesized back into the same `<tool_call>` text format by `_chat_external` so the rest of the loop is backend-agnostic).
- `_run_tool(pid, name, args)` — executes one tool call against `project_dir(pid)/qa/sources/*.json` and `qa_fs`. `create_qa_pairs` silently skips any pair missing a non-empty question/answer (`written` count reflects only accepted pairs — not a bug, it's reported back via `{"written": N}` so the caller sees the real count vs requested).
- `_extract_tool_calls(text)` — two-pass parser: strict regex match on a properly-closed `<tool_call>...</tool_call>` block first; if none found, a brace-balanced fallback walks forward from an unclosed `<tool_call>` tag to recover from models (observed: local Qwen3 GGUF) that forget the closing tag. Deduplicates by `(name, json.dumps(args, sort_keys=True))`.
- `_looks_truncated(text)` — detects `max_tokens` cutoffs mid-`<think>` or mid-`<tool_call>` so the caller gets `_TRUNCATION_MSG` instead of a confusing empty/garbled reply.
- `data_prep_chat` (`POST /projects/{pid}/data-prep/chat`) — backend resolution order: explicit `provider_id` (may call `ModelManager.load()`) → explicit `external_api` → **default: `resolve_helper_backend()`** from `data.prep.generator`, which only returns an *already-loaded* helper backend and never loads anything itself (matches the "one true way" helper-load contract — see `data_prep.py` section). Runs a tool-call loop capped at `max_rounds` (default 6, clamped 1–12; non-integer → error). Blocking work (provider load/engine unload, `_run_tool`) runs via `asyncio.to_thread`; the assistant turn is appended once per round even when several tool calls are emitted.
- `_chat_global_engine` vs `_chat_local`: when `provider_id` resolves to the model the global `inference_engine` already has loaded, the handler reuses that engine directly instead of calling `ModelManager.load()` again — avoids a second `Llama()` instance mmap-racing the same GGUF file, which the inline comment says previously wedged the host (CPU pinned, status API hung).

**Wiring:** Called from the Data Prep page's chat panel. Depends on `data.fs.qa` and `data.fs.paths.project_dir` directly — does **not** go through `rag.py`'s `PortableRAG`.

**Gotcha:** `_chat_local`'s fallback path (`mgr.chat()` raises → falls back to `mgr.generate()` with a manually flattened prompt) is a real degraded-mode path, not dead code — it's there for providers that only implement text completion. Both failure branches are logged via `log.exception` before falling through, so a silent swallow is not possible here.

## `training.py`

**Purpose:** Training-run lifecycle — presets, config resolution, start/stop, run-scoped export/abliteration/quantization triggers, auto-suite generation trigger.

**Key functions:**
- `TRAINING_PRESETS` — the real preset table (nano-fast/nano-quality/standard/standard-long/high-rank/large/qlora), each a concrete LoRA config, not a toy default.
- `_resolve_model_path(model_path, allow_download)` — blocks a bare `org/repo` model id from silently triggering a multi-GB Hub download mid-run unless it's already cached (HF hub cache or the app's own `hf_models` download dir) or the caller passes `allow_download=true`. Referenced inline as a fix for a specific prior incident (`c327fa36`).
- `start_training` (`POST /start`) — resolves `data_path` (direct or via `dataset_id` → `db.get_dataset`, with an ownership check against `project_id`), builds a `TrainingConfig` from a preset + overrides or from raw body fields, rewrites a bare `output_dir="output"` default into a per-run path (`output/projects/{pid}/runs/{run_id}`) so concurrent runs can't silently clobber each other's adapters (documented as a fix for issue `E2E-25`), **refuses while the engine is training/loading/saving** (checked before any run row is created, so a busy engine's persister is never hijacked), persists the run row immediately (`db.update_run`) before `attach_run()` wires the training engine's progress callback to it, **unloads all resident models** (`llama_loader.unload_all_models()`) before calling `training_engine.start()` (a `start` failure marks the run `error` and returns `{error}`) so a resident data-prep helper doesn't compete for VRAM with the training process.
- `set_run_output` (`POST /runs/{run_id}/set-output`) — 404-style `{error}` for an unknown run id. Bool body fields (`bf16`, `export_gguf`, `abliterate`, `export_imatrix`) go through `_coerce_bool`, so `"false"` is false.
- `export_run` (`POST /runs/{run_id}/export`) — delegates to `training.run_export.export_trained_run`.
- `trigger_auto_suite` (`POST /runs/{run_id}/auto-suites/generate`) — **manual, on-demand** trigger for the deterministic JSONL→benchmark-suite converter (`testing.generate_suite.generate_suite_from_training_data`). Default is full coverage (every dataset row becomes one case); `sample_size` opts into a deterministic uniform sample, labeled `-sampledKofN` so a sampled run can never be misread as full coverage.
- `list_exports` (`GET /runs/{run_id}/exports`) — **see Cross-module findings below.**

**Wiring:** Mounted at `/api/training` (see `app.py:310`). `start_training` is the entry point the project training page's "Start" button calls. **Auto-suite generation on training completion is not triggered from this file** — it is wired directly into `training.engine` (the `TrainingConfig.data_path` field is explicitly commented "for auto-suite generation", and `engine.py` calls `generate_suite_from_training_data` + inserts into `auto_suites` on its own completion path, independent of the `/runs/{run_id}/auto-suites/generate` manual endpoint here). Confirmed real (not vestigial): `training/engine.py:1082-1098` both imports and calls `generate_suite_from_training_data`, and inserts into the same `auto_suites` table this file's `trigger_auto_suite` uses — one true code path for the suite-row shape, two triggers (automatic post-training, manual on-demand).

**Gotcha — `min(gguf_files)` in `list_exports`:** when a run's `gguf/` output dir has multiple quantizations, the handler reports only one `"path"` for the whole `gguf` format entry, chosen via `os.path.join(gguf_dir, min(gguf_files))` — i.e. the **alphabetically smallest filename** (e.g. `q4_k_m` sorts before `q5_k_m`/`q8_0`/`f16`), not "most recent" or "highest quality." This is cosmetic (the full `files` list is also returned) and, per the cross-module finding below, this whole endpoint has no found template/JS caller — left as-is rather than fixed, since the real listing endpoint (`exports.py`) does not have this issue and is the one in active use.

## `data_prep.py`

**Purpose:** The data-prep pipeline routes (upload → parse → mine Q&A → curate → export dataset) plus `ModelManager` provider CRUD (load/unload/list providers used by data-prep and the Testing tab).

**Key functions:**
- `_RUNS: dict[tuple[pid, run_id], dict]` — the in-memory registry of active/recent prep runs (`activity.py` reads this directly for live rows). Each entry holds the `runner` object, its `progress_log` list, filename, and byte count.
- `_spawn_bg(coro)` — fire-and-forget helper holding strong refs in `_BG_TASKS` (used by resume and reprocess so tasks are not GC'd). `_RUNS` entries are never evicted (known leak, bounded by runs per process lifetime). `update_qa_route` only accepts `question/answer/status/chunk_idx`; `export_qa` rejects unknown `only` values (400) because `only` is part of the written filename; SSE `stream_events` drains the log once after seeing a terminal stage (no duplicate last entry).
- `enqueue_prep_run` / `enqueue_source_prep_run` — create a DB run row, build a `DataPrepRunner`/`QueuedSourcePrep`, register it in `_RUNS`, and schedule `_run_prep_background` via `BackgroundTasks`. The "source" variant takes only a path (not bytes) so a bulk queue of many files doesn't hold all their content in RAM simultaneously — `start_bulk_prep`'s docstring is explicit that Starlette runs background tasks in insertion order and that's relied on (one GPU, one helper model at a time).
- `resume_stale_data_prep_runs()` — called on startup (presumably from `app.py`, not in this file) to re-derive and re-enqueue any run left `queued`/`running` by a prior crash/restart, instead of just marking everything `failed` (explicitly documented as a fix for a prior silent-data-loss behavior — `db.reconcile_stale_data_prep` used to drop the rest of a multi-file batch).
- `load_provider` (`POST /providers/{pid}/load`) — **routes through the shared loader contract**: calls `llama_loader.resolve_loader_overrides(extra, caller="providers/load", model_path=pid, default_ctx=False)` before `ModelManager.load()`, and unloads the global `inference_engine` first if it holds a model, so the two engines never compete for VRAM simultaneously. No standalone `Llama()` construction anywhere in this file.
- `export_qa` (`GET /projects/{pid}/data-prep/export`) — runs `coverage_fill.fill_all_project_gaps(pid)` first and **blocks the export with HTTP 409** if the returned report marks a chunk uncovered, unless `force=true` (then it logs a warning and proceeds). This is a guard over the coverage helper's current heuristic, not proof that every declared or on-disk chunk has a valid Q&A pair; see `modules/data-prep.md` for the missing trailing-chunk and validation caveats.

### Source-promotion trust boundary

`POST /projects/{pid}/data-prep/sources` accepts either `file_id` or an
absolute `data_path`. The `file_id` path checks the owning project before
calling `promote_file_library_upload`; the `data_path` path is passed to
`project_filesystem.register_qa_source`, which reads the supplied file and
copies its bytes into the project's content-addressed store. The shipped UI
only sends `file_id`, but `tests/test_data_prep_promote.py::test_promote_with_data_path_directly`
preserves external-path ingestion (including a temporary file outside the
project root). The route has no authentication guard, like the rest of the
current FastAPI app. This is an unresolved trust-boundary issue, not proof
that deployment exposure is safe: do not treat `pid` or the session-bar UI as
authorization, and place network isolation/authentication outside the app.
A fix that removes or confines `data_path` must first decide whether the
tested external-path API contract is still needed.

**Wiring:** Mounted alongside `_pages` sub-router (`data_prep_page`, the HTML page) and the main `router` (JSON API). `providers/*` routes back the Testing tab's model selector and `data_prep_chat.py`'s `provider_id` path. **Priority check 3 result:** confirmed — both the chat tool-loop (`data_prep_chat.py`, via `resolve_helper_backend()`) and this file's own provider endpoints go through `ModelManager`/`inference_engine`/`resolve_loader_overrides()`; no standalone load path found in either file.

**Gotcha:** `reprocess_source` schedules its background work via a raw `asyncio.get_event_loop().create_task(_bg())` instead of the `BackgroundTasks` pattern every other upload/start route in this file uses — the inline comment calls this out as "historical behaviour," not an oversight, but it's the one inconsistent concurrency pattern in the file if someone is hunting for "why does this run's row not show up the same way."

## `benchmarks.py`

**Purpose:** The project Benchmarks tab's backend — suite discovery, running a suite against a trained run or the base model, judging (heuristic/AI/local/secondary-local), comparing two runs, and the training-leakage evaluation (benchmark against the model's own training data).

**Key functions:**
- `_discover_suites` / `discover_suites` (imported from `benchmarks.suite_defs`) — the actual suite inventory: real HF benchmarks (MMLU/GSM8K/HellaSwag/etc. via `real_benchmarks.RealBenchmarkSuite`), synthetic suites, local JSON suite files, and per-project `auto_suites` (the self-generated full-corpus suite). **See cross-module finding below regarding the separate class-based `BenchmarkSuite`/`BaseBenchmark` system.**
- `_validate_suite_file` — branches on `parse_real_suite_path(suite_path)`: a `real://...` path loads HF rows via `RealBenchmarkSuite.load_cases()` (supports `num_samples`/`full_run`/`seed`/`order`); anything else is treated as a local JSON suite file loaded via `testing.suite.load_test_suite`.
- `_resolve_trained_target(run)` — prefers `output_path/merged/` (if it has a `config.json`) over the raw `output_path`, so benchmarking a run always targets the actually-loadable merged artifact when one exists.
- `_int_field` / `_parse_sample_knobs` — integer body fields (`max_tokens`, `seed`, `num_samples`, `max_cases`) raise `ValueError` on junk and the handlers answer 400 instead of 500.
- `_execute_benchmark` — the shared runner for both `run_benchmark` (trained run) and `run_benchmark_base` (untrained base model): loads the target into a fresh `InferenceEngine`, unloads the global engine first (`_unload_global_inference`), runs the suite, judges (heuristic only at run-time — `judge_mode in ("ai","local")` logs a warning and falls back to heuristic rather than actually running those judges inline; those are meant to run later via `judge_benchmark`), persists via `db.create_benchmark`, and **always** unloads its engine in a `finally` block regardless of success/failure.
- `judge_benchmark` (`POST /{bid}/judge`) — the real home for `ai`/`local`/`secondary_local` judge modes deferred by `_execute_benchmark`. `secondary_local` is explicitly non-authoritative (`authoritative_scores_unchanged: True` in the response) — it records a second opinion into `judge_input.secondary_judge` without touching the case's primary `verdict`/scores.
- `compare_runs` (`GET /{pid}/compare`) — per-suite score delta between two runs using each run's *latest* benchmark per `suite_name` (`_suite_scores_for_run`).
- `evaluate_training_for_run` — benchmarks a run against its own training dataset (`testing.training_eval.build_training_eval`), explicitly labeled `eval_kind=training_leakage` / `leakage_warning` in the persisted scores so this can never be misread as a held-out generalization score.

**Wiring:** Mounted at `/api/benchmarks` (`app.py:316`) — routes like `run_benchmark` resolve to `POST /api/benchmarks/projects/{pid}/runs/{rid}/run`. `_discover_suites` is also imported and called directly by `pages.py` (`project_testing_page`, `benchmarks_page`). `training.py` does not import this module — it calls `testing.generate_suite.generate_suite_from_training_data` directly for its own auto-suite trigger. `testing.py` does not import from `benchmarks.py` either; each module's suite-run path is independent (see `testing.py`'s Wiring note below).

**Priority check 2 result — cross-module finding (do not edit, different lane owns `benchmarks/__init__.py`):** `src/finetune_studio/benchmarks/__init__.py` defines a class-based system (`BenchmarkSuite`, `BaseBenchmark`, and ten `BaseBenchmark` subclasses: `MMLUSample`, `HellaSwagSample`, `ARCChallengeSample`, `TriviaQASample`, `WinoGrandeSample`, `IFEvalSample`, `ToolBenchSample`, `GSM8KSample`, `HumanEvalSample`, `TruthfulQASample`, `PersonaTest`). **This route file imports none of it** — only `finetune_studio.benchmarks.real_benchmarks.RealBenchmarkSuite` (a different class in a different submodule: `benchmarks.py:16-22`, `chat_v2.py:252` is the only other caller, also `RealBenchmarkSuite` not `BenchmarkSuite`) and `finetune_studio.benchmarks.suite_defs.discover_suites`/`is_selectable_suite`. A repo-wide grep (`grep -rn "from finetune_studio.benchmarks import\|BenchmarkSuite\|BaseBenchmark" src/ tests/`) found **zero** callers of `benchmarks/__init__.py`'s `BenchmarkSuite`/`BaseBenchmark` classes anywhere in `src/` or `tests/` outside the file itself. This class-based system appears to be fully dead weight superseded by the real `RealBenchmarkSuite` + `suite_defs.discover_suites` system this route file actually drives.

## `testing.py`

**Purpose:** The Testing-tab backend — load/unload a model into the global `inference_engine`, chat, run a test suite (plain or RAG-grounded), and run training-data/held-out evaluation.

**Key functions:**
- `load_model` (`POST /load`) — goes through `llama_loader.resolve_loader_overrides()` (shared loader contract) and explicitly unloads `ModelManager`'s active provider first, mirroring `data_prep.py`'s `providers/{pid}/load` symmetry (both directions free the other engine before loading).
- `_ensure_model_loaded(project_id, override_path)` — shared by `run_test_suite`, `run_rag_test_suite`, and `evaluate_training_dataset`: an explicit `override_path` always (re)loads that model; otherwise reuses whatever's already loaded in `inference_engine`; otherwise auto-loads the project's latest merged model via `testing_models.resolve_latest_merged_model` — returns a `JSONResponse` error (never raises) on any load failure so callers can `isinstance(result, JSONResponse)`-check and pass it straight through.
- `run_test_suite` (`POST /run-suite`) — plain suite run against `inference_engine` (400 on missing `suite_path`, 404/400 when the suite file is missing/invalid, before the engine lock is taken); returns full `CaseResult` fields (verdict/judge/scoring_method/validity/keywords/transcript) per case, not just pass/fail.
- `run_rag_test_suite` (`POST /run-rag-suite`) — RAG-grounded suite run via `testing.rag_suite.run_rag_suite_evaluation`, then `_persist_rag_report` writes the result into `benchmark_runs`/`cases` (same tables `benchmarks.py` uses) — tries to attach the result to an existing `done` run whose `output_path` is a prefix of the model path actually used, else creates a placeholder run (`status="done"`, notes="Evaluation-only RAG benchmark; no training performed") so there's always a valid run FK.
- `evaluate_training_dataset` (`POST /evaluate-training`) — `eval_kind` body field selects `build_heldout_eval` vs `build_training_eval` (`testing.training_eval`); results are always labeled with a `leakage_warning` so a high training-set score can't be misread as generalization.

**Wiring:** Mounted at `/api/testing` (`app.py:312`); `_ensure_model_loaded`/`resolve_latest_merged_model` back the Testing page's auto-load behavior, independent of `benchmarks.py`'s separate `_execute_benchmark` load path (the two do not share a helper — each loads into its own engine instance or the shared `inference_engine`, respectively; this is intentional, not duplication, since Testing-tab runs use the persistent global engine while Benchmarks-tab runs use a disposable per-request `InferenceEngine()`).

**Gotcha:** `_persist_rag_report`'s run-matching heuristic (`output_path and model_path.startswith(output_path)`) only works when `model_path` is literally inside the run's `output_path` tree (e.g. `.../merged/...`) — a model loaded from an unrelated absolute path (e.g. a manually-copied export) will always fall through to creating a new placeholder run rather than being attributed to the real training run. Not fixed — no evidence this is wrong behavior vs. an accepted limitation of best-effort attribution, and changing the matching heuristic risks misattributing RAG eval results to the wrong run.

## Cross-module findings — NOT fixed, needs parent coordination

1. **Duplicate export-listing logic (`training.py` vs `exports.py`, lane B1 owns `exports.py`).**
   - `training.py:775-805` (`GET /api/training/runs/{run_id}/exports`, mounted at `/api/training` per `app.py:310`) scans `output_path/{merged,adapter,gguf}` directly on disk via `os.path.isdir`/`os.listdir`, not project-scoped (only `run_id` in the path, no `pid`).
   - `exports.py:378-379` (`GET /api/projects/{pid}/runs/{rid}/exports`, mounted at `/api` per `app.py:328`) is the DB-backed, project-scoped equivalent (`list_run_exports`).
   - These are two independently-maintained implementations of "list what's been exported for a run" with different URL shapes (one project-scoped, one not) and different data sources (disk-scan vs DB). A repo-wide grep of `webui/templates/*.html` found no caller of either exact path pattern via simple string search, so it's unclear which (if either) is actually exercised by the frontend today — that needs live traffic/log inspection, not grep, to resolve safely. Recommend the parent session decide whether `training.py`'s copy should be deleted in favor of `exports.py`'s DB-backed version, or whether it serves a still-needed non-project-scoped use case (e.g. a script/API consumer that only has a `run_id`).
   - Not fixed per the task's explicit instruction to report rather than resolve blind across lane boundaries.

2. **Dead class-based benchmark system (`src/finetune_studio/benchmarks/__init__.py`, different lane).**
   - `BenchmarkSuite` (`benchmarks/__init__.py:39`) and `BaseBenchmark` (`benchmarks/__init__.py:172`) plus ten subclasses (`MMLUSample`, `HellaSwagSample`, `ARCChallengeSample`, `TriviaQASample`, `WinoGrandeSample`, `IFEvalSample`, `ToolBenchSample`, `GSM8KSample`, `HumanEvalSample`, `TruthfulQASample`, `PersonaTest`) have **zero callers anywhere in `src/` or `tests/`** (verified via `grep -rn "from finetune_studio.benchmarks import\|BenchmarkSuite\|BaseBenchmark"`).
   - The real, live benchmark backend this project's `benchmarks.py` route file actually drives is `finetune_studio.benchmarks.real_benchmarks.RealBenchmarkSuite` + `finetune_studio.benchmarks.suite_defs.discover_suites` — a *different* class in a *different* submodule of the same package, verified working against a real trained model earlier today (91.1% on the full-ingested-corpus suite per the task brief).
   - Recommend the lane owning `benchmarks/__init__.py` either delete the dead class hierarchy or confirm it's intentionally kept for a documented future use (e.g. an external plugin API) — not something this lane can safely decide or touch.

## Discrepancy fixed in this lane

- **`rag.py:410-434` — `rag_rebuild_vectors` silently discarded a pending embedder change.** Full detail under the `rag.py` section above. Fix: fall back to the manifest's `rag_settings.embedder` (the value `POST /rag/settings` writes) instead of letting `PortableRAG.rebuild_vectors(embedder=None)` silently re-use the stale `embedding_model.name`. Regression test added: `tests/test_webui_routes_workflow_audit.py` — `test_rebuild_vectors_applies_pending_embedder_from_settings_patch`, verified passing: `.venv/bin/python -m pytest tests/test_webui_routes_workflow_audit.py -q` → `1 passed`.
