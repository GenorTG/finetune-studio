# webui/routes — core project lifecycle, data, settings, models, export, chat (lane B1)

Module purpose: the FastAPI route handlers that back the Studio's core
Project/RAG/Run CRUD, file + dataset management, global and project-scoped
inference chat, model discovery/loading/download, GGUF export, and
system/settings/update surfaces. These are the routes a browser session
actually calls; each file below is one `APIRouter` mounted in
`webui/app.py`.

All 19 files in this lane were read in full (not grepped) for this audit.

---

## `routes/data.py` (44 lines, mounted at `/api/data`)

Legacy, **non-project-scoped** data API: list files under `settings.data_dir`,
upload, validate, preview, dedup. Project work uses `file_library.py` and
`datasets.py`. The old `templates/data.html` is orphaned: there is no `/data`
page renderer, and its multipart form submits `files`/`project_id` while this
API accepts one `file`. The API routes remain mounted for direct/legacy API
clients, but the template is not a working UI for them.

- `list_files()` → `scan_data_files(settings.data_dir)`.
- `upload_file(file)` → writes `file.filename` under `settings.data_dir`.
- `validate(path)` → `validate_file(path)`, arbitrary absolute path accepted (CLI-equivalent, not project-scoped).
- `preview(path, limit)` → `load_jsonl(path)[:limit]`, wrapped so a bad file returns `{"error": ...}` instead of 500.
- `dedup(path)` → read-only report of `dedup_data()`'s `(unique, dupes)`; does not persist the deduped file.

**Wiring discrepancy (2026-10-01):** an earlier audit draft called the
orphaned template a live page. The only matching page route is the project
file browser at `/projects/{pid}/data`, which renders `project_data.html`.

**Gotchas / invariants**
- `upload_file` now does `os.path.basename(file.filename)` before joining
  into `settings.data_dir` — fixed 2026-10-01 (was a path-traversal write:
  a filename like `../../etc/cron.d/x` escaped `data_dir` entirely).
- The `aiofiles.open(...).write(content)` call was never `await`ed — the
  coroutine was created and immediately discarded, so **the file write
  silently never happened** even though the route returned `{"path":
  dest, "size": len(content)}` as if it had succeeded. Fixed 2026-10-01 to
  `await f.write(content)`. Any future route added to this module with
  `aiofiles` must await every I/O call — `pytest -W error` will not catch a
  missing await on its own; the regression test
  (`tests/test_webui_routes_core_audit.py::test_upload_sanitizes_path_traversal_filename`)
  asserts the file actually exists afterward, not just a 200 status.
- `validate`/`preview` accept arbitrary absolute paths with no scoping — by
  design (CLI-equivalent power-user page), but means any caller who can
  reach this router can read/validate any file the process has permission
  to open. Flagged, not changed, since restricting it would be a behavior
  change beyond this audit's scope.

## `routes/project_export.py` (48 lines, no router — template-context helpers)

Pure functions consumed by `pages.py`'s `export_page()` to annotate runs for
the Export tab's "expand row" UI. No HTTP routes of its own.

- `runs_by_id(runs)` — `{run["id"]: run}` index for O(1) lookup by export rows.
- `export_path_row_id(path)` — turns a filesystem path into a stable DOM id (`#m-...`).
- `annotate_runs_for_export(runs)` — copies each run dict and adds `has_merged` / `has_adapter` / `exportable` booleans. A run is exportable once it's `done|completed|failed` **and** has either a merged dir or a raw adapter — export-time merging is assumed for adapter-only runs.

**Wired into**: `pages.project_export_page`/`export_page` (not in this lane) imports these three functions directly; depends on `training.run_export.{merged_dir_ready,adapter_dir_ready}`.

## `routes/settings.py` (80 lines, mounted bare — self-prefixed `/api/settings`)

Global app settings (host/port/CORS/proxy), persisted at
`~/.finetune-studio/settings.json`, merged over in-code `DEFAULTS` on read.

- `GET /api/settings` — `DEFAULTS` merged with the saved file.
- `PATCH /api/settings` — shallow dict merge + rewrite the whole file; 400 if body isn't a JSON object.
- `POST /api/settings/reload` — reports `needs_restart=True` if `host`/`port` are present in the saved overrides (the actual listening socket is bound at process start, so changing these here never hot-reloads).

**Gotchas**: `_load()` swallows any JSON parse error and returns `{}` (just
logs a warning) — a corrupted settings.json silently resets to defaults
rather than surfacing to the caller. This is a deliberate "never 500 the
settings page" tradeoff, not an oversight.

## `routes/project_models.py` (88 lines, tag `project-models`)

Expand-row file listing for a trained model export directory on the Models
tab. Only one route.

- `list_dir_contents(path, limit=20)` — top-level (non-recursive) dir listing with size/is_dir, sorted by name, truncated at 20.
- `_normalize_export_path(raw)` — reconstructs an absolute path from a FastAPI `{export_path:path}` capture (which drops the leading `/`).
- `export_belongs_to_project(pid, export_path)` — **ownership check**: true only if the resolved path equals or is nested under some run's `output_path` for that project. This is the correct pattern other routes in this lane should imitate (see `projects.py` gotchas below).
- `GET /projects/{pid}/models/{export_path:path}/contents` — 404 if project missing or path doesn't exist, **403** if the path doesn't belong to the project. Clean example of priority-check-3 done right.

## `routes/quality.py` (120 lines, self-prefixed `/api/data`, tag `data-quality`)

Thin GUI wrappers around CLI data-quality commands (`fts analyze/augment/
optimize/validate-hallucination/convert`) — not project-scoped, takes a bare
filesystem `path` in the request body. Every handler: 404 if `path` doesn't
exist, else try the real CLI class and return `status: ok|error` with HTTP
200 either way (the `DataJobResponse.error` field is the signal, by design —
this page predates the project-scoped 404 convention and there is no
`{pid}`/`{id}` in the URL for priority-check-3 to apply to).

- `/analyze` → `DataQualityAnalyzer.analyze()`
- `/augment` → `DataAugmenter.run()`
- `/optimize` → `ConfigOptimizer.recommend()`
- `/hallucination-check` → `HallucinationGuard.scan()`
- `/convert` → `FormatConverter.convert()` (separate `ConvertRequest` model: `target_format` instead of `output`-only)

## `routes/comparison.py` (126 lines, mounted at `/api/compare`)

Side-by-side model comparison + a **third, independent** RAG-chat
implementation (global, not project-scoped).

- `POST /compare/load` → `comparator.load_model(name, path)` (benchmarks.comparison singleton) — loads a *second* model alongside whatever the global `inference_engine` holds, for A/B comparison.
- `POST /compare/run` → `comparator.run_comparison(test_suite, config)`.
- `POST /compare/cleanup` → `comparator.cleanup()`.
- `POST /rag/chat` → retrieves from the global `VectorStore(settings.rag_store_path)` (not any project's corpus), augments messages with a system-prompt context block, generates via the global `inference_engine`.

**Gotcha — route-path duplication (cross-module, flagged not fixed)**: this
file's `/api/compare/rag/chat` is the **third** RAG-chat implementation
visible from this lane alone:
1. `comparison.py::rag_chat` — global `VectorStore`, no project scoping.
2. `chat_v2.py::chat` (`/api/chat-v2/projects/{pid}/chat`) — multi-RAG, PortableRAG-or-Chroma via `_search_rag_attachment`, per-project.
3. `projects.py::query_rag` (`/api/projects/{pid}/rags/{rid}/query`) — single-RAG via `RAGManager`, returns raw chunks (no generation).

`routes/rag.py::rag_chat` (lane B2, **not read for this audit**) is likely a
*fourth*. None of these share retrieval or generation code. This is a
pre-existing "one true way" violation that predates today's work — flagged
here for parent/cross-lane coordination since consolidating into one chat
implementation would touch `rag.py` outside this lane's file list.

## `routes/system.py` (151 lines, mounted bare — self-prefixed `/api/system`)

Host RAM + per-GPU VRAM snapshot, polled every ~3s by page templates.

- `_ram()` — via `psutil.virtual_memory()`; returns `available: False` if psutil isn't installed (never raises).
- `_vram()` — tries `torch.cuda.mem_get_info()` first, falls back to parsing `nvidia-smi --query-gpu=...` subprocess output if torch/CUDA isn't available; returns `[]` if both fail.
- `GET /api/system/resources` — `{ram, vram}` for the UI bars.
- `GET /api/system/gpu` / `/gpu-text` — `PlainTextResponse` one-liners for the dashboard stat tile (not JSON — the frontend polls these as raw text).
- `GET /api/system/version` — reads `__version__`/`__release_channel__` plus the short git SHA straight from `.git/HEAD` (no `git` subprocess call).

**Wired into**: `models.py::inference_memory_estimate` imports `_vram` directly from this module to attach `currently_used_gb`/`total_gpu_gb` to a memory estimate — a private-function cross-module import, acceptable since both are small, closely related modules in the same package, but worth knowing it isn't a public API.

## `routes/datasets.py` (168 lines, self-prefixed `/api/projects/{pid}/datasets`)

Project-scoped JSONL dataset registry — distinct from `data.py` (global,
unregistered files) and from `file_library.py` (raw uploaded source files,
pre-dataset-export).

- `list_datasets_route` / `get_dataset_route` — 404 correctly on missing project or on a dataset whose `project_id` doesn't match `pid`. `get_dataset_route` also **refreshes** `qa_count`/`size_bytes` on-the-fly if the file's on-disk size has drifted from the DB row (best-effort, logged on failure, never raises).
- `register_existing_route` — registers an existing on-disk file as a dataset. Accepts either `data_path` directly or a `file_id` to resolve via `project_files`/`file_versions`; explicitly checks the resolved file's owning project and returns **403** (not 404) if it belongs to a different project — correct ownership-check pattern.
- `upload_dataset_route` — multipart upload; coerces any extension to `.jsonl` and avoids clobbering existing files by appending `-2`, `-3`, ... suffixes.
- `patch_dataset_route` — only `name` is patchable (an allowlist, not a blind `**body` merge).
- `delete_dataset_route` — `remove_file=False` by default (unregisters without touching the file on disk).

## `routes/project_settings.py` (211 lines, tag `project-settings`)

One route: the Settings-tab log tail. Prefers the **systemd journal** when
`finetune-studio.service` is active (fan-dragon production), falls back to
tailing `/tmp/uvicorn.log` or a couple of other candidate paths in dev.

- `resolve_log_path()` — first existing candidate from `LOG_CANDIDATES`.
- `systemd_unit_active()` / `journal_tail()` — `systemctl --user is-active` / `journalctl --user -u ... -o cat`, both with short timeouts and `check=False` (never raise on a missing systemd).
- `build_logs_payload()` — the real contract logic: `live` is only ever `True` for a source believed to be the actual running process. A stale `/tmp/uvicorn.log` (mtime > 120s old) while systemd is active is explicitly reported `live: False, stale: True` rather than silently served as if current — this guards against showing a dead dev process's old log as if it were the live service.
- `GET /projects/{pid}/logs?lines=N` — 404 via `HTTPException` if project missing (this file is the one place in the lane that raises `HTTPException` directly instead of returning `JSONResponse`; both styles coexist across the lane by file, not inconsistently within one file).

## `routes/data_editor.py` (255 lines, mounted at `/api/data-editor`)

Project-scoped JSONL row-level editor (approve/reject/edit/delete individual
Q&A rows), with careful path-scoping to prevent a `dataset` query param from
escaping the project.

- `_has_traversal` / `_normalize_raw_path` / `_resolve_path` — reject any `..` path component, then resolve `dataset` to an absolute path 3 ways in order: (1) a path already registered in `db.datasets` for this project, (2) absolute-as-given, (3) relative under `settings.data_dir`.
- `_assert_project_scope(pid, path)` — the actual gate: allowed only if the path matches a dataset already registered to `pid`, **or** falls under `_project_root(pid)` (`{db_path parent}/projects/{pid}/`); raises 403 otherwise. This is the strictest and most correct scoping implementation in the lane — worth reusing as the reference pattern if any sibling file needs similar hardening.
- `preview` / `get_row` / `update_row` / `approve_row` / `reject_row` / `delete_row` / `review_list` / `batch_save` — all require the project to exist (404) and route every write through `db.record_review()` for an audit trail.

**Gotcha**: `update_row`/`delete_row`/`get_row` take a raw integer `index`
into the currently-loaded row list, not a stable row id — concurrent edits
from two tabs on the same dataset can silently clobber each other's index
assumptions. Pre-existing design, not introduced today; flagged for
awareness, not fixed (would need a schema change to give rows stable ids).

## `routes/updates.py` (256 lines, self-prefixed `/api/system/update*`)

Self-healing update pipeline: queues `update.sh` as a background subprocess
and streams its stdout into a DB row so the UI can show live progress.

- `trigger_update` — validates `mode` in `{update, check, repair}`, creates a DB row, queues `_update_worker` via `BackgroundTasks`.
- `update_events` — SSE stream; polls `_latest_update_payload()` every second, de-duplicates frames by a `status|log_length|error` fingerprint, caps runaway streams at 1800 idle frames (~30 min).
- `get_update_status(uid, full=False)` — status + log tail for one attempt.
- `_update_worker` — `FTS_SKIP_UPDATE=1` short-circuits with a canned log for tests; otherwise spawns `update.sh` with flags derived from `options`, streams output line-by-line into the DB. Treats a nonzero exit as success **if** `"Restarting finetune-studio.service"` was seen in the output first — the restart SIGTERMs the script itself, so `-15` there means every real step already finished, not a failure.

**Gotchas / fixed 2026-10-01**: `get_update_status`'s docstring always
promised a `?full=1` escape hatch for the full (untruncated) log text, but
the route signature never declared a `full` query parameter — it was
hardcoded `full = False` inline, making the documented feature permanently
unreachable (a silent truncation-without-a-way-out, worse than just
truncating without ever promising otherwise). Fixed by adding `full: bool =
False` to the route signature.

## `routes/versions.py` (267 lines, self-prefixed under `/api/projects/{pid}/...`)

Project version snapshots (manifest of datasets + RAG builds + training
runs at a point in time) + two related features that share this file:
hand-picked-source dataset builds, and a RAG coverage gate.

- `_save_version` — auto-fills `manifest.datasets`/`rag_corpora`/`training_runs` from current DB state when the caller doesn't pin them explicitly; validates `parent_version_id` belongs to the same project (404 if not).
- `build_subset_dataset` — builds a dataset from only the picked `source_ids`; runs `fill_sources_gaps` as a coverage gate **before** export (409 if any selected source has uncovered chunks — this blocks building a "100% facts" dataset claim from a subset that's actually missing chunks). After export, tallies `per_source` pair counts by re-reading `source_id` out of each exported JSONL line — this is an honest reconciliation step (not a silent count), logging (not swallowing) any line whose `source_id` fails to parse.
- `rag_coverage` — compares parsed sources (`qa/sources/*.json`) against the RAG manifest's `documents_meta` by filename/stem match; returns per-source `in_corpus` booleans and an overall `coverage_pct`. Imports `_corpus_dir` from `routes/rag.py` (lane B2) — a cross-lane private-function import worth knowing about if `rag.py`'s corpus layout ever changes.

## `routes/project_rag.py` (382 lines, tag `project-rag`)

Project RAG document **inventory** (read side) + whole-corpus rebuild.
Distinct from `routes/rag.py` (lane B2, build/settings/search) and from
`chat_v2.py`'s retrieval helper — this file never searches, only lists and
rebuilds.

- `list_indexed_docs(pid)` — reads `chunks.parquet` grouped by `document_id` as the primary source of per-doc chunk counts; falls back to the manifest's `documents_meta` only when the parquet is empty/missing. Explicitly does **not** scan `sources/*.txt` to invent inventory rows — only attaches size/mtime to docs already found via parquet/manifest, so a stale orphan `.txt` file left behind by a reset never resurfaces as a phantom "indexed" document.
- `list_doc_chunks(pid, doc_id)` — chunk previews (first 120 chars) for one document, sorted by `chunk_index`.
- `rag_mcp_package` — packages the corpus into a self-installing tarball/zip (MCP server + HTTP server + README) via `data.rag_portable.mcp_package.build_package`; 404 if no corpus yet, 409 on a `FileNotFoundError` from the packager (e.g. missing embedder when `include_models=true`).
- `rag_rebuild` — full-corpus rebuild via `PortableRAG.build_from_directory`; `doc_id` in the request body is accepted but only logged — **PortableRAG has no per-document vector splice**, so "rebuild this one doc" always rebuilds everything. This is documented honestly in both the `RebuildRequest` docstring and the route docstring, not silently ignored.

**Gotcha**: `rag_rebuild` with `reset=True` (the default) does
`shutil.rmtree(corpus)` before rebuilding — a rebuild that fails partway
through (e.g. embedder crash) can leave the corpus directory empty until
the next successful rebuild. The DB `rag_corpora` build-history row is
marked failed correctly via `mark_rag_build_failed`, so the *history* isn't
lost, but the on-disk corpus itself has a window of being gone. Pre-existing
behavior, not changed.

## `routes/chat_v2.py` (409 lines, mounted at `/api/chat-v2`)

Global + per-project inference chat, vision-capable, multi-RAG. This is the
file `models.py`'s `/api/inference/*` aliases point back to for some
endpoints (comment in `models.py`: "Single source of truth stays in
chat_v2").

- `project_context(pid)` — resolves the project's production model path (via `project["production_run"]` → `db.get_run()["output_path"]`), system prompt, RAG list, and currently-discovered models. Raises `HTTPException(404)` for a missing project — this route used to return `{"error": ...}` as a 200 (the fix is already landed and explicitly explained in-line, with a comment citing exactly the anti-pattern `test_project_not_found_404.py` now guards against repo-wide).
- `load_model` (`POST /load`) — **goes through the shared loader**: `resolve_loader_overrides()` from `models/llama_loader.py`, then `inference_engine.load(model_path, **kwargs)`. Also unloads whatever `ModelManager` (data-prep's helper) currently holds first, to avoid two models resident in VRAM simultaneously.
- `inference_chat`/`inference_status`/`unload_model`/`model_info`/`memory_estimate`/`inference_benchmark` — global (non-project) inference surface; `unload_model` calls `unload_all_models()` which tears down both engines (chat_v2's `inference_engine` and data-prep's `ModelManager`) — see E2E-22 in the inline comments for why both must go together.
- `chat` (`POST /projects/{pid}/chat`) — the project-scoped, multi-RAG chat. Validates the project **before** checking anything else (explicitly ordered per an inline comment explaining a past regression where checking the model first made a missing project report as "No model loaded"). Retrieves from every `enabled_rag_ids` store via `_search_rag_attachment` (PortableRAG or legacy Chroma `VectorStore`, auto-detected by `_is_portable_corpus`), deduplicates by `chunk_id`, takes top-8 by score, builds an augmented system prompt, generates.

**Verified (priority check 2)**: `load_model` in this file correctly routes
through `resolve_loader_overrides()` — no ad-hoc `Llama(...)` construction
or hand-rolled `n_gpu_layers`/`n_ctx` defaults found.

## `routes/models.py` (420 lines, mounted at `/api/models` + `/api/inference`)

Local model registry, GPU-aware load-failure diagnostics, and the
`/api/inference/*` alias surface for the dedicated Inference page.

- `_identify_process`/`_gpu_snapshot`/`_vram_hint`/`_resource_snapshot_detail` — when a model load fails, these build an actionable error message naming exactly which other process (ComfyUI, Ollama, vLLM, ...) is holding GPU memory and how much headroom is actually free, instead of a bare exception string.
- `_load_failure_payload` — the single place that shapes a load failure into `{"status": "error", "loaded": False, "error": ..., "model": None}`; HTTP 200 is kept deliberately so callers that only branch on the JSON body (not status code) still work.
- `load_model_endpoint` (`POST /load`) — accepts `path` **or** `model_path` (back-compat with the older chat_v2 body shape). **Goes through the shared loader**: `resolve_loader_overrides()` then `inference_engine.load(model_path, **overrides)`. Also unloads `ModelManager` first, same rationale as `chat_v2.load_model`.
- `inference_router` (mounted separately at `/api/inference`) — `/status`, `/chat`, `/memory-estimate` are independent implementations (not delegating to chat_v2), but `/load` and `/unload` are literal aliases: `return await load_model_endpoint(request)` / `return await unload_model_endpoint()`. The inline comment is explicit that chat_v2 "stays single source of truth" for the underlying load/unload *logic* even though two URL namespaces exist for ergonomics.
- `inference_memory_estimate` imports `_vram` from `routes/system.py` to attach live GPU usage to the estimate.

**Verified (priority check 2)**: `load_model_endpoint` correctly routes
through `resolve_loader_overrides()`. No bypass found.

## `routes/hf_models.py` (436 lines, mounted at `/api`)

HuggingFace Hub browser + background downloader (LM-Studio-style), local
favorites list, and the shared-model-pool stats endpoint. **Never loads a
model for inference** — downloads only — so priority check 2 (shared
loader) does not apply to this file.

- `_search_hf` — fixes two real Hub-search gaps inline (documented in the docstring): free-text queries now require every whitespace-split token to be a substring of the repo id (was: the whole query had to literally match), and a too-strict `pipeline_tag` filter retries once without the filter if it yields zero results (catches `conversational`-tagged chat models that aren't tagged `text-generation`).
- `hf_download`/`_download_worker` — background job via `huggingface_hub.snapshot_download`/`hf_hub_download` into `~/.finetune-studio/hf_models/<repo_id with "/" → "__">`; on completion calls `_refresh_model_registry()` so the new model shows up without a manual refresh click.
- `restore_in_progress_downloads()` — called at startup (from `app.py`, not in this lane) to repopulate the in-memory `_DOWNLOADS` dict from the DB. A job that was `downloading` when the process died is explicitly flipped to `error` ("service restarted while downloading") rather than silently left `downloading` forever or silently resumed as if nothing happened.
- `list_favorites` — explicitly catches and logs any exception, always returns `[]` rather than ever 500ing the page (docstring: "Never 500 the page").

**Gotcha**: `_DOWNLOADS` is a plain process-wide module-level dict with no
locking — fine for the FastAPI single-process deployment this app targets,
but would race under multiple worker processes.

## `routes/exports.py` (445 lines, self-prefixed under `/api/projects/{pid}/...`)

GGUF / abliterated / merged-safetensors export, driving llama.cpp's
`convert_hf_to_gguf.py` + `llama-quantize`.

- `export_run` — two modes in one handler, selected by `use_sync` (true when the body has `quants`, or `format` is `abliterated|merged|awq|gptq`, or `force` is set without a singular `quant`): **sync** multi-format path (used by the Export page) calls `export_trained_run()` directly and blocks until done; **legacy async GGUF** path (singular `quant`) queues a `BackgroundTasks` job and returns an `export_id` to poll.
- Ownership check: `run.get("project_id") and pid and run["project_id"] != pid` → 400 "run does not belong to this project" (400, not 404 — a minor inconsistency with the 404-for-foreign-resource convention established elsewhere in this lane today, but pre-existing and outside this audit's fix list since it already correctly refuses the operation rather than leaking data).
- **Verified intact (per task brief)**: the multi-quant GGUF registration fix from earlier today is still correct — when `per_quant` is true (`fmt == "gguf"` and `len(payload.files) == len(payload.quants)`), the code does `zip(payload.quants, payload.files, strict=True)` and calls `db.create_export()` + `db.mark_export_done()` **once per quant/file pair**, each sized from `os.path.getsize(file_path)` of that specific file — not a single row hardcoded to `quants[0]` with the combined directory size. `strict=True` means a length mismatch between `quants` and `files` raises immediately rather than silently zipping short.
- `_export_worker` — background GGUF conversion; refuses to call `mark_export_done` if the output file is missing or zero bytes ("Refuse to mark export done without a non-empty artifact").

## `routes/projects.py` (562 lines, mounted at `/api/projects`)

The core Project/RAG/Run CRUD surface — every Project "owns" N RAGs and N
Training Runs, and this file is where that ownership is supposed to be
enforced.

- Project CRUD: `list/create/get/update/delete`, `export_project` (streams a `tar`/`tar.gz` of the project's data + RAG corpus dirs plus a `manifest.json`), `import_project` (extracts an uploaded archive, creates a **new** project per `projects/<old_id>/` dir found inside, copies files — note: re-imports always mint new ids, never overwrite an existing project by its old id).
- RAG CRUD + `ingest_into_rag` (file/dir → `RAGManager`, tracks both a live `project_rags` summary row and a `rag_corpora` build-history row) + `query_rag` (raw chunk search, no generation) + `rag_stats`.
- Run CRUD + `start_run` (builds a `TrainingConfig`, guards against every run sharing the literal `output` dir — rewrites to `output/projects/{pid}/runs/{rid}` — then `training_engine.start()`), `stop_run`, `run_benchmark` (loads the run's output model into a throwaway `InferenceEngine`, always unloads it in a `finally` even on a suite crash), `merge_run` (idempotent unless `?force=true`).

**Fixed 2026-10-01 — cross-project ownership was not enforced on any
run/rag sub-resource route.** Before today, every route of the shape
`/{pid}/runs/{rid}/...` or `/{pid}/rags/{rid}/...` fetched the resource by
`rid` alone and never checked it actually belonged to `pid`:
`get_run`/`update_run`/`delete_run`/`start_run`/`stop_run`/`run_benchmark`/
`merge_run`/`update_rag`/`delete_rag`/`ingest_into_rag`/`query_rag`/
`rag_stats` all accepted **any** valid `rid`/`rag id regardless of which
project's URL it was reached through** — a run or RAG id from Project B was
fully readable, startable, stoppable, deletable, and mergeable via
`/api/projects/{project-A-id}/runs/{project-B-run-id}/...`. `create_rag`/
`create_run` also never checked the parent project existed at all, so a
bogus `pid` could still mint an orphan row. Only `promote_run` and
`export_run` (this lane) and `model_export_contents` (`project_models.py`,
this lane) already had the correct ownership-check pattern — `exports.py`
and `project_models.py` were the reference implementations copied here.

Fixed by adding two shared helpers, `_get_owned_run(pid, rid)` and
`_get_owned_rag(pid, rid)`, both returning `(resource, None)` on success or
`(None, 404 JSONResponse)` otherwise, and wiring every listed route through
them (`create_run`/`create_rag` got a plain `_project_404` existence check
instead, since there's no child resource yet to own). `promote_run` was
additionally fixed to return a real 404 status instead of `{"error": ...}`
as a 200 (same anti-pattern `test_project_not_found_404.py` already guards
against elsewhere; `tests/test_project_training.py::
test_promote_unknown_run_errors` was updated to assert 404 instead of 200
since it had pinned the old, wrong behavior).
`tests/test_webui_routes_core_audit.py` pins the new 404-for-foreign-owned
behavior across all of these routes, plus a same-project 200 case so the
fix doesn't overcorrect into 404-everything.

**Still worth knowing**: `export_run` (`exports.py`, this lane) returns
**400** rather than 404 for a run that belongs to a different project — it
already refuses the operation (no data leak), just with a different status
code than the convention `projects.py` now follows. Not changed, since
`exports.py`'s own behavior wasn't broken and changing its status code was
outside today's run/rag ownership fix.

## `routes/file_library.py` (537 lines, self-prefixed under `/api/projects/{pid}/...`)

The modern file-upload/library system: streamed hashing upload, dedup by
sha256, versioning, trash/restore/purge, folders, bulk actions, background
parsing. The cleanest file in the lane — every single route calls
`_project_or_404(pid)` first, and every file-scoped operation delegates to
`data.fs.file_library` (`fl`) functions that are themselves written to take
`(pid, fid)` together, so project scoping is enforced at the data layer, not
just the route layer.

- `_stage_upload` — streams the multipart body to a temp file in `.uploads/` in 8MB chunks while hashing it with sha256, so a huge upload never has to live fully in memory before being written.
- `upload_files` — per-file try/except so **one bad file in a bulk upload never aborts the rest**; returns a structured `{uploaded, duplicates_skipped, errors}` count plus a per-file `report` list. Dedup is by exact sha256 match within the project; a duplicate still gets staged as a data-prep source if useful (e.g. uploading the same PDF under a different folder).
- Route ordering note called out explicitly in the module docstring: FastAPI matches in registration order, so `/files/trash` and `/files/trash/purge` **must** be declared before the generic `/files/{fid}` catch-all or they'd be swallowed as `fid="trash"` — confirmed this ordering is in fact preserved in the file (trash routes at L280-290, pipeline/bulk/zip/search routes at L294-336, generic `{fid}` routes starting L341).
- `download_raw_route` — works for files already in trash (`include_deleted=True`) so a user can recover+inspect before permanently purging; 410 (not 404) if the DB row exists but the bytes are missing on disk — correctly distinguishes "gone from the index" from "index says it should be here but isn't."

**Verified (priority check 3)**: every `{fid}`/`{folder_id}` route in this
file 404s through `fl.get_file(pid, fid)` / `fl.*` helpers scoped by
`(pid, ...)` — no cross-project leak pattern found here, unlike
`projects.py`.

---

## Cross-module findings — NOT fixed, needs parent coordination

1. **Three-to-four independent RAG-chat implementations**, at minimum:
   `comparison.py::rag_chat` (`/api/compare/rag/chat`, this lane, global
   `VectorStore`), `chat_v2.py::chat` (`/api/chat-v2/projects/{pid}/chat`,
   this lane, multi-RAG PortableRAG/Chroma), `projects.py::query_rag`
   (`/api/projects/{pid}/rags/{rid}/query`, this lane, raw search only, no
   generation), and almost certainly `routes/rag.py::rag_chat` (lane B2,
   **not read** — file list shows a `rag_chat(pid, req)` at L588 per
   `docs/CODEMAP.md`). None share retrieval or generation code. Consolidating
   would require editing `routes/rag.py`, which is outside this lane's file
   list — flagging for whichever lane owns `rag.py` plus parent
   coordination on which implementation should become canonical.

2. **`versions.py::rag_coverage`** (this lane) imports `_corpus_dir` from
   `routes/rag.py` (lane B2) as a private cross-module dependency —
   `src/finetune_studio/webui/routes/versions.py:231` imports from
   `src/finetune_studio/webui/routes/rag.py` (function at its L38 per
   `docs/CODEMAP.md`). Not a bug, just a coupling point: if lane B2 changes
   `rag.py`'s corpus directory layout, this lane's coverage gate breaks
   silently until someone greps for callers.

3. **`exports.py::export_run`** (this lane) returns HTTP **400** (not 404)
   for a run belonging to a different project
   (`src/finetune_studio/webui/routes/exports.py:147-155`), while
   `projects.py` (fixed today, same lane) now consistently returns 404 for
   the same class of foreign-resource access. Both already refuse the
   operation — no data leak in either — this is a status-code consistency
   gap, not a security issue. Left unchanged since it wasn't broken and
   fixing it was outside today's specific run/rag-ownership fix in
   `projects.py`; flagging in case a future pass wants one consistent
   convention project-wide.
