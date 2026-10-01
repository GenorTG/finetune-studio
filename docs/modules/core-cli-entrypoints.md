# Core / CLI / Entrypoints (Lane F)

Covers the application's glue layer: the FastAPI app composition root and
its non-route helpers (`webui/`), the `fts` command-line tool (`cli/`), the
two top-level entry-point shims (`cli.py` — removed, see below — and
`__main__.py`), package-wide settings/naming/versioning (`config.py`,
`naming.py`, `__init__.py`), and the Jinja2 chat-template engine
(`templates/`).

---

## Entry points: `cli.py`, `__main__.py`, `cli/` package

**`src/finetune_studio/__main__.py`** (18 lines) — makes `python -m
finetune_studio` work. Imports `main` from `finetune_studio.cli` and calls
it under `if __name__ == "__main__"`.

**`src/finetune_studio/cli/__init__.py`** (18 lines) — the real
`finetune_studio.cli` package. Re-exports `COMMANDS` and `main` from
`_registry.py`, and defines `cli_main()` as an alias for `main()`.

**`src/finetune_studio/cli/__main__.py`** (6 lines) — makes `python -m
finetune_studio.cli` work directly (as opposed to `python -m
finetune_studio`, which goes through the top-level `__main__.py` first).

**`src/finetune_studio/cli/_parser.py`** (215 lines) — the single
`argparse.ArgumentParser` definition for every `fts` subcommand
(`models`, `train`, `test`, `suite`, `validate`, `convert`, `webui`, `rag`
+ 5 sub-subcommands, `compare`, `benchmark`, `analyze`, `augment`,
`optimize`, `validate-hallucination`, `rag-test`, `vram` + 3
sub-subcommands, `files` + `trash` sub-subcommand). All flags and defaults
for the CLI live here and nowhere else — `argparse` builds subparsers by
side effect on one parent parser, so splitting this across files isn't
practical.

**`src/finetune_studio/cli/_registry.py`** (61 lines) — imports every
`cmd_<name>` handler from `cli/commands/`, maps them in the `COMMANDS`
dict keyed by subcommand name, and defines `main()`: build the parser,
parse `sys.argv`, print help and exit 0 if no subcommand given, otherwise
dispatch to `COMMANDS[args.command](args)`.

**`src/finetune_studio/cli/_vram_print.py`** (23 lines) — `print_vram_check()`,
a pretty-printer for a `VRAMEstimate` (from `training/vram/`), used by
`fts vram check`.

### `cli.py` vs `cli/` — dead code, removed

The repo previously had **both** `src/finetune_studio/cli.py` (a 10-line
file) **and** `src/finetune_studio/cli/` (the package above) in the same
parent directory. This is legal to have on disk, but in CPython's import
resolution a regular package (a directory with `__init__.py`) **always**
wins over a same-named `.py` module file in the same parent package —
verified empirically:

```python
>>> import finetune_studio.cli as m
>>> m.__file__
'.../src/finetune_studio/cli/__init__.py'   # never cli.py
```

`pyproject.toml`'s console-script entry points (`finetune-studio =
"finetune_studio.cli:main"`, `fts = "finetune_studio.cli:main"`) therefore
always resolved to the package too. `cli.py` was **completely
unreachable** — no import path in Python could ever load it — despite its
own docstring claiming to be "the real code" location's back-compat shim
("Old: `from finetune_studio.cli import main` ... still works"). That
claim was accidentally true, but for a reason unrelated to the shim's own
existence (the package provides it regardless). `cli.py` has been
**deleted** as part of this audit; see
`tests/test_core_entrypoints_audit.py` for a regression pin (file absence
+ empirical import-resolution check + `python -m finetune_studio[.cli]
--help` smoke tests).

---

## `cli/commands/*.py` — one file per subcommand handler

Each file exports exactly one `cmd_<name>(args) -> None` function, called
from `_registry.COMMANDS`. All of them lazy-import their heavy
dependencies (torch/transformers-adjacent modules) inside the handler
function body, not at module top level, so `fts --help` stays fast.

| File | Subcommand | Delegates to |
|---|---|---|
| `analyze.py` (46 l) | `fts analyze` | `training.data_quality.DataQualityAnalyzer` |
| `augment.py` (56 l) | `fts augment` | `training.data_augmentation.DataAugmenter` + `data_quality.DataQualityAnalyzer` |
| `benchmark.py` (79 l) | `fts benchmark` | `benchmarks.real_benchmarks.RealBenchmarkSuite` + `testing.inference.InferenceEngine` |
| `compare.py` (56 l) | `fts compare` | `benchmarks.comparison.comparator` (module-level singleton) + `testing.suite` |
| `convert.py` (32 l) | `fts convert` | `data.converter.{csv_to_jsonl,json_to_jsonl,jsonl_to_json}` |
| `files.py` (100 l) | `fts files trash` | `data.fs.file_library.{list_trash,purge_trash}` + `db` |
| `models.py` (24 l) | `fts models` | `models.registry.scan_models` |
| `optimize.py` (40 l) | `fts optimize` | `training.config_optimizer.TrainingConfigOptimizer` |
| `rag.py` (64 l) | `fts rag {ingest,query,list,remove,stats,clear}` | `rag.manager.RAGManager` |
| `rag_test.py` (42 l) | `fts rag-test` | `rag.query.RAGQuery` + `rag.store.VectorStore` + `testing.inference.InferenceEngine` |
| `suite.py` (47 l) | `fts suite` | `testing.inference.InferenceEngine` + `testing.suite.{load_test_suite,run_suite,score_results}` |
| `test.py` (38 l) | `fts test` | `testing.inference.InferenceEngine` (interactive REPL loop) |
| `train.py` (51 l) | `fts train` | `training.engine.{TrainingConfig,TrainingEngine}` + `training.data.load_jsonl` |
| `validate.py` (14 l) | `fts validate` | `data.validator.validate_file` |
| `validate_hallucination.py` (31 l) | `fts validate-hallucination` | `training.hallucination_guard.TrainingDataValidator` |
| `vram.py` (113 l) | `fts vram {report,check,profile}` | `training.vram_profiler` (shim, see below) |
| `webui.py` (10 l) | `fts webui` | `uvicorn.run("finetune_studio.webui.app:app", ...)` |

`cli/commands/__init__.py` (7 lines) is docstring-only — no re-exports.

### CLI/WebUI drift check (Priority Check #4)

Spot-checked `train.py`, `benchmark.py`, `suite.py`, `rag.py`,
`rag_test.py`, `vram.py`:

- **`train.py` does NOT reimplement training.** It builds a
  `training.engine.TrainingConfig` and drives a **fresh**
  `training.engine.TrainingEngine()` instance — the exact same class the
  webui's `webui/routes/training.py` uses. The CLI's instance and the
  webui's shared `training_engine` singleton (defined in `webui/app.py`,
  see below) are two different *objects* of the same class, which is
  correct and expected: the CLI runs as its own standalone process, so it
  cannot and should not share the webui process's singleton.
- **`vram.py` imports from `training.vram_profiler`**, which is itself
  confirmed to be a legitimate back-compat shim (its own docstring says
  so) re-exporting from `training.vram/` (the real package) — this is the
  same pattern the repo already uses correctly elsewhere, unlike the
  broken `cli.py` case above.
- **`benchmark.py`, `suite.py`, `rag_test.py`, `test.py`** each construct
  their own `testing.inference.InferenceEngine()` instance, same class the
  webui uses for its shared `inference_engine` (see `webui/app.py`
  below) — again correct for a standalone CLI process, not drift.

No CLI/webui logic duplication was found; the CLI is a thin argument-
parsing + printing layer over the same engine/manager classes the webui
calls into.

---

## `webui/app.py` (368 lines) — FastAPI app composition root

The actual ASGI app (`app = FastAPI(...)`) and its `lifespan` startup
hook. Responsible for:

1. **Process-wide singletons** (module level, created at import time):
   - `training_engine = TrainingEngine()` — the one `TrainingEngine` the
     webui process uses for live training runs. `webui/routes/training.py`
     imports this exact object (`from finetune_studio.webui.app import
     training_engine`) rather than constructing its own.
   - `inference_engine = get_manager().engine` — the one *persistent*
     `InferenceEngine` the webui process uses for interactive
     testing/chat. `ModelManager` (in `models/manager.py`, owned by
     another lane) owns the actual engine object via its `.engine`
     property; `app.py` just holds a reference to it. Several routes
     (`webui/routes/{models,rag,comparison,testing,chat_v2}.py`) and
     `engine_guard.py` depend on this name resolving to the exact same
     object `ModelManager` uses internally, so a model load through one
     path is visible to generate calls through any other path.
   - **This audit verified** `engine_guard.py` does *not* construct a
     second competing engine — it only defines an `asyncio.Lock`
     (`ENGINE_LOCK`) used to serialize calls against the one
     `inference_engine` above. See `engine_guard.py` below.
   - A short-lived **second** `InferenceEngine()` is legitimately
     constructed in a few other webui routes outside this lane
     (`webui/routes/benchmarks.py` three call sites,
     `webui/routes/projects.py` one call site) for one-shot
     benchmark/judge runs. The comment above `inference_engine` in
     `app.py` was corrected during this audit (see Discrepancies below)
     to describe the actual contract: a second engine is fine for a
     one-shot operation **only if** the caller frees the persistent
     engine's VRAM first (`models.llama_loader.unload_all_models()`, see
     `benchmarks.py`'s `_unload_global_inference()` helper) and unloads
     its own temp engine when done. One of the four call sites
     (`webui/routes/projects.py:480`, in `run_benchmark`) does **not**
     call that unload helper before loading — flagged below as a
     cross-module finding, not fixed here (file outside this lane).

2. **`lifespan` async context manager** — runs once at startup (and once
   at shutdown, but there's no teardown logic after the `yield`):
   scans `settings.model_dirs` (+ `model_dirs_extra`) for models, calls
   `db.init_db()`, then four best-effort recovery steps each wrapped in
   its own `try/except Exception: pass` so a failure in one doesn't block
   the others or the app from starting:
   - `restore_in_progress_downloads()` — re-attach to HF downloads that
     were in flight when the process last exited.
   - `db.reconcile_stale_updates()` / `db.reconcile_stale_runs()` — mark
     `system_updates`/`training_runs` rows left in a `loading`/`training`/
     `saving` state by a hard kill as failed/cancelled, and log the count.
   - `resume_stale_data_prep_runs()` — resume (not fail) data-prep runs
     interrupted mid-flight, since each run's project/source/settings
     persist durably; logs resumed vs. failed counts.
   - `db.reconcile_stale_rag_builds()` / `db.reconcile_stale_exports()`.

3. **Activity-feed middleware** (`record_activity_operations`) —
   wraps every `POST`/`PUT`/`PATCH`/`DELETE` request (except
   `/api/activity*` and anything ending `/events`) and persists a
   best-effort `db.record_activity_event(...)` row after the response
   (or after a 500 on exception, which is re-raised). `_activity_kind()`
   classifies the path into a bucket (`model_load`, `inference`,
   `rag_build`, `rag_query`, `upload`, `testing`, `benchmark`, `export`,
   `training`, `download`, `data_prep`, `system_update`, or the
   `"operation"` fallback) via ordered `if` checks — order matters, e.g.
   model load/unload/refresh is checked before the broader
   `/api/inference/` prefix so it isn't swallowed by it.
   `_activity_summary()` screen-scrapes the path for a human one-liner
   (e.g. "Start training run"). Both classifier functions and
   `_record_activity_event` itself swallow all exceptions by design —
   activity logging must never break the actual API response — but note
   this means a bug in the classifier logic fails silently with no log
   line; there's no telemetry on how often that `except Exception: pass`
   in `_record_activity_event` actually fires.

4. **CORS / trusted-host / proxy-header middleware**
   (`_apply_hosting_middleware`) — reads user-editable
   `~/.finetune-studio/settings.json` (`cors_origins`,
   `cors_allow_credentials`, `proxy_headers`) and conditionally installs
   `CORSMiddleware` / `ProxyHeadersMiddleware`. Runs once at import time,
   not per-request — changing `settings.json` requires a process restart
   to take effect.

5. **Static files** — mounts `/static` from `webui/static/` via a custom
   `_NoCacheStatic(StaticFiles)` subclass that sets
   `Cache-Control: no-cache` + `Pragma`/`Expires` headers on `.css`/`.js`
   responses only (images/fonts keep default browser caching), so UI
   fixes apply without a hard refresh.

6. **Router registration** — imports and `include_router()`s every
   route module under `webui/routes/` (owned by other lanes), each with
   its documented prefix. Several routers self-prefix their full path
   (`data_prep`, `hf_models`, `system`, `exports`, `updates`, `activity`,
   `versions`, `settings`, `datasets`, `file_library`, `project_models`,
   `project_rag`, `project_settings`) and are registered with a comment
   noting their real mount point — this is intentional per the repo's
   "routes self-prefix their full path" convention, not a bug.

---

## `webui/engine_guard.py` (25 lines)

Defines exactly one thing: `ENGINE_LOCK = asyncio.Lock()`. Documents (and
enforces, by convention — not by any runtime check) the pattern every
route handler must follow when touching the shared `inference_engine`:

```python
async with ENGINE_LOCK:
    result = await asyncio.to_thread(inference_engine.generate, ...)
```

Rationale in the module docstring: llama.cpp/transformers generation
isn't safe to call concurrently on the same model instance, and running
blocking engine ops inline (without `to_thread`) previously froze the
entire event loop (SSE, status polls, navigation) for the whole duration
of a load/generate call. **Confirmed during this audit**: this module
does *not* construct any `InferenceEngine`/`ModelManager` object itself —
it is purely a lock, not a second engine. 5 route modules
(`rag.py`, `comparison.py`, `models.py`, `testing.py`, `chat_v2.py`)
import and use `ENGINE_LOCK`.

---

## `webui/__init__.py` (1-line docstring, no exports)

Just marks `webui/` as a package. No re-exports, no side effects.

---

## `webui/live_sse.py` (41 lines)

Small SSE (Server-Sent Events) toolkit so long-running surfaces (training,
activity, testing, export) can stream JSON snapshots instead of polling
every 2s:

- `SSE_HEADERS` — `Cache-Control: no-cache`, `X-Accel-Buffering: no`,
  `Connection: keep-alive` (defeats proxy/nginx buffering).
- `sse_data(payload)` — formats one `data: <json>\n\n` frame
  (`json.dumps(..., default=str)` so non-JSON-native values like
  `datetime` don't crash the stream).
- `sse_comment(text="keepalive")` — a `: <text>\n\n` comment frame,
  ignored by `EventSource` but keeps the connection from timing out.
- `sse_response(generator)` — wraps an async generator in a
  `StreamingResponse` with `media_type="text/event-stream"` and the
  headers above.

Used by `webui/app.py` (`sse_response(training_events(training_engine))`
for the training SSE route is actually wired in `webui/routes/training.py`,
not `app.py` itself — `app.py` only re-exports the router).

---

## `webui/model_labels.py` (39 lines)

One function: `model_label(path: str | None) -> str` — renders a short
display name for a model path or HF repo id. Three cases, checked in
order:
1. HF hub cache path (`.../models--<org>--<repo>/snapshots/<hash>`) →
   `<org>/<repo>` via regex `models--(.+?)/snapshots/`.
2. Plain `org/repo` HF id (no leading `/`, exactly one `/`, doesn't start
   with `.`) → returned as-is.
3. Otherwise → the last non-empty path segment.

Single consumer: `webui/routes/pages.py`. Not to be confused with
`naming.py`'s `short_base()`/`display_for_path()` (below), which build a
*fuller* canonical display name (project · base · version · kind · quant)
for training-run artifacts specifically — `model_label()` here is a
simpler, path-shape-only label for the model-discovery list. These two
are not duplicates: they serve different UI surfaces with different
inputs (raw model-registry paths vs. project/run-scoped artifact paths)
and neither calls the other.

---

## `webui/project_dashboard.py` (305 lines)

Pure helper functions (no FastAPI routes) that build the data for the
project overview dashboard template. Called from `project_detail_page` in
`webui/routes/pages.py` after `_project_ctx` has already attached
rags/runs/datasets/models to the project dict.

Key functions:
- `format_relative(ts, *, now=None)` — Unix timestamp → `"3h ago"` /
  `"2d ago"` / `"—"` for `None`/non-numeric/`<=0`.
- `truncate_description(text, *, limit=120)` — returns
  `{short, long, full}`; truncates on the last space before `limit` so
  words aren't cut mid-word, falls back to a hard cut if there's no space.
- `run_status_counts(runs)` — tallies `done`/`running`/`failed` from each
  run's `status` field (case-insensitive), ignoring any other status
  value (e.g. `"queued"`) — those runs are silently excluded from all
  three buckets, which is correct for a 3-bucket summary stat but worth
  knowing if a 4th status is ever added upstream.
- `models_size_gb(models)` — sums `size_gb` across model dicts, coercing
  non-numeric/missing values to 0 via `except (TypeError, ValueError):
  continue`.
- `recent_runs(runs, *, limit=5)` / `recent_models(models, *, limit=5)` —
  sort-and-slice by `started_at`/`created_at` or `mtime` respectively,
  descending, with the same safe-float coercion pattern.
- `build_activity(*, runs, models, files, pid, limit=10, now=None)` —
  merges run start/done/failed events, benchmark events (nested under each
  run's `benchmarks` list), model-export events, and file-upload events
  into one chronological timeline, sorted descending by timestamp, sliced
  to `limit`, each event enriched with a human `"when"` string via
  `format_relative`. Note: `status == "failed" and started and not
  finished` is handled as a separate branch from `status == "failed" and
  finished` (one event uses `started` as its timestamp, the other
  `finished`) — a failed run with neither `started` nor `finished` set
  produces **no** activity event at all (silently dropped, by design —
  there's nothing to time-order it by).
- `resolve_production_run(project, runs=None)` — resolves the project's
  `production_run` id to `{id, name}` for header pills; falls back to
  `run_id[:8]` as the name if the run isn't found in the provided list
  (still returns a non-`None` dict in that case, not `None`).
- `build_dashboard_ctx(project, pid, *, files=None, now=None)` — the
  top-level entry point; assembles all of the above into the dict
  `project.html` actually renders.

---

## `webui/project_data_browser.py` (80 lines)

Pure helpers for the project file-browser page (`/projects/{pid}/data`).
Imports `truncate_description` from `project_dashboard.py` (reused, not
duplicated).

- `format_bytes(n)` — humanizes a byte count (`B`/`KB`/`MB`/`GB`), coerces
  `None`/non-numeric to `0`.
- `file_browser_stats(files)` — counts active vs. trashed files (by
  presence of `deleted_at`) and sums `size_bytes` across **all** rows
  (active + trash) into one `storage_bytes` total — i.e. trashed files
  still count toward the displayed storage figure until they're purged,
  which is intentional (trash still occupies disk) but worth knowing if a
  "storage used" stat elsewhere in the UI only counts active files.
- `runs_started_max(runs)` — latest `started_at` across runs, for drift
  badges.
- `build_file_browser_ctx(project, *, files=None)` — assembles template
  extras for `project_data.html`.

Called from `project_data_page` in `webui/routes/pages.py`.

---

## `webui/testing_models.py` (76 lines)

Resolves which on-disk model exports are usable from the Testing tab,
scoped to a project's own runs (not the global HF-discovery model list).

- `is_run_done(run)` — `status` in `{"done", "completed"}`
  (case-insensitive) — both spellings are accepted because different
  callers/versions of the training engine have historically written
  either one.
- `resolve_latest_merged_model(pid, *, list_runs_fn=None)` — walks a
  project's runs (assumed newest-first, as returned by `db.list_runs`),
  returns the first completed run's `<output_path>/merged` dir that
  passes `training.run_export.merged_dir_ready()` (i.e. actually has
  weight files, not just an empty directory). `list_runs_fn` is
  injectable for tests; defaults to `db.list_runs` via a lazy import.
- `models_for_testing_page(project_models)` — filters project export rows
  to `{"safetensors", "gguf"}` formats with a non-empty `path`, sorted by
  `mtime` descending (newest first).
- `default_model_path_for_testing(models)` — prefers a `safetensors`
  export whose path ends in `merged` (trailing slash stripped first);
  falls back to the first item in the already-sorted list, or `None` for
  an empty list.

Called from `webui/routes/testing.py` and `webui/routes/pages.py`.

---

## `webui/thinking.py` (48 lines)

One function: `split_thinking(raw: str) -> dict` — splits Qwen3-style
`<think>...</think>` reasoning out of a raw model response into
`{"thinking": str, "response": str}`. Three cases:
1. A **complete** `<think>...</think>` block anywhere in `raw` (matched
   non-greedily inside, greedily/DOTALL across the whole string via
   `_FULL_RE.search`, so only the **first** complete block is extracted —
   a second `<think>` block later in the same response, if a model ever
   produced one, is left untouched inside `response`) → its contents
   become `thinking`, everything else (before + after, concatenated)
   becomes `response`.
2. An **unterminated** `<think>` opener (no closing tag yet — mid-stream)
   → the entire remainder is `thinking`, `response` is `""`.
3. Neither → `response` is the whole (stripped) string, `thinking` is
   `""`.

Explicitly designed to handle partial/streaming tokens (case 2) so a
streaming UI doesn't show a stray `<think>` tag as literal text while the
model is still reasoning. Used by `webui/routes/{models,rag,chat_v2}.py`.

---

## `naming.py` (191 lines)

Canonical **display**-name construction for training-run artifacts (per
project convention, documented in the module's own header: every artifact
should render as `<Project> · <Base model>[ · v<N> <label>][ · <Kind>][ ·
<QUANT>][ · abliterated]` instead of a bare hash like `6f64c46a/merged`).

Key functions:
- `detect_quant(name_or_path)` — regex-extracts a quantization token
  (`Q4_K_M`, `BF16`, `F16`, etc.) from a path/name, stripping the file
  extension first so `.gguf` can't be mistaken for a second quant token.
- `detect_abliterated(name_or_path)` — case-insensitive substring match
  for `"abliterat"`.
- `short_base(name_or_path)` — strips HF-cache/export-dir noise
  (`models--org--repo/snapshots/<hash>`, `org__repo`, `snapshots`/`hub`/
  `models` path segments, hash-looking segments, known uploader-name
  prefixes like `unsloth-`/`TheBloke-`) down to a clean base model name,
  also stripping any quant/abliteration tokens (those belong in the
  modifier tail per the naming scheme, not the base name).
- `kind_label(dirname)` — maps known export-dir names (`merged`,
  `abliterated` → `"merged"`, `adapter` → `"LoRA adapter"`, `gguf` →
  `"GGUF"`, `gptq` → `"GPTQ"`, `export` → `"export"`, `model` → `""`) to
  their display label; unknown dirnames pass through as-is.
- `model_full_name(...)` — joins the non-empty parts with `" · "`.
- `_version_label_for_run(project_id, run_id)` — DB-aware: looks up
  whether a pinned project version's manifest references this run id
  (prefix-matched both directions), returns `"v<N> <label>"` if so, else
  `""`. Swallows all exceptions (display helper must never raise).
- `resolve_run_path(path)` — parses a
  `projects/<pid>/runs/<rid>/<kind>` path via regex, then enriches it
  from the DB (project name, run's base model, version label, quant,
  abliteration flag). Returns `None` if the path doesn't match the
  expected shape at all.
- `display_for_path(path, size_hint="")` — the main public entry point.
  Tries `resolve_run_path` first (DB-aware, fullest name); falls back to
  a path-only best-effort name (last path segment, stripped of
  quant/abliteration tokens) if the path isn't a recognized run-export
  path or the DB lookup produced nothing.

**Priority Check #3 (naming-collision silent-overwrite risk) — not
applicable to this file.** `naming.py` is **read-only / display-only**: it
never writes a file, never decides an output directory name, and never
creates or renames anything on disk. The *actual* identity of every
artifact it describes is the project/run UUID baked into the filesystem
path (`projects/<pid>/runs/<rid>/...`), which is generated and guaranteed
unique elsewhere (training/run-persistence code, owned by another lane,
not touched here). Two runs that happen to produce the *same displayed
string* (e.g. two runs both named "v1" with the same base model) are
still stored under distinct `<rid>` directories and remain
distinguishable in the DB — the cosmetic name collision has no
data-loss consequence. No fix needed; documenting this as a verified
non-issue per the audit's explicit priority check.

---

## `config.py` (65 lines)

Two plain (non-Pydantic, despite the module docstring's "Pydantic
BaseModel" mention — see discrepancy note below) `@dataclass`es:

- `RAGSettings` — `store_path`, `enabled`, `chunk_size`, `chunk_overlap`,
  `min_score`, `documents_path`, `embedding_model`.
- `Settings` — `host`, `port`, `debug`, `model_dirs` (defaults to
  `["models", "output"]`), `model_dirs_extra` (defaults to three
  `~`-prefixed paths: HF hub cache, HF-explorer pull destination, shared
  embedder/reranker cache), default training hyperparameters
  (`default_lora_rank=64`, `default_lr=8e-5`, `default_epochs=4`,
  `default_batch_size=2`, `default_max_seq_length=2048`), `data_dir`,
  `db_path`, `rag_store_path`, `rag_embedding_model`, and a nested
  `rag: RAGSettings`.
- Module-level singleton: `settings = Settings()`.

Note the top-level `rag_store_path`/`rag_embedding_model` fields on
`Settings` duplicate `RAGSettings.store_path`/`embedding_model` under
different names — both exist simultaneously and nothing in this file
keeps them in sync. This is a latent "two sources of truth" risk (if a
caller mutates `settings.rag.store_path` expecting it to also affect
`settings.rag_store_path`, it won't) but this audit found no caller in
this lane's files that reads the stale duplicate incorrectly — flagging
here for awareness rather than fixing, since fixing it (collapsing to one
field) would require checking every caller across the whole repo,
which spans other lanes' files.

**Module docstring discrepancy (not fixed — see Discrepancies section):**
the docstring claims "Pydantic BaseModel... Frozen dataclass: immutable
config that can't be accidentally modified," but the actual code uses
plain `@dataclass` (mutable, no `frozen=True`, no Pydantic import
anywhere in the file). `settings.port = 1234` would work fine at runtime.
Left as a documentation-only note rather than a code fix since correcting
"what it should say" without knowing whether immutability was an
abandoned intent (vs. just stale prose) risks asserting something not
verified from the code; flagged under Discrepancies for the reader's
attention.

---

## `__init__.py` (package root, 46 lines after fix)

Package-level docstring describing `finetune_studio` as the application
layer (training, benchmarks, compare, webui, rag, data) sitting on top of
`templates`. Defines:

- `_BASE_VERSION = "0.1.0"`, `__release_channel__ = "EARLY BETA"`.
- `build_version()` — reads a repo-root `VERSION` file (two candidate
  paths tried: `parents[2]` and `parents[1]` relative to this file, to
  handle both an editable/source checkout and a wheel-install layout),
  returns its stripped contents if non-empty, else falls back to
  `f"{_BASE_VERSION}.0"`. The `VERSION` file is described as
  "auto-bumped by the pre-commit hook so every commit is a new version."
- `__version__ = build_version()` — computed once at import time (not
  re-read per call), consumed by the UI version chip and
  `/api/system/version`.

---

## `templates/__init__.py` (29 lines after fix)

Public API surface for the `templates` subpackage. Re-exports from
`manager.py` (`DEFAULT_TOOLS`, `ChatTemplate`, `TemplateManager`) and
`renderer.py` (`extract_template_from_gguf`, `render_chat`,
`render_with_model_template`), with an explicit `__all__`. Its own
docstring states `renderer.py` is "the SINGLE SOURCE OF TRUTH for Jinja
template rendering" and instructs callers to always import from this
package, not a copy — **confirmed during this audit**: `grep` across
`src/` found no second `class TemplateManager`/`class ChatTemplate`
definition anywhere in the repo.

---

## `templates/manager.py` (174 lines after fix)

Higher-level interface over `renderer.py`: "render messages for model X"
by name, with per-model template caching.

- `ChatTemplate` (dataclass) — `name`, `template`, `tool_format`
  (`"generic"` by default), `supports_tools`, `custom_system_prompt`,
  `bos_token`, `eos_token`, `add_bos`.
- `TemplateManager` — `self.templates: dict[str, ChatTemplate]`.
  - `register_from_gguf(model_path, model_name=None)` — calls
    `renderer.extract_template_from_gguf`, auto-detects the template's
    tool-calling format via `_detect_format` (string-pattern matching
    against `FORMAT_PATTERNS` — `qwen`, `hermes`, `llama3`, `mistral`,
    `gemma`, `gemma4`, `chatml`, `phi3`), caches the resulting
    `ChatTemplate` keyed by `model_name` (or the GGUF filename stem if
    not given), and returns it.
  - `get_template(model_name)` — cache lookup; returns a **default**
    (empty) `ChatTemplate()` instance for an unknown name rather than
    raising, so callers never need a `try/except KeyError`.
  - `render(model_name, messages, tools=None, add_generation_prompt=True)`
    — delegates straight to `renderer.render_chat`.
  - `build_tool_system_prompt(model_name=None, tools=None)` — builds a
    tool-definitions system-prompt string in the format matching the
    named model's detected `tool_format` (falls back to `"generic"` if
    the model name isn't registered); dispatches to one of
    `_build_{qwen,gemma4,mistral,generic}_tool_prompt`. Note `"hermes"`
    format shares the `qwen` builder (`_build_qwen_tool_prompt` is a thin
    alias calling `_build_generic_tool_prompt`) — Hermes and Qwen don't
    get genuinely different tool-prompt text despite being distinct
    entries in `FORMAT_PATTERNS`; this may be intentional (same
    underlying prompt convention) rather than a bug, but is worth
    knowing if Hermes-format models ever need their own prompt shape.
- `DEFAULT_TOOLS` — 6 hardcoded example tool definitions (`web_search`,
  `calculator`, `file_read`, `note_save`, `rag_search`, `weather_check`)
  used as the default tool set for `build_tool_system_prompt` when no
  explicit `tools` list is passed, and for testing.

---

## `templates/renderer.py` (420 lines)

The canonical Jinja2 chat-template renderer. Extremely thoroughly
commented in the source itself (explains Jinja2, GGUF, ChatML, BOS/EOS
concepts inline for a reader unfamiliar with LLM chat formatting) — this
summary covers behavior only.

- `CHATML_CLOSE = "im_end"` — the fallback ChatML closing-tag name,
  pulled into a constant (comment explains this is partly to avoid shell
  heredoc/tooling edge cases around literal `</...>`-shaped strings).
- `render_chat(template_str, messages, tools=None, bos_token="",
  eos_token="", add_generation_prompt=True, enable_thinking=False)` — the
  main entry point.
  - Empty `template_str` → immediately falls back to
    `_render_chatml_fallback`.
  - Non-empty → builds a `jinja2.Environment(loader=BaseLoader(),
    autoescape=False)`, registers a `tojson` filter
    (`lambda x: json.dumps(x, ensure_ascii=False)`), compiles the
    template, and renders it with `messages`, `tools` (defaulting to
    `[]`), `bos_token`, `eos_token`, `add_generation_prompt`,
    `enable_thinking`, plus **two aliases** for the tools list
    (`functions=tools or []`, `tool_definitions=tools or []`) since
    different model families' templates reference the tools list under
    different variable names.
  - **Any** exception during compile/render (malformed GGUF template,
    Jinja2 syntax error) is caught (`except Exception as e: # noqa: BLE001`),
    logged via `print()` (not the `logging` module — note this goes to
    stdout, not a structured log, so it won't show up in log-level
    filtering), and falls back to ChatML rather than propagating/crashing
    the inference call.
- `_chatml_wrap(tag, content_str)` — wraps content as
  `<|{tag}|>{content}</im_end>` (uses `CHATML_CLOSE`, not a tag-matched
  closer like `</{tag}>` — every role's closing tag is the same literal
  `</im_end>` regardless of whether the opening tag was `<|system|>`,
  `<|user|>`, etc. This matches real-world ChatML convention, which
  genuinely uses one shared `<|im_end|>` terminator for every turn — not
  a bug).
- `_render_chatml_fallback(messages, tools=None, bos_token="",
  add_generation_prompt=True)` — the ChatML fallback used when a model
  has no (or a broken) template. Wraps `system`/`user`/`assistant`/`tool`
  roles; **any other role value is silently skipped** (no log, no error —
  a message with role `"function"` or a typo'd role would simply vanish
  from the rendered prompt with no trace). `tools` is accepted for API
  symmetry with `render_chat` but genuinely unused in this fallback path
  (ChatML has no native tool-definition block).
- `extract_template_from_gguf(model_path)` — lazy-imports
  `llama_cpp.Llama`, loads the GGUF with `n_ctx=512, n_threads=1,
  verbose=False` (minimal settings purely to read `llm.metadata`, not to
  generate), reads `tokenizer.chat_template`,
  `tokenizer.ggml.{bos_token,eos_token,add_bos_token}`, and
  heuristically detects tool support by substring-scanning the template
  string for 5 known tool-call markers (`format_function_declaration`,
  `tool_call`, `AVAILABLE_TOOLS`, `tool_response`, a zero-width-space
  variant `<​tool_call>` for Qwen-style markers). On **any**
  exception (file not found, corrupt GGUF, missing llama_cpp) returns a
  hardcoded default dict (`chat_template=""`, generic `<bos>`/`<eos>`
  tokens, `add_bos=True`, `supports_tools=False`) after `print()`-logging
  the error — same stdout-only logging caveat as `render_chat` above.
- `render_with_model_template(model_path, messages, tools=None,
  add_generation_prompt=True)` — one-shot convenience: calls
  `extract_template_from_gguf` then `render_chat`. No caching — each call
  re-reads the GGUF metadata from disk. (Callers who need caching should
  go through `TemplateManager` in `manager.py` instead, which is the
  entire reason `manager.py` exists as a separate layer.)

---

## Discrepancies found + fixed (this lane)

1. **`src/finetune_studio/cli.py` — dead/unreachable file, deleted.**
   Verified empirically that `import finetune_studio.cli` always resolves
   to the `cli/` package regardless of `cli.py`'s presence (Python's
   regular-package-vs-module import resolution), confirmed
   `pyproject.toml`'s console-script entry points
   (`finetune_studio.cli:main`) resolve the same way, and confirmed no
   other file in the repo imports `cli.py` by path. Deleted the file.
   Regression test:
   `tests/test_core_entrypoints_audit.py::test_cli_py_shim_file_removed`
   + `::test_finetune_studio_cli_resolves_to_package` +
   `::test_entry_point_target_is_importable_and_callable` +
   `::test_python_dash_m_finetune_studio_runs_cli_help` +
   `::test_python_dash_m_finetune_studio_cli_runs_cli_help`. Ran:
   `.venv/bin/python -m pytest tests/test_core_entrypoints_audit.py -q`
   → **6 passed**.

2. **`src/finetune_studio/templates/__init__.py:15` and
   `src/finetune_studio/templates/manager.py:18` — dead duplicate
   docstring-shaped statements.** Each file had a real module docstring
   (the first statement) followed immediately by a *second* bare string
   literal (a no-op expression statement — evaluated and discarded,
   never stored anywhere, not accessible as `__doc__` or any attribute).
   Removed both dead lines. Regression test:
   `tests/test_core_entrypoints_audit.py::test_templates_package_has_no_dead_duplicate_docstring_statement`
   (parses both files with `ast` and asserts at most one leading string
   `Expr` statement). Ran as part of the same pytest invocation above
   → included in the 6 passed.

3. **`src/finetune_studio/webui/app.py` — misleading comment on the
   shared `inference_engine`.** The original comment stated "Do not
   construct a second InferenceEngine() anywhere," which is contradicted
   by the actual, legitimate codebase pattern of short-lived judge/
   benchmark engines in `webui/routes/benchmarks.py` (3 sites, all
   correctly call `_unload_global_inference()` first) and
   `webui/routes/projects.py` (1 site, which does **not** — see
   Cross-module findings below). Rewrote the comment to state the real
   contract precisely (a second engine is fine for a one-shot op only if
   the persistent engine's VRAM is freed first and the temp engine is
   unloaded in a `finally`), so a future reader of this file isn't misled
   into either (a) thinking the `projects.py` pattern is already a bug by
   design, or (b) copying the unsafe no-unload variant believing the old
   comment permitted it. No test added for this (it's a comment-only
   change with no observable behavior to pin); verified via `ast.parse`
   that `app.py` still parses correctly after the edit (see commands run
   below).

Verification commands run for all three fixes:
```
.venv/bin/python -c "import ast; [ast.parse(open(f).read()) for f in [...]]"   # syntax check, all OK
.venv/bin/python -m pytest tests/test_core_entrypoints_audit.py -q             # 6 passed in 2.95s
```

---

## Cross-module findings — NOT fixed, needs parent coordination

1. **`src/finetune_studio/webui/routes/projects.py:480`
   (`run_benchmark` handler) constructs a fresh `InferenceEngine()` and
   calls `engine.load(target_model)` without first freeing the shared
   persistent `inference_engine`'s VRAM.** Compare with the equivalent
   pattern correctly used three times in
   `src/finetune_studio/webui/routes/benchmarks.py` (lines ~221, ~628,
   ~667), each of which calls `_unload_global_inference()` (defined at
   `benchmarks.py:149`, which calls
   `models.llama_loader.unload_all_models()`) **before** loading its own
   temporary engine, and unloads the temp engine in a `finally` block
   when done. `projects.py:480`'s `run_benchmark` does load→use→unload
   its own temp engine in a `try/finally` (not shown here but present a
   few lines later), but never frees the *other*, already-resident
   persistent engine first. If a user has a model loaded in the Testing
   tab (via the shared `inference_engine`) and then runs a per-run
   benchmark from the project page, this call path can attempt to load a
   second model onto the GPU while the first is still resident — exactly
   the "mixed GPU/CPU offload" scenario the GH-AAA contract
   (`models/llama_loader.py`, owned by another lane) and the
   `_unload_global_inference()` helper exist to prevent. **Not fixed**
   because `webui/routes/projects.py` is outside this lane's file list
   (owned by the routes lane). Suggested fix for whichever lane owns that
   file: call the same `_unload_global_inference()` pattern (or a shared
   helper promoted out of `benchmarks.py` so both routers call the same
   function) before `engine.load(target_model)` at line ~480.

2. **`cli/commands/train.py` constructs its own `TrainingEngine()`
   instance, separate from `webui/app.py`'s shared `training_engine`
   singleton.** This is almost certainly correct (CLI runs as a
   standalone process, not inside the webui's ASGI process, so there is
   no singleton to share), documented above under "CLI/WebUI drift
   check" — noted here only so the parent session has the explicit
   confirmation on record that this was checked and is **not** a drift
   bug, in case another lane's audit flags the same observation without
   this context.

3. **`config.py`'s `Settings.rag_store_path` /
   `Settings.rag_embedding_model` duplicate
   `Settings.rag.store_path` / `Settings.rag.embedding_model`** (two
   independent fields holding conceptually the same value, with nothing
   keeping them in sync). Not fixed here — collapsing them safely
   requires auditing every caller across the whole repo (multiple lanes)
   to confirm which of the two each one actually reads, which is outside
   this lane's scope. Flagged for the parent session to decide whether a
   follow-up consolidation task is warranted.
