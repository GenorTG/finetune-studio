# db/ + models/ — persistence layer and model-loading stack

`db/` is the entire SQLite persistence layer for finetune-studio: one
stdlib-`sqlite3` schema (no ORM), one short-lived connection per call, CRUD
modules grouped by table. `models/` is the model-loading, provider-registry,
and model-discovery stack: it tracks what models exist on disk, how to load
them (one canonical GGUF loader, no mixed CPU/GPU offload), and which
provider (local GGUF or remote OpenAI-compatible) is currently active.

## db/connection.py (547 lines)

Low-level SQLite plumbing. Owns the entire schema as one `_SCHEMA` string
(`CREATE TABLE IF NOT EXISTS ...` for every table in the app — projects,
project_rags, training_runs, benchmark_runs/cases, auto_suites,
abliteration_runs, quant_exports, data_review, project_datasets,
project_versions, data_prep_runs, rag_corpora, hf_downloads, model_exports,
system_updates, activity_events, file_folders/project_files/file_versions/
file_conversions/folder_membership/model_favorites).

- `new_id()` — 8-hex-char id via `secrets.token_hex(4)`. Every table's PK.
- `_connect()` — opens a fresh `sqlite3.connect(settings.db_path)` per call,
  sets `row_factory = sqlite3.Row`, and turns on `PRAGMA foreign_keys` +
  `PRAGMA defer_foreign_keys`. Because every `cursor()` call opens a *new*
  connection, `defer_foreign_keys` (which SQLite resets at the end of each
  outermost transaction) is correctly re-armed every time — this is not a
  bug, just worth knowing if you ever introduce connection pooling.
- `cursor()` — the one context manager every db/*.py module uses. Commits
  on clean exit, rolls back on exception.
- `_safe_alter()` — runs an `ALTER TABLE ... ADD COLUMN` and swallows only
  "duplicate column name" errors, so `init_db()` is idempotent across app
  versions that added columns later.
- `init_db()` — runs `_SCHEMA`, then a list of `_safe_alter` migrations
  (widening pre-existing tables with new columns), then a one-off `UPDATE`
  that disambiguates auto-named `"Run · <dataset>"` training runs that
  collide across runs on the same dataset, then re-creates `benchmark_cases`
  defensively (the "new in v2" comment is stale — this duplicate
  `CREATE TABLE IF NOT EXISTS` block pre-dates the one already in `_SCHEMA`
  and is a harmless no-op since both target the same table; the first one
  to run wins and the second is a no-op per `IF NOT EXISTS`).
- `row_to_dict()` — converts a `sqlite3.Row` to a `dict` and auto-decodes a
  fixed list of `*_json` TEXT columns into unsuffixed dict/list keys:
  `rag_ids_json`, `settings_json`, `metrics_json`, `scores_json`,
  `options_json`. **Gotcha:** `project_versions.manifest_json` and
  `auto_suites.categories_json` are NOT in this list — callers (e.g.
  `webui/routes/versions.py`) decode `manifest_json` manually with
  `json.loads(v.get("manifest_json") or "{}")`. This is intentional (routes
  own the manifest-key allowlist), not an oversight, but it means
  `get_version()` / `list_versions()` return raw JSON strings, unlike every
  other `_json` column in the schema.

Called by: every other `db/*.py` module (`cursor`, `new_id`, `row_to_dict`).
`db/__init__.py` calls `init_db()` once at import time.

## db/__init__.py (270 lines)

Facade: imports every public CRUD function from the per-table modules below
(often renamed on import, e.g. `model_exports.mark_done` → `mark_export_done`,
`rag_corpora.mark_done` → `mark_rag_build_done`) and re-exports them in one
flat `__all__` list so callers write `db.create_run(...)`, `db.mark_export_done(...)`
etc. Calls `init_db()` at module import time, so importing `finetune_studio.db`
anywhere creates/migrates the DB file as a side effect.

**Verified invariant:** every name imported into this module is present in
`__all__` and vice versa (checked programmatically during this audit — no
drift). The `__all__` list exists specifically so `ruff F401` doesn't flag
the whole facade as dead imports; its own comment says ~56 call sites depend
on attribute access working, which `__all__` membership doesn't actually
gate (Python attribute access ignores `__all__`; it only affects
`from db import *`) — but keeping the two in sync is still good hygiene and
is enforced here.

## db/activity_events.py (50 lines)

Append-only log for operations that don't have their own job table (uploads,
chat/query, model loads, small mutations). `record()` inserts one row and
re-selects it to return a fully-populated dict (two queries, intentionally —
simpler than hand-building the dict from the insert's known values).
`list_recent(limit=100)` — newest-first across all projects, feeds the
global activity feed. No update/delete; events are immutable once recorded.

## db/benchmarks.py (161 lines) — FIXED

CRUD for `benchmark_runs` (parent) and `benchmark_cases` (per-question
results). Two independent insert paths exist for cases:

1. `create_benchmark(run_id, suite_name, scores, time_ms, cases, model_path)`
   — creates the parent row, and if `cases` is given, inserts each one
   inline with ALL 20 benchmark_cases columns (including
   `scoring_method`, `validity`, `error`, `judge_input`, `source_id`,
   `chunk_idx`). This is the path every live route uses
   (`webui/routes/projects.py`, `benchmarks.py`, `testing.py`).
2. `create_case(benchmark_id, run_id, name, category, question,
   correct_answer, model_answer, transcript, judge=, judge_model=,
   verdict=, judge_reasoning=, scored_at=, scoring_method=, validity=,
   error=, judge_input=, source_id=, chunk_idx=)` — a standalone
   single-case inserter, used today only by a test
   (`tests/test_browser_result_surfaces.py`).

**Bug found and fixed:** `create_case()`'s function signature accepted
`scoring_method`, `validity`, `error`, `judge_input`, `source_id`, and
`chunk_idx` as keyword parameters, but its `INSERT` statement only wrote 14
of the table's 20 columns — those six were silently dropped. Any caller
passing `scoring_method="exact"` (for example) would get no error and no
persisted value; `list_cases()` would just return the column defaults
(`''`, `''`, `''`, `{}`, `''`, `0`). Fixed by extending the `INSERT` to
cover all six fields, matching what `create_benchmark()`'s inline inserter
already did correctly. Regression test:
`tests/test_db_models_audit.py::test_create_case_persists_all_declared_fields`.

`list_cases()` manually decodes `transcript` and `judge_input` JSON columns
itself (not via `row_to_dict`, which only handles the fixed `*_json` list
and these columns aren't suffixed `_json` in the schema — they're
`transcript` and `judge_input` as plain TEXT). `update_case(cid, **kwargs)`
JSON-encodes `transcript`/`judge_input` if given as non-str, then does a
raw `UPDATE ... SET <cols> WHERE id = ?` with no column allowlist (unlike
every other `update_*` function in this package) — any kwarg name is
written as a literal column name via f-string interpolation. This is safe
only because every call site is internal (the judge pipeline), never
user-supplied keys; worth knowing if a future caller ever forwards a
request body here directly.

## db/connection.py — see above (listed first since every module imports it)

## db/data_prep_runs.py (141 lines)

CRUD for `data_prep_runs` — durable record of the data-prep pipeline so the
in-memory `_RUNS` dict in `webui/routes/data_prep.py` can survive a
restart. `create_run()` inserts a `queued` row; `mark_running` /
`mark_done` / `mark_failed` are thin wrappers over `update_run(**fields)`'s
column allowlist. `mark_done()` computes `duration_ms` from `started_at`
read back from the DB (not from the caller) — safe against clock drift
between process threads. `list_stale()` returns full rows for
`queued`/`running` runs after a restart so a caller can resume them;
`reconcile_stale()` is the fallback that marks them `failed` when resume
isn't possible (e.g. source file gone).

**Gotcha (not a bug, but inconsistent):** `mark_failed()` sets
`status="error"`, while `reconcile_stale()` sets `status='failed'` for the
same semantic "did not complete" state. The route layer
(`webui/routes/data_prep.py:563`) already compensates:
`"error" if status in ("error", "failed") else status` — so the two values
are normalized before display. If you add a new consumer of this table's
`status` column, check both strings, not just one.

## db/datasets.py (119 lines) — FIXED

CRUD for `project_datasets` — jsonl training files registered to a project
(from upload or data-prep export). `datasets_dir(pid)` returns (and
creates) `<db_dir>/projects/<pid>/datasets/`. `create_dataset()` computes
`size_bytes` from the file on disk if not given. `get_dataset_by_path()`
dedupes registrations by `(project_id, data_path)`. `count_qa_pairs()`
counts non-empty lines in a jsonl file, best-effort (returns 0 on any read
error, never raises).

**Bug found and fixed:** `update_dataset(did, **fields)`'s `allowed` set
included `last_used_at` as a caller-settable field via the normal
`for k, v in fields.items(): if k in allowed: ...` loop — but the function
*also* unconditionally appended its own `SET last_used_at = ?` with
`time.time()` after that loop, regardless of whether the caller had
already set one. If any future caller passed `last_used_at=<specific
value>` expecting it to persist, SQLite would apply both `SET
last_used_at = ?` clauses in order and the caller's explicit value would be
silently overwritten by "now" (no current caller does this — it was a
latent landmine, not an active symptom). Fixed by removing `last_used_at`
from `allowed`, since the function already always stamps it; the docstring
now says why. Regression test:
`tests/test_db_models_audit.py::test_update_dataset_does_not_clobber_explicit_last_used_at`.

## db/hf_downloads.py (100 lines)

CRUD for `hf_downloads` — durable HF Hub download job tracking, replacing
the in-memory `_DOWNLOADS` dict in `webui/routes/hf_models.py` so downloads
survive a restart. Standard `create_job` / `mark_running` / `mark_done` /
`mark_failed` / `mark_cancelled` over `update_job()`'s column allowlist.
`list_in_progress()` returns `queued`/`downloading` jobs so the UI can
reattach after a restart. Clean — no discrepancies found.

## db/model_exports.py (129 lines)

CRUD for `model_exports` — one row per GGUF export attempt (format/quant
extensible). `create_export(project_id, run_id, *, format="gguf",
quant="Q4_K_M")` → queued row. `mark_running` → `running`. `mark_done(eid,
*, output_path="", size_bytes=0, size_human="", intermediate_path="")` →
`done`, computing `duration_ms` from the DB's own `started_at` (same
pattern as `data_prep_runs`/`hf_downloads`). `mark_failed` → `error`
(truncates error to 1000 chars). This is the exact contract
`webui/routes/exports.py` (owned by another lane) calls once per quant:
create → mark_running → mark_done/mark_failed, keyed by
`(project_id, run_id, format, quant)`. **Verified sound** — every field
`mark_done` accepts has a matching column in `update_export()`'s allowlist
and the schema; no silent drops found here, unlike `benchmarks.create_case`.
`reconcile_stale()` mirrors `rag_corpora`/`data_prep_runs`: marks
in-flight (`queued`/`running`) rows `failed` after a restart.

## db/project_versions.py (117 lines)

CRUD for `project_versions` — immutable, append-only snapshots of a
project's pinned inputs (datasets, source files, RAG corpora, training
runs, base model). `_ALLOWED_MANIFEST_KEYS` whitelists exactly 6 keys
(`datasets`, `source_ids`, `rag_corpora`, `training_runs`, `base_model`,
`suites`); `_clean_manifest()` drops anything else before JSON-encoding.
`create_version()` derives `version_number` as `MAX(version_number)+1` per
project when not given, keeping numbers monotonic. `version_lineage()`
walks `parent_version_id` back to the root (cycle-safe via a `seen` set;
also stops if a parent belongs to a different project, guarding against a
corrupted/cross-project link). See the `manifest_json` gotcha under
`connection.py` above — this module returns it undecoded; the caller in
`webui/routes/versions.py` (another lane's file) is the one place that
`json.loads()`s it.

## db/projects.py (124 lines)

CRUD for `projects` plus `model_favorites` (a separate, unrelated table
that lives here because it's small and project-adjacent in the UI).
`add_model_favorite()` uses `INSERT ... ON CONFLICT(model_path) DO UPDATE`
(upsert by path) and explicitly warns in its docstring: `model_favorites.id`
is `INTEGER AUTOINCREMENT`, so never pass a hex `new_id()` string into it —
SQLite would silently coerce it to `0`. `delete_project()` always returns
`True` regardless of whether a row actually existed (no `cursor.rowcount`
check) — harmless today since no caller branches on the return value, but
if you add one, check `rowcount` instead of trusting this return value.

## db/rag_corpora.py (126 lines)

CRUD for `rag_corpora` — one row per RAG build/ingest attempt; `project_rags`
holds the latest summary, this table is the history. Same
create/mark_running/mark_done/mark_failed/reconcile_stale shape as
`data_prep_runs`/`hf_downloads`/`model_exports`. `latest_for_rag()` is what
the UI shows as "last build" (`ORDER BY created_at DESC LIMIT 1`). Clean.

## db/rags.py (120 lines)

CRUD for `project_rags` (the per-project vector-store registration, not the
build-history table above). `ensure_portable_rag(project_id, store_path, *,
name=, doc_count=, chunk_count=)` is the idempotent upsert used after every
PortableRAG build: it scans existing rags for one whose `store_path`
(absolute-normalized) matches, updates it if found, else creates + updates
a fresh row. Both branches set `status="ready"`, `last_build_status="ok"`,
clear `error=""`. Clean — no partial-state issue (the create-then-update
two-step means a crash between them would leave a `project_rags` row with
stale `doc_count=0`/`status` defaults rather than losing the row entirely,
which is the safer failure mode).

## db/reviews.py (33 lines)

Row-level approve/reject/edit decisions for dataset rows (`data_review`).
`record_review()` deletes any prior decision for
`(project_id, dataset, row_index)` then inserts the new one — so it's a
true "last decision wins" overwrite, not an append-only history. Smallest
file in the lane; nothing to flag.

## db/runs.py (123 lines)

CRUD for `training_runs`. `create_run()` JSON-encodes `rag_ids` →
`rag_ids_json` and `settings_obj` → `settings_json`. `list_runs()`
post-processes every row after `row_to_dict()`: computes `duration` from
`started_at`/`finished_at` when both are set, and backfills `final_loss`
from `metrics.get("final_loss", metrics.get("loss"))` for legacy rows that
only ever recorded the last training-step loss (the `final_loss` column is
authoritative when present; this is a read-time compatibility shim, not a
write-time bug). `update_run()`'s allowlist deliberately excludes
`rag_ids_json`/`rag_ids` — a run's RAG attachments are fixed at creation,
not editable after the fact; `webui/routes/projects.py`'s generic
`PATCH .../runs/{rid}` handler forwards the whole request body to
`db.update_run(rid, **body)`, so if a client ever sent `rag_ids` in that
PATCH it would be silently ignored (same allowlist pattern used by every
other `update_*` function in this package — intentional defense against
arbitrary body keys hitting SQL, not a new regression). `reconcile_stale_runs()`
treats `queued`/`loading`/`training`/`saving`/`running` as the stale set and
marks them `failed` with a fixed message after a restart.

## db/system_updates.py (150 lines)

CRUD for `system_updates` — one row per self-healing update/check/repair
attempt, with `log_text` accumulating the script's stdout/stderr via
`append_log(uid, chunk, max_bytes=256_000)` (trims from the top, keeping
the most recent `max_bytes`, so the row never grows unbounded — this is a
deliberate truncation **with** a documented policy, not the "truncation
without warning" anti-pattern; the trimmed content is simply gone, no
warning marker is inserted at the trim point, which would be a nice-to-have
but isn't a correctness bug since `log_tail`-style UIs only care about
recent output anyway). `reconcile_stale()` has real domain logic: because
`update.sh` runs as a child of the service and systemd kills *both* the
script and the streaming worker at the "restart" step, a genuinely
successful update's row would otherwise be stuck "running" forever. It
reads the accumulated `log_text` and treats `"Update complete."` or
`"Restarting finetune-studio.service"` appearing in the log as proof every
real step succeeded (→ `mark_done` + an explanatory log line), and anything
else as a genuine failure (→ `mark_failed`).

## models/__init__.py (2 lines)

Just a docstring (`"""Models subpackage — model registry and loading."""`).
No exports.

## models/gguf_layers.py (68 lines)

Reads real GGUF header values via the `gguf` pip package (ships alongside
`llama-cpp-python`) instead of trusting a caller-supplied `n_gpu_layers=99`
magic value ("all layers"), which used to lie to the UI about model depth.
`gguf_header_values(path)` returns every scalar uint32/uint64 header field
as `{field_name: int}`, swallowing any read error into an empty dict (the
file may be corrupt/unreadable — never raises). `resolve_block_count(path)`
extracts just `block_count` and `context_length` by matching the
unprefixed suffix of each header key (keys are architecture-prefixed, e.g.
`qwen3.block_count` or `llama.block_count` — this function deliberately
never assumes one prefix, taking the *first* match for each field name).
Returns `{"block_count": None, "context_length": None}` immediately for
non-`.gguf` paths. Called by `models/providers.py` (`LocalGGUFProvider`)
and `models/manager.py` (`ModelManager.load`) to report true layer counts
("36/36") instead of the legacy fake 99.

## models/helper.py (222 lines)

Resolves "the configured local helper model" (used for data-prep mining
and LLM-assisted suite generation) as a first-class, explicitly-identified
concept — callers must resolve it by provider id or path, never silently
reuse whatever happens to be loaded on the Inference tab. Two helper seats
are hardcoded: `DEFAULT_HELPER_PROVIDER_ID = "local-default"` (Gemma 4 12B
Uncensored Q4_K_M) and `ALTERNATE_HELPER_PROVIDER_ID = "local-qwen30b-a3b"`
(Qwen3-30B-A3B IQ4_XS) — both seeded into the DB by
`models/manager.py::_ensure_db()`. `default_helper_gguf_path()` /
`alternate_helper_gguf_path()` resolve the on-disk path, honoring
`FTS_HELPER_GGUF`/`FTS_ALT_HELPER_GGUF` env overrides first. `paths_match()`
/ `normalize_model_path()` do absolute-path-normalized equality so
relative-vs-absolute path spellings of the same file compare equal.
`is_helper_provider(row)` checks provider id OR resolved path — covers a
provider row whose id was renamed but whose `model_id` still points at a
known helper basename. `annotate_provider()` is what every provider dict
returned to the UI gets passed through (`models/manager.py::list_providers`
and `get_provider`), adding `is_helper` + a display `label`.
`get_configured_helper_provider()` is the canonical "give me the helper, or
None" entry point used by data-prep and suite-generation call sites, via
`ModelManager.get_provider(DEFAULT_HELPER_PROVIDER_ID)` with a fallback
scan of all providers for any helper-shaped row.

## models/llama_loader.py (236 lines)

**The one true GGUF loader.** `load_llama_gguf(gguf_path, *, n_ctx=32768,
n_gpu_layers=-1, n_batch=512, n_threads=None, seed=None, rope_freq_base=0.0,
rope_freq_scale=0.0, flash_attn=True, mmap=True, mlock=False, type_k=0,
type_v=0, detect_mmproj=True) -> LlamaLoadResult` is the single place that
calls `llama_cpp.Llama(**kwargs)`. Before this module existed,
`LocalGGUFProvider.load()` and `InferenceEngine._load_gguf()` each built
their own `Llama(...)` call independently and drifted (only one had
OOM-retry, only the other had mmproj/vision auto-detection). Every caller —
data-prep's helper, chat, RAG, testing, benchmarks, the inference tab — now
goes through this one function.

Key behavior:
- **mmproj/vision auto-detection** (`detect_mmproj=True`): globs the GGUF's
  directory for `mmproj*.gguf`, falling back to `*mmproj*<base_name>*.gguf`
  with quant suffixes stripped from `base_name`. On a hit, wraps a
  `Qwen25VLChatHandler`; any failure here is caught and downgraded to a
  warning (`result.warnings`) — vision is optional and must never block
  text-only loading.
- **GH-AAA contract: no mixed CPU/GPU offload.** On CUDA/VRAM/OOM errors
  (detected by substring match on the lowercased exception message:
  `"out of memory"`, `"cuda"`, `"vram"`), the retry loop (max 6 attempts)
  halves `n_ctx` (floor 512) and retries — **model layers always stay
  fully on GPU; only the KV cache shrinks.** It never falls back to
  offloading layers to CPU. Any other exception type is re-raised
  immediately, un-retried.
- `LOADER_PARAM_NAMES` is the single source of truth for which loader
  kwargs exist — both `resolve_loader_overrides()` below and
  `models/manager.py::_LOADER_KEYS` are meant to stay in sync with it
  (duplicated as a plain tuple in `manager.py` rather than importing this
  one, since `manager.py` needs `keep_in_memory` too, which isn't a
  `Llama()` kwarg at all — see that file's note).
- `resolve_loader_overrides(body, *, caller, model_path="", default_ctx=True)`
  is the single place every `/load` HTTP route derives Llama kwargs from a
  request body — previously copy-pasted with drifted wording into
  `models.py`, `chat_v2.py`, and `testing.py` routes (one of which,
  `chat_v2.py`, didn't enforce the GPU guard at all). Forces
  `n_gpu_layers=-1` always, logging a warning (naming the caller) if the
  body asked for anything else. `default_ctx=False` is for
  `ModelManager.load()` call sites specifically, since those already fall
  through to the provider's *persisted* `n_ctx` and forcing
  `DEFAULT_N_CTX` here would silently overwrite that persisted value on
  every load.
- `DEFAULT_N_CTX = 32768` — the single floor every caller must agree on;
  the comment explicitly calls out that it used to be 16384 or 4096 in
  various now-unified places.
- `unload_all_models()` does two distinct things for two distinct reasons
  (not two engines to coordinate — there's structurally only one
  `InferenceEngine` in the process now): (1) `inference_engine.unload()`
  frees the actual model/VRAM; (2) `get_manager().unload()` clears
  `ModelManager`'s own `_provider`/`_active_id` bookkeeping, which step 1
  alone wouldn't touch if a model was loaded directly via
  `inference_engine.load()` bypassing `ModelManager.load()` (as
  `testing.py`/`chat_v2.py` do). Both steps are wrapped in their own
  try/except — a missing/broken engine on one side must not block
  unloading the other.

Called by: `models/providers.py` (`LocalGGUFProvider` doesn't call
`load_llama_gguf` directly — it delegates to the shared `InferenceEngine`,
which does), `/load` routes (`resolve_loader_overrides`), and any
unload/model-switch path (`unload_all_models`).

## models/loader.py (48 lines) — FIXED (docstring, not deletion)

**This file is NOT dead code** despite HANDOFF.md's note about an earlier
pass deleting `load_for_inference`/`load_gguf_inference` from this module —
those were a different, genuinely dead third Llama-construction path that
has in fact been removed. What remains, `load_model_info(model_path) ->
dict`, is alive: it's imported by `webui/routes/models.py` and
`webui/routes/pages.py` (both outside this lane) to produce the "what is
this model" display metadata (GGUF size, or safetensors
architecture/hidden_size/num_layers/vocab_size/shard count/total size) shown
in the UI's model picker — it does not build or wrap any inference engine.

**Bug found and fixed:** the file's first statement was
`from typing import Any`, followed immediately by a triple-quoted string
that *looked* like the module docstring (headed "Load GGUF and HuggingFace
models" with sections on metadata tracking, multi-model support, lazy
loading — none of which this module actually does; it only reads a path
and returns a dict). Because Python only captures a string literal as
`__doc__` when it is the very first statement in the module, that string
was a dead, side-effect-free expression statement — `import
finetune_studio.models.loader as m; m.__doc__` returned `None`, proven
during this audit. Fixed by moving a corrected, accurate docstring (stating
what the module actually does, and pointing at
`models/llama_loader.py` for the real GGUF loader) to be the file's first
statement, ahead of the `pathlib`/`typing` imports it needs.

## models/manager.py (412 lines)

`ModelManager` — holds the single active inference provider for the whole
process; "one local model loaded at a time" is enforced here, not left to
callers. Embeddings have their own separate resident model elsewhere that
this class never touches.

- `_ensure_db()` creates `model_providers` + `app_state` tables in a
  *separate* SQLite file (`_DB_PATH`, default
  `~/.finetune-studio/fts.db`, overridable via `FTS_DB` env var — this is
  intentionally not the same DB file `db/connection.py` uses) and seeds the
  two helper provider rows (`local-default`, `local-qwen30b-a3b`) on first
  run. The rest of the function is a chain of one-off migrations for
  renaming/retiring legacy helper rows (27B → current default, old 8B
  Qwen3 → Gemma 4 12B, etc.) — each guarded by `WHERE id = ? AND name = ?`
  / `WHERE model_id LIKE ?` so they're no-ops on a fresh or already-migrated
  DB. This runs every time `ModelManager()` is constructed, not just once —
  cheap because every migration is a narrowly-scoped idempotent `UPDATE`.
- `_LOADER_KEYS` is a near-duplicate of `llama_loader.LOADER_PARAM_NAMES`
  plus `keep_in_memory` (which `load_llama_gguf` doesn't accept at all —
  it's consumed elsewhere, not passed through to `Llama()`). Keep these
  two tuples in sync by hand if you add a new loader parameter; there is
  no shared single source of truth between the two files.
- `ModelManager.engine` (property) — lazily constructs exactly one
  `InferenceEngine` instance (`finetune_studio.testing.inference`) and
  caches it on `self._engine_instance`. `webui/app.py`'s global
  `inference_engine` is set to point at this exact object (see that file,
  outside this lane), so a model loaded via a named provider and a model
  loaded via a raw path can never both be resident — there's structurally
  one holder of local-model state, not two coordinated ones. Uses
  `getattr(self, "_engine_instance", None)` rather than assuming
  `__init__` ran, so a test subclass that skips `ModelManager.__init__` (to
  avoid touching the real DB) still gets a working engine.
- `list_providers()` / `get_provider(pid)` — read `model_providers`,
  JSON-decode `extra_json` → `extra`, and run every row through
  `models.helper.annotate_provider()` for `is_helper`/`label`.
- `upsert_provider(**kw)` — the public API's field name is `extra`
  (dict); it's renamed to `extra_json` (JSON string) before hitting SQL.
  Insert vs. update is a pre-check `SELECT id ... WHERE id = ?`, not an
  `ON CONFLICT` upsert (unlike `db/projects.py::add_model_favorite`) — fine
  since this whole method already runs inside one `with sqlite3.connect`
  block, so there's no cross-call race within a single call, only
  cross-thread (same risk as every other write in this file; nothing here
  changes that).
- `ModelManager.load(pid, extra=None)` — the core state machine:
  1. Merges `extra` (runtime override) on top of the provider's persisted
     `extra` JSON; only keys in `_LOADER_KEYS` and not `None` count as
     "explicit".
  2. **Persists explicit overrides back to the provider row** — this is
     deliberate, documented behavior: without it, a context-length change
     made on one page (Inference) would only apply to that one in-memory
     load, and the next auto-load from any *other* page (data-prep's
     "Generate pairs", RAG's "Ask the model", a benchmark judge-model load,
     ...) would silently reload from the stale old default. One explicit
     override becomes the shared default for every future load of that
     provider, not nine independently-drifting copies.
  3. Translates the legacy `n_gpu_layers=99` ("all layers") sentinel to the
     real llama.cpp idiom `-1`, and resolves real GGUF topology via
     `gguf_layers.resolve_block_count()` to attach `topology_block_count`.
  4. **Fast path:** if the same `pid` is already loaded with byte-identical
     `merged_extra` (compared against `new_provider._used_extra`, a
     snapshot attribute set at construction time), returns the already-
     active provider's `describe()` without doing anything — this exists
     specifically because a naive re-load used to try constructing a
     second `Llama()` while the first still held the mmap/VRAM, raising
     "Failed to load model from file".
  5. Otherwise builds a fresh provider via `build_provider()`, and only
     after a successful `new_provider.load()` call does it assign
     `self._provider = new_provider; self._active_id = pid` inside the
     lock — **verified during this audit (Priority Check #3): if
     `new_provider.load()` raises, the exception propagates before either
     assignment runs, so a failed load never leaves `ModelManager`
     reporting a provider as active that isn't actually loaded.** The
     prior provider (if of a different kind, or a different id) is
     unloaded *before* the new one is constructed via `_safe_unload()`,
     which itself never raises (wraps `provider.unload()` in its own
     try/except and always clears `self._provider = None` regardless of
     whether unload succeeded) — so a crash mid-switch leaves the manager
     in a clean "nothing loaded" state, never a stale "old model marked
     active" state.
- `chat()` / `generate()` — read `self._provider` under `self._lock`, then
  invoke it under a *separate* `self._invoke_lock` (not the same lock) —
  this is deliberate: `llama_cpp.Llama` isn't thread-safe so concurrent
  generations must serialize, but holding the *state* lock for the whole
  duration of a generation would block `load()`/`unload()` from even
  reading `self._provider` while a long generation is in flight.
- `get_manager()` — process-wide singleton behind `_manager_lock`.

## models/providers.py (333 lines)

Defines the `ModelProvider` abstraction (`load`/`unload`/`chat`/`generate`/
`describe`/`is_loaded`) and its two concrete implementations, plus the
`PROVIDER_PRESETS` catalog used to populate the provider picker UI.

- `LocalGGUFProvider` — despite the name, handles both GGUF and
  HF/safetensors paths, because it delegates everything to the shared
  `InferenceEngine` (`self.engine`, lazily constructed if not injected),
  and `InferenceEngine.load()` dispatches on file extension. This class
  used to build its own `Llama()` directly; now every load/unload/chat/
  generate call goes through `self.engine`, which is the *exact same
  object* as `webui/app.py`'s `inference_engine` global (enforced by
  `ModelManager.engine`) — structurally impossible for two local models to
  be resident at once, not just coordinated to behave that way.
  - `_mine()` — the real "is this provider's model the one currently
    loaded in the shared engine" check, compared by `model_path` string
    equality. This is what `is_loaded()` and `describe()`'s `loaded` field
    actually use, not a local flag — so even if two `LocalGGUFProvider`
    instances existed for the same `config.id` (shouldn't happen, but
    isn't structurally prevented), only the one that's actually "mine"
    per the shared engine would report loaded.
  - `load()` — **verified during this audit (Priority Check #3): calls
    `self.engine.load(...)` first; `self._loaded_at = time.time()` is only
    set after that call returns successfully.** If `engine.load()` raises,
    `_loaded_at` stays at its previous value and the exception propagates
    to `ModelManager.load()` (see above) — no false "loaded" state.
    Re-reads `self.engine.n_ctx` afterward since the OOM-retry inside
    `load_llama_gguf` may have shrunk `n_ctx` below what was requested;
    `describe()` must report what's actually resident, not the ask.
  - `unload()` — calls `self.engine.unload()` only `if self._mine()`, then
    unconditionally sets `self._loaded_at = 0.0`. If `engine.unload()`
    itself raises, the `_loaded_at = 0.0` line is skipped (no try/except
    here) — but this doesn't cause a stale-active bug because `is_loaded()`
    is overridden to call `_mine()` (which re-derives from the shared
    engine's real state), never reading `_loaded_at` directly.
  - `describe()` — when `block_count` is known (real GGUF topology from
    `gguf_layers.resolve_block_count`), reports `gpu_layers_on`/
    `gpu_layers_total` as the full block count whenever `n_gpu_layers == -1`
    (all layers), giving the UI a real "36/36" instead of the legacy fake
    99-layers display.
- `OpenAICompatProvider` — any OpenAI chat-completions-shaped remote
  endpoint. `load()` is a true no-op (just stamps `_loaded_at`; there's no
  connection to actually establish for a stateless HTTP client) so it can
  never fail partway — nothing to verify here for Priority Check #3, the
  class of bug doesn't apply. `generate()` tries `/completions` first and
  falls back to `chat()` on any exception (many OpenAI-compat hosts only
  implement the chat endpoint).
- `build_provider(config, engine=None)` — the only place that decides
  `local_gguf` → `LocalGGUFProvider` vs. `openai_compat` →
  `OpenAICompatProvider`; raises `ValueError` on an unknown `kind`.
- `PROVIDER_PRESETS` — static catalog (OpenAI, OpenRouter, opencode-go,
  Anthropic-via-proxy, MiniMax, Custom, plus the two local helper presets
  built from `models/helper.py`'s constants) used to seed the provider
  picker's "add a provider" dropdown.

## models/registry.py (393 lines)

Filesystem/HF-cache model discovery — "what models exist on disk" — plus
the category-based filtering that the Inference and Training pages use to
decide which discovered models to offer.

- `ModelInfo` dataclass — `category` is one of `discovered`, `base_model`,
  `trained_export`, `local_helper`, `downloaded`.
- `models_for_selectors(models)` — Inference/chat dropdown filter: drops
  anything under a `shared_models/` path (embedder/reranker caches) and
  anything whose category isn't in `_SELECTOR_CATEGORIES`. Accepts either
  `ModelInfo` instances or plain dicts (checked via `isinstance(m, dict)`
  at each attribute read) since some call sites already pre-serialize to
  JSON-friendly dicts before this filter runs.
- `is_trainable_base_model(m)` / `models_for_training(models)` — Training
  page's base-model filter: excludes GGUF/GPTQ/AWQ by both the `format`
  field (`_TRAINING_EXCLUDED_FORMATS`) and path heuristics (trailing path
  segment or any `/gptq//gguf//awq/` component), since some legacy export
  layouts still say `format="safetensors"` in their `ModelInfo` despite
  actually being a quantized export directory.
- `_safe_model_name(root, cfg, project_name="")` — turns a directory into a
  human-readable display name: prefers `config.json`'s `name`/`model_name`,
  then unwinds HF-cache hash directories (`models--org--repo/snapshots/
  <hash>` → `org/repo`) and this app's own HF-Explorer cache naming
  (`Org__Repo@rev` → `Org/Repo`), and prefixes generic export dirnames
  (`merged`, `abliterated`, `gguf`, `gptq`, `adapter`, `checkpoint-N`) with
  the project name for context instead of showing a bare "merged".
- `_lookup_project_name(output_path)` — resolves `(project_id,
  project_name)` for a trained-export path. Primary path: parse via
  `naming.resolve_run_path()` then `db.get_project()` through the real app
  DB layer. The comment explicitly documents a past bug this replaced: an
  older raw-sqlite lookup pointed at
  `~/.finetune-studio/finetune_studio.db`, which is *not* where the
  deployed DB actually lives (`settings.db_path`/`FTS_DB_PATH`), so it
  silently failed and every run export showed its bare hash-dir name
  instead of a readable name. Fallback: walks up `output_path`'s parent
  directories (max 6 levels) querying `training_runs` by
  `output_path = ? OR ? LIKE output_path || '%'` for non-standard layouts.
- `scan_models(directories)` — walks each directory tree (skipping
  dotfiles and `__pycache__`), classifying every `.gguf` file and every
  `safetensors`+`config.json` directory into a `ModelInfo` with category
  inferred from path markers (`shared_models` → `local_helper`,
  `hf_models` → `downloaded`, `huggingface/hub` → `base_model`, anything
  matching `naming.resolve_run_path()` or containing `/output`/`output_`/
  `/models/safetensors/` → `trained_export`). Skips known non-chat
  architectures (`_SKIP_ARCHES`: Bert/Roberta/XLMRoberta/MPNet/
  Clip/Siglip-vision/MultiModalProjector) and multimodal projector/vision
  GGUF files (`_SKIP_GGUF_PATTERNS`). `_weight_bytes()` sums only
  *readable* weight files, explicitly ignoring dangling HF-hub symlinks
  (broken cache entries) rather than letting `os.path.getsize` on a
  dangling symlink abort the whole directory's size calculation — so one
  broken symlink doesn't zero out or crash the size for an otherwise-valid
  model directory. Safetensors dirs under 0.5 GB total are skipped
  entirely (`dirs.clear(); continue`) as incomplete installs/config-only
  leftovers.

Called by: `webui/routes/models.py` and `webui/routes/pages.py` (model
picker / dashboard, both outside this lane) to populate "what models exist"
listings; `models_for_training` / `models_for_selectors` gate which
discovered models each page's dropdown offers.

## Cross-cutting invariants for this lane

- **GH-AAA (no mixed GPU/CPU offload):** enforced in exactly one place,
  `llama_loader.load_llama_gguf`'s OOM-retry loop (shrinks `n_ctx`, never
  falls back to partial CPU offload) and `resolve_loader_overrides`
  (forces `n_gpu_layers=-1` on every `/load` route regardless of what the
  request body asked for). No violation of this contract was found
  anywhere in `models/` during this audit.
- **One true GGUF loader:** `llama_loader.load_llama_gguf` is the only
  function in the codebase that constructs `llama_cpp.Llama(...)`.
  `LocalGGUFProvider` and `ModelManager` both delegate to it indirectly via
  the shared `InferenceEngine`, never duplicating the construction logic.
- **One shared InferenceEngine:** `ModelManager.engine` and
  `webui/app.py`'s `inference_engine` global are the same object — verified
  by reading both `models/manager.py` and the `unload_all_models()`
  docstring in `models/llama_loader.py`, which explains exactly why two
  separate `unload()` calls (engine + manager bookkeeping) are still
  needed even though there's only one engine.
