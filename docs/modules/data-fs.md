# data/fs — Project filesystem storage layer

On-disk source of truth for one project's files, chunks, parsed content and
Q&A records, plus the shared cross-project model cache. There are, by design,
**two parallel storage schemes** sharing the same `files/` directory on disk:
an older content-addressed scheme (`data/fs/files.py`, keyed by sha256) used
by the data-prep/RAG ingestion pipeline, and a newer DB-backed file-library
scheme (`data/fs/file_library.py`) used by the file-browser UI (folders,
versions, soft-delete/trash). Both are live and actively called — this is not
dead code left over from a migration, it is two different subsystems that
happen to share a root directory. See "Two storage schemes" under Gotchas.

## `data/fs/paths.py`

Pure path helpers, no I/O beyond `mkdir`. `root()` → `~/.finetune-studio`
(override via `FTS_ROOT` env var, used by tests). `project_dir(pid)` →
`<root>/projects/<pid>`. `file_dir(pid, sha256)` → the **legacy**
content-addressed directory `files/<sha256[:12]>/`, used by the old
`files.py`/data-prep pipeline. `project_files_root(pid)` → `files/` itself,
the root the **new** `file_library.py` builds `raw/` and `converted/` under.
Every function creates its directory eagerly (`mkdir(parents=True,
exist_ok=True)`) as a side effect of being called — callers rely on this to
avoid explicit `ensure_dirs()` calls in most places.

## `data/fs/project.py`

`write_project_json`/`read_project_json` — single `project.json` document per
project. `read_project_json` returns `{}` if the file is missing **or** if it
fails to parse; a corrupt `project.json` is now logged at `WARNING` (it used
to swallow the exception with no trace at all — fixed in this pass).

## `data/fs/metadata.py`

`FileMetadata` dataclass for the **legacy** content-addressed scheme
(`sha256`, `original_filename`, `mime_type`, `char_count`, `chunk_count`,
`parser`, `aliases`, …). `read_file_metadata`/`update_file_metadata` persist
it as `metadata.json` inside `file_dir(pid, sha256)`. `update_file_metadata`
only mutates fields that already exist on the dataclass (`hasattr` check) —
unknown kwargs are silently ignored by design (callers pass named fields like
`chunk_count=`, `char_count=`), now logged at `WARNING` if an unknown field is
passed so a typo'd caller doesn't lose the write silently. `read_file_metadata`
also now logs at `WARNING` on a corrupt/unreadable `metadata.json` instead of
returning `None` with no trace.

## `data/fs/files.py`

The **legacy** content-addressed file store: `store_file`, `list_files`,
`delete_file`. Stores uploaded bytes at `files/<sha256[:12]>/<safe filename>`;
re-uploading the same content under a different name renames the on-disk file
to the latest name and keeps the old name in `metadata.aliases`. Used
exclusively by the old data-prep ingestion path
(`data/prep/ingest.py:parse_and_chunk`, `data/prep/runner.py`) via
`finetune_studio.data.project_filesystem.store_file` — **not** by the file
browser UI, which uses `file_library.py` instead. If an existing
`metadata.json` fails to parse during a re-upload merge, it used to be
silently discarded (losing alias history with no trace) — now logged at
`WARNING` before falling back to an empty dict.

## `data/fs/chunks.py`

`write_chunks(pid, sha256, chunks, chunk_meta=None)` — writes
`files/<sha12>/chunks/NNNN.txt` plus a `manifest.json` with one entry per
chunk (`index`, `char_count`, `source_section`, `chunk_path`). Clears any
previous chunk files first (`*.txt` glob + unlink) since a fresh parse is
assumed to fully replace the old chunk set. Manifest length always matches
`len(chunks)` — no off-by-one or truncation observed.

## `data/fs/parsed.py`

`write_parsed_outputs(pid, sha256, parsed_text, parsed_json)` — writes
`files/<sha12>/parsed.txt` and `parsed.json`. Trivial, no validation; callers
(`data/prep/ingest.py`) are responsible for calling this only after a
successful parse.

## `data/fs/ingestion.py`

Append-only `logs/ingestions.jsonl` audit trail per project.
`log_ingestion(pid, event)` stamps `event["timestamp"]` if missing and
appends one JSON line. `read_ingestion_log(pid, limit=200)` tails the last
`limit` lines and parses each as JSON; malformed lines were silently dropped
with no indication the log is incomplete — now logs a single `WARNING` with
the count of skipped lines per call.

## `data/fs/qa.py`

On-disk Q&A pair + source storage, parallel to (and older than) the DB-backed
pipeline — `qa/pairs/<qa-id>.json` and `qa/sources/<source-id>.json` per
project. Key functions:
- `should_auto_promote` / `maybe_auto_promote_upload` — text uploads
  (`.txt`/`.md`/`.markdown`/`.log`) are auto-registered as data-prep sources
  on upload so they show up in the source picker without a second request.
  Failures are caught and logged (`log.warning`), returning `None` rather
  than raising — a deliberate best-effort contract, documented on the
  function.
- `stage_file_library_upload` / `parse_staged_qa_source` — the two-step path
  bulk uploads use: stage a `status: "queued"` manifest cheaply in the
  request path, then parse it later in a background task so a page refresh
  shows `queued` instead of the file "vanishing".
- `register_qa_source` — idempotent on `(pid, path)`; re-registering an
  existing ready source returns it as-is, an existing-but-unparsed source is
  re-ingested.
- `write_qa_pair`/`write_qa_source`/`read_qa_source`/`list_qa_pairs`/
  `list_qa_sources`/`update_qa_pair`/`delete_qa_source` — straightforward
  JSON-file CRUD. **All of the list/read functions used to silently drop any
  pair or source file that failed to parse as JSON**, with a bare
  `except: continue` and zero logging — a listing could silently return
  fewer items than actually exist on disk with no way to tell. Fixed in this
  pass: each of these now logs a `WARNING` with a count of skipped files
  (`list_qa_pairs`, `list_qa_sources`) or the specific id (`read_qa_source`,
  `update_qa_pair`). `delete_qa_source` had the same problem in the opposite
  direction — a pair file it couldn't parse would never be matched against
  `source_id` and therefore never deleted, silently leaving orphaned pairs
  behind after "delete source and all its pairs"; now logs a `WARNING`
  naming the file that was skipped.

## `data/fs/workbench.py`

File-browser backend for bulk/aggregate operations, built on top of
`file_library.py` (`fl`) and the `project_filesystem` shim (`pfs`):
- `bulk_action(pid, ids, action, payload)` — applies delete/restore/move/
  reparse/tag-add/tag-remove to up to 500 ids, per-id isolated (one failure
  doesn't abort the batch); returns `{requested, succeeded, failed, results}`
  so counts always reconcile with the input.
- `download_zip(pid, ids)` — zips raw bytes for up to 500 ids, capped at 2 GiB
  total, with collision-safe arcnames (`_unique_arcname`). Raises 413 if the
  cap would be exceeded, 404 if nothing was readable.
- `file_usage(pid, file_id)` — traces one file through source → QA pairs →
  datasets (`_jsonl_contains_ids`, capped at a 20k-line scan) → training runs,
  plus RAG corpus membership.
- `search_content(pid, q, limit=25)` — substring search over parsed text,
  bounded by `SEARCH_MAX_FILES=400`; reports `scanned`, `skipped_unparseable`,
  and `truncated` so a partial result is always visible to the caller, not
  silent. This file is a good model for the "always report what you
  skipped" pattern the rest of the module is missing in places.

## `data/fs/file_library.py` (1443 lines — the new file-library backend)

DB-backed file library: `project_files` / `file_versions` / `file_conversions`
/ `file_folders` / `folder_membership` tables (schema owned by the DB lane,
not this lane) plus a disk layout under `files/raw/{auto_kind}/` (MIME-routed,
system-managed, immutable per version) and `files/converted/{user_folder}/`
(user-organized). Soft-delete moves bytes to `files/raw/.RAW_TRASH/` or
`files/converted/.CONVERTED_TRASH/` and sets `project_files.deleted_at`;
restore reverses it; `purge_trash`/`purge_file` hard-delete.

Key functions: `auto_kind_for` (MIME→bucket routing), `raw_path_for`/
`converted_path_for` (naming conventions — raw filenames are always
`{file_id}_{safe_stem}.{ext}` so trash/rename logic can glob by file id),
`record_uploaded_file`/`write_uploaded_file`/`write_staged_upload` (insert +
disk write, with cross-project file-id collision retry and soft-deleted-row
"revival" on re-upload — see QABUG 2026-09-18 in the code comments),
`list_files`/`list_folders`/`get_file`/`list_versions`/`list_conversions`
(read-only DB queries), `create_folder`/`rename_folder`/`delete_folder`/
`move_file_to_folder` (folder management, with MIME-bucket invariants
enforced in `move_file_to_folder`), `soft_delete_file`/`restore_file`/
`purge_trash`/`purge_file`/`list_trash` (the trash lifecycle), `rename_file`,
and `get_parsed_markdown` (the parsed-content resolution chain described
below).

### `get_parsed_markdown` resolution order
1. Optional `parsed_md`/`parsed_path` columns on `project_files` (checked live
   via `PRAGMA table_info` since the current schema doesn't have them yet).
2. `file_conversions` row with `format` md/txt/markdown and `status != error`.
3. Manual override file `<raw_name>.parsed.md` written by the parsed-text
   editor (`data/parsed_edit.py`).
4. Sibling `.md` next to the raw file (both `{raw_name}.md` and
   `{safe_stem}.md` are checked).
5. Legacy `files/<sha12|sha16|file_id>/parsed.txt` (the old content-addressed
   scheme's output, for files registered as a data-prep source).
6. On-the-fly conversion (`_convert_raw_to_md`) for text-like extensions; 422
   for binary formats; 410 if the raw bytes are missing entirely.
Results are cached per-process in `_PARSED_CACHE` keyed by `"pid:fid"`;
`invalidate_parsed_cache` must be called after rename/reparse or the cache
serves stale content.

## `data/fs/migrate_legacy_files.py`

Standalone one-off CLI (`python -m finetune_studio.data.fs.migrate_legacy_files
[project_id]`), **never imported by anything else in the app** (confirmed by
grep). Walks `files/<hash>/metadata.json` under the legacy content-addressed
layout and `INSERT OR IGNORE`s a row into `project_files` so old files show up
in the file-library UI. Reports `(inserted, skipped, errors)` per project and
prints per-item progress — it does not silently drop candidate files: a
missing/corrupt `metadata.json` is counted as an `error` and printed. Hash-dir
name filtering (`all(c in "0123456789abcdef" ...)`) correctly excludes the
new file-library's `raw/`/`converted/` subdirectories (which aren't meant to
be migrated — they're already in the DB from upload time) but excludes them
from every counter with no trace; this is a legitimate design choice (those
aren't migration candidates) rather than a bug, but a future reader scanning
printed output for "did it see my file_library files" should know this script
only looks at the legacy sha-dir layout. `DB_CANDIDATES` includes a
machine-specific absolute path (`/home/genortg/finetune-studio/...`) as one of
three fallback locations — worth knowing if this script is ever run somewhere
that path doesn't apply (it just won't be selected, `find_db()` tries the
next candidate).

## `data/fs/__init__.py`

Re-exports the public API of the legacy scheme (`chunks`, `files`, `ingestion`,
`metadata`, `parsed`, `paths`, `project`, `qa`) for `data/project_filesystem.py`'s
back-compat shim. **Discrepancy fixed in this pass**: the entire import block
was duplicated verbatim (lines 46–68 repeated byte-for-byte at 69–91, the
second copy wrapped in `# noqa: F401, F811` to suppress the redefinition
lint it created) — pure dead weight with no behavioral difference, just two
copies of the same 8 import statements. Collapsed to one copy with the
necessary `# noqa: F401` markers.

## `data/project_filesystem.py`

Back-compat shim: re-exports the legacy `data/fs/*` API (including
`data/fs/files.py`'s `list_files`/`store_file`/`delete_file`, **not**
`file_library.py`'s functions of the same names) under the historical
`finetune_studio.data.project_filesystem` import path. Actively used — not
dead — by `data/prep/ingest.py`, `data/prep/runner.py`, `data/parsed_edit.py`,
`data/audit.py`, `data/prep/coverage_fill.py`, `data/prep/source_state.py`,
`data/prep/export.py`, `data/fs/workbench.py`, and several `webui/routes/*`
files, almost always imported as `pfs`.

## `data/organizer.py`

Standalone directory scanner/deduper used by `webui/routes/data.py`
(`GET` file-browser listing + a dedup action), unrelated to the per-project
`fs/` layout — it scans `settings.data_dir` directly. `scan_data_files`
walks a directory for `.jsonl`/`.json`/`.csv`/`.txt` files, skipping
dot-directories, sorted newest-first. `dedup_data(data)` dedupes a list of
dict items by `hash(json.dumps(item, sort_keys=True))` and returns
`(unique, dupes_count)` — correctness note: Python's built-in `hash()` is not
collision-free, so in principle two distinct items could theoretically hash
identically and one would be wrongly treated as a duplicate; the actual
callers pass small QA-pair-sized datasets where this is not a practical risk,
so it was left as-is rather than "fixed" by swapping in sha256 (that would be
a behavior-preserving style change, not a bug fix, for the datasets this is
actually called on).

## `data/shared_models.py`

Cross-project shared model cache for embedders/rerankers
(`~/.finetune-studio/shared_models/{embedders,rerankers}/<name>@<hash16>/`).
`register()` downloads-or-copies a model into the shared store keyed by
content hash (`content_hash` — hashes `model.safetensors` if present, else the
whole tree) so two different "all-MiniLM-L6-v2" versions with different
weights get distinct cache entries. `resolve(short_id, kind)` looks the dir
up and bumps `META.json.use_count` via `_touch_use` (best-effort — a corrupt
`META.json` there is caught and ignored by design, since a use-count miss
must never block loading a model). `stats()` — used by the model-store UI —
reads every `META.json` with a bare `json.loads(...)` (no try/except); a
single corrupted `META.json` among many installed models would raise and
fail the whole stats listing rather than skipping just that one entry. Left
undocumented-as-a-gotcha-only (not fixed) since it's an unguarded read rather
than a silent drop, and fixing it would mean deciding on behavior (skip vs.
surface) that's a product call outside this audit's scope.

## Wiring summary

- Legacy content-addressed scheme (`files.py`/`chunks.py`/`parsed.py`/
  `metadata.py`, all under `data/fs/`) is driven by `data/prep/ingest.py` and
  `data/prep/runner.py` via the `project_filesystem` shim — this is the
  RAG/data-prep source-ingestion pipeline (parse → chunk → Q&A generation).
- New file-library scheme (`file_library.py`) is driven by
  `webui/routes/file_library.py`-style route handlers and `workbench.py` —
  this is the file-browser UI (upload, folders, rename, soft-delete/trash,
  search, bulk actions, zip export).
- `qa.py` bridges the two: `promote_file_library_upload`/
  `maybe_auto_promote_upload` take a file-library `file_id`, resolve its
  current raw bytes via `file_library.list_versions`, and register it as a
  **legacy-scheme** QA source (keyed by raw sha256, not the file-library
  `file_id`) so the data-prep pipeline can parse/chunk it.
- `shared_models.py` is independent of per-project storage; it's consulted by
  `data/rag_portable/embedders.py` and `rerankers.py` when building a RAG
  corpus.

## Gotchas / invariants

1. **Two storage schemes, one `files/` directory.** `files/<sha12>/...` (flat,
   content-addressed, legacy) and `files/raw/{kind}/...` +
   `files/converted/{folder}/...` (file-library, DB-backed) coexist under the
   same project `files/` root. `data.fs.files.list_files` and
   `data.fs.file_library.list_files` are two functions with the same name,
   different signatures, different return types, and different data sources
   — they are not duplicates of each other in the "delete one" sense, they
   serve genuinely different call sites. Do not assume `from
   finetune_studio.data.fs import list_files` gets you the file-library data;
   `data/fs/__init__.py` only re-exports the legacy one.

2. **Stale `raw_path`/`converted_path` after soft-delete (now the subject of
   a fixed bug).** `soft_delete_file` physically moves a file into
   `.RAW_TRASH`/`.CONVERTED_TRASH` but **never updates**
   `file_versions.raw_path` or `file_conversions.converted_path` in the DB —
   those columns keep pointing at the pre-trash location for as long as the
   row stays soft-deleted. This is a deliberate (if easy to miss) invariant:
   `restore_file`, `purge_file`, and `_current_raw_path` all know about it and
   fall back to globbing the trash directory by `file_id` prefix when the
   recorded path doesn't exist. **`purge_trash` (the time-based bulk sweep)
   did not know about it** — it checked `Path(db_raw_path).exists()` before
   unlinking, which was always `False` post soft-delete, so its unlink
   branches were dead code: the DB rows were deleted and the call reported
   success (`{"purged": [...], "count": N}`) while the actual bytes stayed in
   `.RAW_TRASH`/`.CONVERTED_TRASH` forever — a permanent, silent disk leak on
   every scheduled trash sweep. **Fixed in this pass**: `purge_trash` now
   also globs the trash directories directly (mirroring `purge_file`'s
   already-correct logic) and reports any unlink failures in a
   `disk_errors` list instead of swallowing them. Any future code that reads
   `raw_path`/`converted_path` for a possibly-deleted file must go through
   `_current_raw_path`-style trash-fallback logic rather than trusting the DB
   column directly.

3. **Corrupt on-disk JSON used to vanish silently.** Every JSON-backed
   listing/read in `qa.py`, `metadata.py`, `project.py`, and `ingestion.py`
   caught parse failures with a bare `except: continue`/`return None`/
   `return {}` and no logging — a corrupted pair/source/metadata/log-line
   file would simply disappear from results with zero trace that anything
   was wrong. All of these now log a `WARNING` (with a per-call skipped-count
   for the listing functions). This does not change return values for valid
   data; it only adds visibility for the corrupt-file case.

4. **`file_id` naming convention carries real meaning.** File-library raw
   filenames are always `{file_id}_{safe_stem}.{ext}`
   (`raw_path_for`/`_safe_stem`) — this is what lets `restore_file`,
   `purge_file`, `_current_raw_path`, and now `purge_trash` find a file in
   `.RAW_TRASH` by globbing `{file_id}_*` even when the DB's recorded path is
   stale or missing. Converted-file trash entries are **not** guaranteed to
   contain `file_id` in their name (comment in `purge_file`); matching there
   is a best-effort substring check, not a guarantee.

5. **`update_file_metadata`/`_apply_tag`-style "merge only known fields"
   helpers fail open, not closed.** Passing an unrecognized field name is a
   silent no-op (now logged) rather than a `TypeError` — this is intentional
   (the dataclass is the schema), but it means a typo in a caller's kwarg
   name will not be caught by any test unless that test asserts on the
   persisted value.
