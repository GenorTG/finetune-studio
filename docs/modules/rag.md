# RAG subsystem

Finetune Studio ships **two** RAG implementations that genuinely coexist for
historical reasons rather than being a pure accidental duplicate: the
actively-developed, file-based `finetune_studio.data.rag_portable/` package
(the system behind the real project RAG UI), and a legacy ChromaDB-backed
`finetune_studio.rag/` package that is still imported by three live route
files but has zero test coverage and essentially no ongoing development.
Both let a project search documents by similarity; they are not
interchangeable and should not both be "the" RAG system going forward — see
**"The duplication question" below** for the full verdict and what's left to
decide.

## The duplication question (read this first)

**Verdict: `finetune_studio.rag/` is legacy-but-NOT-dead. It cannot be
deleted outright; it needs a deliberate migration, which is outside this
lane's file scope.**

Evidence gathered by grepping the *entire* repo (not just this lane's file
list) for every import form of both packages:

```
grep -rn "from finetune_studio\.rag\b\|import finetune_studio\.rag\b\|finetune_studio\.rag\." --include="*.py" .
```

Live, reachable call sites of `finetune_studio.rag` (the legacy Chroma
package):

| File | Uses | Reachable how |
|---|---|---|
| `webui/routes/projects.py:24,258,294,310` | `RAGManager` | `POST/GET /api/projects/{pid}/rags/{rid}/ingest\|query\|stats` — mounted, but **no frontend template calls these URLs** (`grep -rn "/rags/" src/finetune_studio/webui/templates/` → no hits). Looks abandoned from the UI's perspective but is still a live API surface. |
| `webui/routes/comparison.py:79` | `VectorStore` | `POST /api/compare/rag/chat` — mounted at `/api/compare`, but **no `comparison.html` template exists and no frontend references `/api/compare` at all** — this whole router looks orphaned from the UI, RAG usage included. |
| `webui/routes/chat_v2.py:44` | `VectorStore` (fallback) | `_search_rag_attachment()` checks `_is_portable_corpus(store_path)` (looks for `manifest.json` + `vectors.npy`) and only falls back to the old Chroma `VectorStore` when a stored corpus is **not** in PortableRAG format — i.e. a deliberate back-compat shim for pre-migration corpora, not accidental reuse. |
| `data/rag_portable/store.py:100` | `from finetune_studio.rag.ingest import chunk_text` | `PortableRAG.build_from_directory()` reaches into the legacy package to reuse its word-based chunker instead of having its own. This is the one place the "portable" system has a **hard runtime dependency on the legacy package** — deleting `rag/ingest.py` would break `rag_portable`. |
| `cli/commands/rag.py`, `cli/commands/rag_test.py` | `RAGManager`, `VectorStore`, `RAGQuery` | CLI commands (`finetune-studio rag ...`) still wired to the old package. |

Confirming the live/primary system is `rag_portable`: the actual project RAG
page (`templates/rag.html`) calls **only** `/api/projects/{pid}/rag/*`
(singular `rag`, not `rags`), all served by `webui/routes/rag.py` and
`webui/routes/project_rag.py` — both of which import exclusively from
`finetune_studio.data.rag_portable` and never touch `finetune_studio.rag` at
all. `chat_v2.html`'s chat route also resolves to `rag_portable.PortableRAG`
for any corpus built since the portable migration.

Supporting signal: `git log --oneline -- src/finetune_studio/rag/` → **2**
commits ever (creation + one ingest fix, 2026-09-07) vs.
`git log --oneline -- src/finetune_studio/data/rag_portable/` → **20**
commits, and zero test files import `finetune_studio.rag.{manager,store,query,ingest}`
directly (only `rag_portable` has a large dedicated test suite:
`test_rag_embedder_dims.py`, `test_rag_export_isolation.py`,
`test_rag_import_bundle.py`, `test_rag_mcp_package.py`,
`test_rag_mime_ingestion.py`, `test_rag_rebuild_sources.py`,
`test_rag_source_labels.py`, `test_rag_eval.py`, `test_rag_build_registration.py`).

**Why it was NOT deleted in this pass:** the task's own safety rule is to
delete only when grepping the whole repo shows zero real callers. Three
files outside this lane's scope (`projects.py`, `comparison.py`, `chat_v2.py`)
still import it live, and `rag_portable/store.py` (in-scope) has a runtime
import dependency on `rag/ingest.py`. Deleting it would break `chat_v2.py`'s
legacy-corpus fallback and `rag_portable`'s chunker import. This is
filed below as a **cross-module finding for the parent/owner to decide**:
whether to (a) migrate `projects.py`'s `/rags/{rid}/*` endpoints and
`comparison.py` onto `rag_portable` and delete `finetune_studio.rag/`
entirely, or (b) keep it intentionally as the back-compat layer for
pre-portable-format corpora and document it as such.

`data/rag.py` and `data/rag_eval.py` are **not** part of the duplication —
they are CLI/eval tooling built entirely on top of `rag_portable` (see below).

## Files

### `data/rag_portable/` — the live portable RAG (file-based corpus)

One directory per corpus (`manifest.json`, `chunks.parquet`, `vectors.npy`,
`vectors.idx.json`, `bm25.json`, `sources/<doc_id>.txt`). No service, no
network, no vendor DB — "just a dir" that can be tarred, copied, and loaded
anywhere. This is what `webui/routes/rag.py`, `webui/routes/project_rag.py`,
and `webui/routes/chat_v2.py` (for post-migration corpora) actually use.

- **`constants.py`** — `SCHEMA_VERSION`, default embedder
  (`intfloat/multilingual-e5-large`) and reranker
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`) names, `RRF_K=60`, and the magic
  `embedder_local:` / `reranker_local:` prefixes that mean "load from a path
  next to the corpus" rather than fetch from HuggingFace.
- **`tokenize.py`** — one function, `tokenize()`: unicode-aware lowercase
  word tokenizer shared by BM25 (ASCII words + CJK/high-unicode runs).
- **`io.py`** — `read_json`/`write_json`/`try_import_pandas` (raises a clear
  `RuntimeError` if pandas/pyarrow are missing, since chunks are stored as
  parquet).
- **`rrf.py`** — `rrf_fuse(ranked_lists, k)`: classic Reciprocal Rank Fusion,
  `score[doc] = sum(1/(k+rank))` across all input rankings.
- **`bm25.py`** — `BM25Index`: classic Okapi BM25 (k1=1.5, b=0.75), built
  in-memory from a list of doc texts, serializable via `to_dict`/`from_dict`
  for persistence in `bm25.json`. `score(query)` returns a float32 array over
  all docs.
- **`schema.py`** — dataclasses for the manifest: `EmbeddingModelInfo`,
  `RagSettings` (embedder/reranker refs, `rerank_top_n=50`,
  `hybrid_enabled`, `rrf_k`), `ChunkSettings` (size=400, overlap=80,
  word-based), and `Manifest` itself (documents/chunks counts, `extra` dict
  used for `documents_meta` and `shared_model_paths`).
- **`shared_refs.py`** — `resolve_model_ref(name, kind)`: translates
  `shared:embedder:<short_id>` / `shared:reranker:<short_id>` manifest refs
  (pointers into the cross-project shared model store,
  `finetune_studio.data.shared_models`) into the `embedder_local:`/
  `reranker_local:` paths the loaders understand. Also strips historically
  malformed prefixes like `sentence-transformers/shared:...`. Resolve
  failures are swallowed deliberately (returns the name unchanged) so the
  downstream load produces a clearer error than this module could fabricate.
- **`source_labels.py`** — turns opaque content-addressed paths
  (`files/<sha12>/parsed.txt`, `chunks/0000.txt`) into human-readable
  citation labels using `metadata.json`'s `original_filename`. Also owns
  `should_ingest_source_file()`, the gate that decides which files under a
  project's file tree get indexed: skips `chunks/*.txt` shards, skips
  `.RAW_TRASH`/`.CONVERTED_TRASH`, and — critically — skips
  `files/raw/<id>_<name>` library copies when a canonical
  `files/<sha12>/parsed.txt` already exists for the same content hash, so a
  document isn't indexed twice (raw + parsed) under different opaque names.
- **`embedders.py`** — `get_embedder(name, device)` returns
  `(encode_fn, EmbeddingModelInfo)`. Loads from a local
  `embedder_local:<path>` dir (via `prepare_local_sentence_transformer_dir`,
  which repairs incomplete shared-store copies missing
  `1_Pooling/config.json`) or fetches from HuggingFace into the canonical
  `hf_cache_dir()` (not `/tmp`). `encode()` always L2-normalizes.
- **`rerankers.py`** — `get_reranker(name, device)` returns
  `(rerank_fn, name)` using `sentence_transformers.CrossEncoder`. Strips the
  same malformed library prefixes as `shared_refs.py`, then resolves
  `shared:reranker:...` refs the same way.
- **`query.py`** — `PortableRAGQuery`: the loaded, in-memory corpus + search
  pipeline. `search()` does dense top-N (cosine via normalized dot product)
  + BM25 top-N → RRF fuse → optional cross-encoder rerank over the top
  `rerank_top_n` candidates → re-numbered top-k. Raises `ValueError` on
  query/corpus embedding dimension mismatch instead of silently truncating
  or erroring deep in numpy. **Audit fix (2026-10-01):** previously called
  `self.bm25.score(query)` once per *candidate result* inside the formatting
  loop — up to `rerank_top_n` (50) redundant full BM25 passes per search —
  instead of reusing the single `bm25_scores` array already computed for
  ranking. Now computed once and indexed.
- **`store.py`** — `PortableRAG`: owns the on-disk layout and the
  build/load/export lifecycle.
  - `build_from_directory()` — parse (via `finetune_studio.data.parsers`,
    33+ formats) → chunk (via `finetune_studio.rag.ingest.chunk_text` — see
    duplication note above) → embed → BM25 → write all five artifact files.
    **Always replaces** corpus content; it is not an append. Collects parse
    results into a `pending` list *before* wiping `sources/`, specifically so
    a run where every file fails to parse does not destroy an existing
    working corpus.
  - `bundle_models()` / `unshare_models()` — copy the shared embedder/
    reranker into the corpus dir (or back out) so an exported corpus can run
    fully offline with no model download.
  - `export_bundle()` / `import_bundle()` — legacy plaintext tar/zip (library/tests only; `out_path` required).
  - `secure_bundle.py` — `export_secure_bundle()` / `import_secure_bundle()`: encrypted `.ftsrag` studio bundle (the only UI/API export path).
    Import uses `tarfile.extractall(..., filter="data")` (traversal-safe) and
    manually validates zip member paths stay inside the staging dir before
    extracting (zip's `extractall` has no built-in path-traversal filter).
  - `load()` — refuses to silently fall back to a different embedder on a
    dimension mismatch; raises with a clear corpus-vs-embedder diagnostic
    instead (a deliberate anti-footgun, called out in its own comment).
  - **`remove_source(source_id)` — audit fix (2026-10-01), this was the
    single most serious bug found in this lane.** Previously it only
    `unlink()`'d the raw `sources/<id>.txt` file and returned `True`/`False`.
    It never touched `chunks.parquet`, `vectors.npy`, `bm25.json`,
    `vectors.idx.json`, or the manifest's `documents_meta`/counts. The
    `DELETE /{pid}/rag/sources/{source_id}` route
    (`webui/routes/rag.py:438-449`, **outside this lane's scope but the only
    caller**) returned `{"ok": True, "removed": source_id}` while the
    "removed" source's chunks stayed fully indexed and kept showing up in
    both `search()` results and `list_sources()` — a textbook
    "claims success, silently keeps the data" bug. Fixed to filter
    `chunks.parquet` + `vectors.npy` by a boolean mask, rebuild
    `vectors.idx.json` and the BM25 index from the surviving chunks, and
    drop the matching `documents_meta` entry (updating `manifest.documents`/
    `chunks` counts). `clear_sources()` (the bulk sibling) was also
    incomplete: it deleted source text and reset manifest
    metadata but left chunks/vectors/BM25 searchable. Fixed in this
    continuation (2026-10-01) to empty the parquet rows and vectors, reset
    the vector map and BM25 index, and clear manifest counts/source metadata
    while retaining corpus/model files.
  - `rebuild_vectors()` — re-embeds all chunks with a (possibly different)
    embedder and rebuilds BM25 too ("cheap, keeps state consistent").
- **`mcp_package.py`** — `build_package(corpus, out, *, name, fmt, include_models,
  config, encrypt, passphrase) -> PackageResult` builds a hostable,
  self-installing export (`tar`/`tar.gz`/`zip`, streamed to `<out>.part` then
  renamed; nothing plaintext is staged on disk). Ships `server.py` (verbatim
  `standalone_server.py`), `rag.config.json`, `install.sh`/`run-http.sh`/
  `run-mcp.sh`/`setup.sh`, an MCP config snippet, README, `requirements.txt`.
  **Encrypted by default** (AES-256-GCM chunked AEAD, key = scrypt(passphrase);
  the key is never shipped or stored): corpus lives in `corpus/corpus.enc`
  (chunks, sources, vectors, BM25, doc names/metadata); only a minimal header
  (version, KDF params, salt, nonce scheme) is plaintext, and `rag_container.py`
  ships alongside. A blank passphrase generates one, returned once in
  `PackageResult.passphrase` (`repr=False`). `encrypt=False` is the explicit
  opt-out (README says NOT ENCRYPTED; route adds `-PLAINTEXT` to the filename);
  the legacy plaintext layout stays readable. **Model weights are never
  encrypted** (README says so). `include_models=True` streams the shared
  embedder/reranker in for offline semantic search.
- **`rag_container.py`** — the container format (`ContainerWriter`/
  `ContainerReader`): 64 KiB frames, nonce = 8-byte prefix + be32 counter,
  AAD = sha256(header)+kind+counter, encrypted index in a trailer. Decrypts in
  memory only. Wrong passphrase / any flipped byte / truncation raises
  `WrongPassphraseOrTampered` (server exits 3). Shipped verbatim; must not
  import `finetune_studio`.
- **`export_config.py`** — `RagExportConfig` (archive_format, include_models,
  include_reranker, reranker_enabled, device, host, port, top_k, encrypt,
  kdf_log_n). Layers: defaults < studio settings (`rag_export`) < project
  override (`rag_export_projects[pid]`) < request. Never holds a passphrase.
  API: `GET/PUT /projects/{pid}/rag/export-config`,
  `POST /projects/{pid}/rag/mcp-package` (JSON; passphrase in the body, never a
  URL), `GET .../mcp-package/download?file=`.
- **Runtime config (shipped server)** — precedence flags > env (`RAG_*`;
  `RAG_EMBED_MODEL/BASE_URL/API_KEY/DEVICE` still work) > `rag.config.json` >
  defaults. Flags: `--config --device --host --port --top-k --no-reranker
  --keyfile --print-config` (redacted effective config + origin per key).
  Passphrase: prompt, `RAG_PASSPHRASE`, or `--keyfile`. **Bind is 127.0.0.1;
  a non-loopback host requires an auth token (bearer, constant-time compare) or
  the server refuses to start (before any passphrase prompt). No CORS.**
- **`standalone_server.py`** — the file that gets copied into every export.
  **Deliberately duplicates** `tokenize()` and BM25 scoring logic from
  `tokenize.py`/`bm25.py` rather than importing them — this is required, not
  an oversight: the exported package must run with zero dependency on the
  `finetune_studio` package being installed. Implements the same dense +
  BM25 + RRF + optional rerank pipeline as `PortableRAGQuery.search()`, plus
  HTTP (`/health`, `/search`) and MCP-stdio (`rag_search`, `rag_info`) server
  modes. Embeds via, in order: bundled `corpus/embedder/` → an
  `RAG_EMBED_BASE_URL` OpenAI-compatible endpoint → keyword-only BM25.
- **`__init__.py`** — public API re-exports (`PortableRAG`,
  `PortableRAGQuery`, `BM25Index`, `get_embedder`, `get_reranker`,
  `rrf_fuse`, `tokenize`, `read_json`/`write_json`, constants and schema
  dataclasses).

### `data/rag_portable.py` — back-compat shim

Single-purpose re-export module so `from finetune_studio.data.rag_portable
import PortableRAG` (the module, pre-package-split spelling) keeps working
now that `rag_portable` is a package (`rag_portable/`). No logic of its own.

### `data/rag.py` — portable RAG CLI

`python -m finetune_studio.data.rag {build,search,eval,rebuild-vectors,info}`.
Thin argparse wrapper entirely over `rag_portable.PortableRAG` and
`rag_eval.run_eval_on_corpus` — not part of the rag/rag_portable
duplication, it's tooling for the portable system. Referenced directly from
`rag_portable/store.py`'s generated corpus README ("Migrate / re-embed"
section), so it's a real, documented entry point, not dead.

**Audit fix (2026-10-01):** removed a module-level `run_eval_on_corpus()`
function that duplicated (and silently diverged from — it hardcoded
`run_portability`/`use_llm` to defaults that `_cmd_eval` does **not** use)
the real CLI path. `_cmd_eval` has always called
`rag_eval.run_eval_on_corpus` directly; the module-level shim had zero
callers anywhere in the repo (verified via
`grep -rn "run_eval_on_corpus" --include="*.py" .` across the whole tree)
and existed only as unreferenced "back-compat" for an import pattern nothing
used.

### `data/rag_eval.py` — evaluation harness

Retrieval/grounding/portability test suite, built entirely on
`rag_portable.PortableRAG`. Notably careful about **not** mislabeling
metrics — the module docstring and several back-compat aliases
(`FactCoverageResult` aka `LLMJudgeResult`, `llm_pass_rate` aka
`fact_coverage_pass_rate`) exist specifically because an earlier version of
this code called a plain substring `must_contain` check "LLM-as-judge",
which it never was; the dataclasses and `EvalMetadata.legacy_llm_pass_rate_alias`
field document the correction in-place rather than silently renaming and
losing the history. Key entry points: `run_rag_evaluation()` (the real
work — retrieval recall@k/MRR, optional lexical grounding, optional
substring fact-coverage, optional no-context-refusal check, optional tar
round-trip portability test) and `run_eval_on_corpus()` (the CLI-facing
wrapper that loads a corpus + QA JSON file and writes
`<corpus>/tests/<timestamp>/results.json` plus an appended
`tests/history.json`). No bugs found here — this file reads as the most
carefully self-audited file in the lane.

### `rag/` — legacy ChromaDB-backed RAG (still live, not dead)

See "The duplication question" above for the full reachability analysis.

- **`ingest.py`** — `Document`/`Chunk` dataclasses, `extract_text()`
  (delegates to the unified `finetune_studio.data.parsers` package — the
  docstring explains this was a deliberate rewrite because the previous
  version only handled PDF/DOCX and silently ingested images as raw binary),
  and `chunk_text()` — a **word-based, fixed-size overlapping chunker that is
  the one `rag_portable/store.py` imports and reuses** (see duplication
  note). `ingest_file()`/`ingest_directory()` walk a directory, parse, chunk,
  and build `Document` objects; parse failures are caught per-file and
  logged to stdout, not raised (acceptable here — the directory-level
  ingest is expected to skip bad files and continue).
  **Audit fixes (2026-10-01):** removed `extract_pdf()`/`extract_docx()` —
  zero callers anywhere in the repo, and each swallowed its exception into a
  string like `f"[extract_pdf failed: {e}]"` that would then be ingested as
  if it were real document content (a silent-corruption pattern, now moot
  since the functions are gone). Also removed a duplicate `continue`
  statement in `ingest_directory()`'s file loop (harmless but clearly a typo
  — the second `continue` was unreachable dead code).
- **`store.py`** — `VectorStore`: ChromaDB-backed, with cosine similarity
  search, `add_chunks()`/`search()`/`remove_document()`/`list_documents()`/
  `clear()`. **Audit fix (2026-10-01):** `_get_embedder(embedding_model)`
  took the parameter but completely ignored it, always loading
  `all-MiniLM-L6-v2` regardless of what was requested. `comparison.py`
  passes `settings.rag.embedding_model` into `VectorStore.search()`
  expecting it to matter; it silently didn't. Fixed to cache one
  `SentenceTransformer` per requested model name instead of a single
  hardcoded singleton.
- **`query.py`** — `RAGConfig` (`top_k=5`, `min_score=0.3`,
  `max_context_length=2000`) and `RAGQuery`: retrieve → filter by
  `min_score` → build a context string (truncated at `max_context_length`,
  stops adding chunks once the budget is hit rather than silently
  overflowing) → augment chat messages → call the inference engine.
- **`manager.py`** — `RAGManager`: thin facade over `store.py` + `ingest.py`
  (`ingest_file`, `ingest_directory`, `remove_document`, `list_documents`,
  `stats`, `clear`). This is what `webui/routes/projects.py`'s
  `/rags/{rid}/ingest|query|stats` endpoints instantiate per-request.
- **`__init__.py`** — re-exports `VectorStore`, `SearchResult`, `Document`,
  `Chunk`, `chunk_text`, `ingest_file`, `ingest_directory`, `RAGConfig`,
  `RAGQuery`, `RAGManager`.

## Gotchas / invariants for future engineers

- **Don't delete `finetune_studio.rag/` without first migrating
  `webui/routes/projects.py`, `webui/routes/comparison.py`, and
  `webui/routes/chat_v2.py`'s legacy-corpus fallback, AND giving
  `rag_portable/store.py:100` its own `chunk_text` (or importing from
  wherever `rag/ingest.chunk_text` ends up).** It is not currently dead code.
- `PortableRAG.remove_source()` is the only safe way to delete one source —
  it now keeps chunks/vectors/bm25/manifest in sync. Don't hand-delete files
  under `sources/` directly; that reproduces the exact bug fixed above.
- `clear_sources()` clears searchable indexes and source text but retains
  corpus files and bundled models. Rebuild from the intended source directory
  to repopulate the corpus.
- A corpus's embedder is load-bearing for its `vectors.npy` — `load()`
  deliberately raises instead of silently substituting a different-dimension
  embedder. If you see a dimension-mismatch error, rebuild the corpus or fix
  the manifest's embedder reference; don't try to "fix" it by relaxing the
  check.
- `standalone_server.py`'s duplication of tokenizer/BM25 logic from
  `tokenize.py`/`bm25.py` is intentional (exported packages must be
  dependency-free), not a "one true way" violation — do not refactor it to
  import from the main package.
- `should_ingest_source_file()` in `source_labels.py` is the single gate
  that prevents double-indexing raw uploads and their parsed twins; any new
  ingestion code path should route through it rather than re-deriving its
  own "which files count as a source" logic.

## Cross-module findings — NOT fixed, needs parent coordination

1. **`webui/routes/projects.py:24,258,294,310`** (outside this lane) still
   defines `/api/projects/{pid}/rags/{rid}/ingest|query|stats` backed by the
   legacy `RAGManager`/Chroma store, parallel to `webui/routes/rag.py`'s
   `/api/projects/{pid}/rag/*` (the real, frontend-used, PortableRAG-backed
   API). No frontend template calls the `/rags/{rid}/*` endpoints. This is
   the actual duplicated-functionality surface the task asked us to resolve,
   but the fix (migrate or delete these three handlers) requires editing
   `projects.py`, which is outside this lane's file list.
2. **`webui/routes/comparison.py:79`** (outside this lane) imports
   `finetune_studio.rag.store.VectorStore` against a single global
   `settings.rag_store_path` (not per-project). No `comparison.html`
   template exists and nothing in the frontend references `/api/compare` —
   this entire router (not just its RAG usage) looks orphaned and is a
   candidate for removal, but that decision and the route file are outside
   this lane's scope.
3. **`webui/routes/chat_v2.py:18-44`** (outside this lane) has a legitimate,
   deliberate fallback to the legacy `VectorStore` for any `project_rags`
   store that isn't in PortableRAG format (`_is_portable_corpus()` checks
   for `manifest.json` + `vectors.npy`). This one looks like intentional
   back-compat, not a bug — flagging only so the parent is aware it's
   another live dependency on `finetune_studio.rag/` if a deletion is ever
   planned.
