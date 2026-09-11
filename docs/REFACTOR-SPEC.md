# App Refactor Spec — v1

> Captures Genor's mental model from the 09:43 + 09:31 sessions.
> Source of truth for the route consolidation + file-library + chat-harness refactor.

## 1. Mental model (one sentence)

One global chat surface; three project-scoped workflows that each produce a **named, versioned, lineage-tracked artifact** (Dataset for training, Corpus for RAG, Run for trained-model); every artifact's metadata is queryable in the app DB.

---

## 2. Pages — final layout

### Global (no project scope)

| Route | Purpose |
|---|---|
| `/` | Project list + dashboard |
| `/inference` | **The single chat surface** for the app. Load / unload / configure one local model. Stream chat. No project context. No tools. |
| `/hf-models` | HF model browser + downloader |
| `/settings` | App settings, debug, version info |
| `/projects/{pid}` | Project home (read-only summary of RAGs + Datasets + Runs + Benchmarks) |

### Project-scoped (the real work)

| Route | Produces | Tool registry |
|---|---|---|
| `/projects/{pid}/data-prep` | **Dataset** (JSONL + manifest) | `data_prep` |
| `/projects/{pid}/rag` | **Corpus** (chunk index + manifest) | `rag` |
| `/projects/{pid}/training` | **Run** (adapter / merged / GGUF) | `training` (no chat, config only) |
| `/projects/{pid}/benchmarks` | **BenchmarkReport** (suite scores + dataset eval + RAG eval) | `benchmark` |

### Pages deleted in this refactor

- `/projects/{pid}/chat` — chat now lives only on `/inference`
- `/projects/{pid}/agentic` — agentic chat lives **inside** Data Prep and RAG pages, not standalone
- `/projects/{pid}/testing` — folded into Benchmarks
- `/projects/{pid}/models` — folded into global HF Models browser (or stripped if redundant)
- `/projects/{pid}/data` — file library is now on Data Prep page itself

---

## 3. File library — the foundation

**Both Data Prep and RAG share the same file library.** Files are project-scoped, organized into user-named folders.

### 3.1 Storage layout on disk

```
{FTS_ROOT}/projects/{pid}/files/
├── raw/                    # original uploads, never modified after upload
│   ├── 01HQ..._contract_v1.pdf
│   └── 01HQ..._handbook_v1.docx
├── converted/              # OCR / parse output, versioned
│   ├── 01HQ..._contract_v1.md        # paired with raw by id
│   └── 01HQ..._handbook_v1.md
└── folders.json            # folder tree (id, name, parent_id, file_ids[])
```

**Naming rule:** `{file_id}_{original_stem}.{ext}` — id is content-derived so duplicate uploads with the same name get a different id; original filename preserved verbatim (plus `_1`, `_2` … suffix in UI when duplicates exist).

### 3.2 DB schema

```sql
CREATE TABLE file_folders (
  id          TEXT PRIMARY KEY,
  project_id  TEXT NOT NULL,
  name        TEXT NOT NULL,
  parent_id   TEXT,                     -- nullable for root folders
  created_at  REAL NOT NULL,
  UNIQUE(project_id, parent_id, name)
);

CREATE TABLE project_files (
  id              TEXT PRIMARY KEY,     -- content-derived hash prefix
  project_id      TEXT NOT NULL,
  original_name   TEXT NOT NULL,
  current_version INTEGER NOT NULL DEFAULT 1,
  size_bytes      INTEGER NOT NULL,
  mime_type       TEXT,
  uploaded_at     REAL NOT NULL,
  last_trained_at REAL,                 -- last version this file was INCLUDED IN a Dataset that got trained
  UNIQUE(project_id, original_name)
);

CREATE TABLE file_versions (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  file_id     TEXT NOT NULL REFERENCES project_files(id),
  version     INTEGER NOT NULL,
  raw_path    TEXT NOT NULL,
  raw_hash    TEXT NOT NULL,            -- sha256 of bytes
  raw_size    INTEGER NOT NULL,
  uploaded_at REAL NOT NULL,
  uploaded_by TEXT,                     -- "user" | "agent:<session_id>"
  UNIQUE(file_id, version)
);

CREATE TABLE file_conversions (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  file_id         TEXT NOT NULL REFERENCES project_files(id),
  version         INTEGER NOT NULL,
  format          TEXT NOT NULL,        -- "md" | "txt" | "structured_json"
  converted_path  TEXT NOT NULL,
  converted_hash  TEXT NOT NULL,
  converted_size  INTEGER NOT NULL,
  converter       TEXT NOT NULL,        -- "docling" | "pypdf" | "tesseract" | ...
  converted_at    REAL NOT NULL,
  status          TEXT NOT NULL,        -- "ok" | "error"
  error_message   TEXT,
  FOREIGN KEY(file_id, version) REFERENCES file_versions(file_id, version),
  UNIQUE(file_id, version, format)
);

CREATE TABLE folder_membership (
  folder_id TEXT NOT NULL REFERENCES file_folders(id),
  file_id   TEXT NOT NULL REFERENCES project_files(id),
  PRIMARY KEY (folder_id, file_id)
);
```

### 3.3 Operations on the library

- **Upload** → writes raw to `files/raw/{id}_{stem}.{ext}`, creates `project_files` row + `file_versions` row, kicks off background conversion → `file_conversions` row. UI shows the file in the library with a "converting…" badge until `file_conversions.status = ok`.
- **Convert** → re-runs the conversion pipeline, writes a new `file_conversions` row. Always versioned.
- **Edit** (raw or converted) → creates a NEW `file_versions` row (file_versions is the source of truth; `project_files.current_version` is a pointer). Old versions remain on disk for audit until explicitly GC'd.
- **Delete** → soft-delete (`project_files.deleted_at`), or hard-delete (default off, requires confirmation in UI). Cascades to versions, conversions, and folder_membership.
- **Move** between folders → updates `folder_membership`. History is preserved.
- **Rename folder** → updates `file_folders.name`.
- **Detect drift**: when user opens "Pick files for next dataset", the UI shows a ⚠️ on any file whose `current_version > version_used_in_last_dataset`.

### 3.4 UI for the library

- Left column: folder tree (`<ul>` with expand/collapse), context menu for create/rename/delete folder, drag-and-drop file move.
- Right column: file list (table with name, version, size, converted-status, last-used, drift-flag, checkbox). Multi-select for bulk operations.
- Top bar: upload button, search, filter by "raw only / converted only / has errors".
- File detail view: tabs = Preview (extracted text or original PDF render), Edit (in-browser text editor for `.md`/`.txt`, download for raw), History (version timeline with diff-view), Metadata (DB row).

---

## 4. Datasets — what Data Prep produces

### 4.1 DB schema

```sql
CREATE TABLE datasets (
  id              TEXT PRIMARY KEY,
  project_id      TEXT NOT NULL,
  name            TEXT NOT NULL,
  format          TEXT NOT NULL,        -- "sharegpt" | "alpaca" | "openai_chat"
  jsonl_path      TEXT NOT NULL,        -- absolute path on disk
  manifest_path   TEXT NOT NULL,        -- absolute path on disk
  qa_pair_count   INTEGER NOT NULL,
  created_at      REAL NOT NULL,
  created_by      TEXT NOT NULL,        -- "user" | "agent:<session_id>"
  lint_status     TEXT NOT NULL,        -- "ok" | "warnings" | "errors"
  schema_version  TEXT NOT NULL DEFAULT "1",
  UNIQUE(project_id, name)
);

CREATE TABLE dataset_source_files (
  dataset_id TEXT NOT NULL REFERENCES datasets(id),
  file_id    TEXT NOT NULL REFERENCES project_files(id),
  version    INTEGER NOT NULL,          -- which version of the file this dataset was built from
  PRIMARY KEY (dataset_id, file_id)
);

CREATE TABLE dataset_includes (
  dataset_id   TEXT NOT NULL REFERENCES datasets(id),
  included_id  TEXT NOT NULL,           -- dataset_id being included as base
  PRIMARY KEY (dataset_id, included_id)
);

CREATE TABLE qa_pairs (
  id              TEXT PRIMARY KEY,
  project_id      TEXT NOT NULL,
  dataset_id      TEXT REFERENCES datasets(id),  -- nullable while still "pending" before save
  source_file_id  TEXT REFERENCES project_files(id),
  source_version  INTEGER,
  source_chunk_id TEXT,                  -- for traceability: which chunk the pair was mined from
  question        TEXT NOT NULL,
  answer          TEXT NOT NULL,
  status          TEXT NOT NULL DEFAULT "pending",  -- "pending" | "approved" | "rejected"
  importance      INTEGER NOT NULL DEFAULT 5,      -- 1-10 weighting
  tags            TEXT,                   -- JSON array
  created_at      REAL NOT NULL,
  created_by      TEXT NOT NULL,
  dedup_group_id  TEXT                    -- for clustering near-duplicates
);
```

### 4.2 On-disk artifacts

```
{FTS_ROOT}/projects/{pid}/datasets/{dataset_id}/
├── data.jsonl              # the actual training file
└── manifest.json
```

`manifest.json` shape:

```json
{
  "schema_version": "1",
  "dataset_id": "ds_01HQ...",
  "name": "Q4-2026-contracts-v1",
  "format": "sharegpt",
  "created_at": "2026-09-11T09:00:00Z",
  "created_by": "agent:session_abc123",
  "source_files": [
    {
      "file_id": "...",
      "original_name": "Q3-contracts.pdf",
      "version": 3,
      "raw_hash": "sha256:...",
      "converted_hash": "sha256:...",
      "uploaded_at": "...",
      "converted_at": "...",
      "converter": "docling"
    }
  ],
  "qa_pair_count": 1234,
  "schema_lint": {
    "errors": 0,
    "warnings": ["5 pairs had answers > 2000 chars"],
    "details": [...]
  },
  "stats": {
    "avg_question_length": 142,
    "avg_answer_length": 320,
    "unique_question_tokens": 4521,
    "dedup_ratio": 0.03,
    "importance_distribution": {"1-3": 12, "4-7": 980, "8-10": 242}
  },
  "includes": ["ds_01HQ..."],            // base datasets this built on top of
  "training_history": [
    {"run_id": "run_01HQ...", "started_at": "...", "ended_at": "...", "loss_final": 0.34}
  ]
}
```

### 4.3 Tool suite — Data Prep chat

System prompt: *"You are a senior data prepper. Your job is to read the user's company files, mine them for Q&A pairs that teach a model to answer questions on that material, and assemble a clean JSONL training file. Be conservative with hallucination — every answer must be traceable to a source. Be opinionated about quality — cut junk, dedupe, weight by importance."*

Tools:

| Tool | Args | Effect |
|---|---|---|
| `list_files` | `{folder_id?, status?, search?}` | List files in library |
| `read_file` | `{file_id, version?, format?}` | Return file content (truncated, paginated) |
| `search_files` | `{query, file_ids?, top_k}` | Semantic search across file content |
| `get_file_metadata` | `{file_id}` | Return full DB row + version history |
| `list_qa_pairs` | `{file_id?, status?, dedup_group_id?, limit?}` | Already exists, expand filters |
| `create_qa_pairs` | `{file_id, pairs[]}` | Already exists, add `importance` and `tags` per pair |
| `update_qa_pair` | `{id, fields{}}` | Edit any field of an existing pair |
| `delete_qa_pair` | `{id}` | Remove from set |
| `bulk_importance` | `{ids[], weight}` | Set importance on multiple |
| `deduplicate` | `{file_id?, similarity_threshold?}` | Run embedding-based dedup, return clusters + suggested merges |
| `lint_dataset` | `{dataset_id?}` | Validate JSONL schema, return errors/warnings |
| `compose_dataset` | `{name, format, file_ids[], include_dataset_ids?, min_importance?}` | Build the JSONL + manifest, register in DB |
| `tag_pairs` | `{ids[], tag, mode: 'add'|'remove'|'replace'}` | Tag management |

---

## 5. RAG — same shape, different artifacts

### 5.1 Storage layout on disk

```
{FTS_ROOT}/projects/{pid}/rags/{corpus_id}/
├── index.faiss               # dense index
├── bm25.pkl                  # sparse index
├── chunks.jsonl              # canonical chunk list
├── manifest.json
└── embedder/                 # bundled embedder if self-contained
```

### 5.2 DB schema

```sql
CREATE TABLE rags (                                -- "rag" name kept for compat; concept = corpus
  id              TEXT PRIMARY KEY,
  project_id      TEXT NOT NULL,
  name            TEXT NOT NULL,
  chunk_size      INTEGER NOT NULL,
  chunk_overlap   INTEGER NOT NULL,
  embedder_id     TEXT NOT NULL,        -- references shared_models or path
  reranker_id     TEXT,
  hybrid          INTEGER NOT NULL DEFAULT 1,
  created_at      REAL NOT NULL,
  created_version TEXT NOT NULL,         -- monotonic, bumped on every rebuild
  index_path      TEXT NOT NULL,
  manifest_path   TEXT NOT NULL,
  chunk_count     INTEGER NOT NULL DEFAULT 0,
  doc_count       INTEGER NOT NULL DEFAULT 0,
  UNIQUE(project_id, name)
);

CREATE TABLE rag_source_files (
  rag_id   TEXT NOT NULL REFERENCES rags(id),
  file_id  TEXT NOT NULL REFERENCES project_files(id),
  version  INTEGER NOT NULL,
  PRIMARY KEY (rag_id, file_id)
);

CREATE TABLE rag_chunks (
  id              TEXT PRIMARY KEY,
  rag_id          TEXT NOT NULL REFERENCES rags(id),
  file_id         TEXT NOT NULL REFERENCES project_files(id),
  file_version    INTEGER NOT NULL,
  chunk_index     INTEGER NOT NULL,
  text            TEXT NOT NULL,
  token_count     INTEGER,
  quality_score   INTEGER,              -- agent-set 1-10
  notes           TEXT
);
```

### 5.3 Tool suite — RAG chat

System prompt: *"You are a senior RAG engineer. Your job is to judge retrieval quality, find missing or broken chunks, flag ambiguous content, suggest re-chunking strategies, and write evaluation Q&A pairs for this corpus."*

Tools:

| Tool | Args | Effect |
|---|---|---|
| `list_rags` | `{}` | List project RAGs |
| `list_files` | (same as data-prep) | |
| `read_file` | (same as data-prep) | |
| `search_corpus` | `{rag_id, query, top_k}` | Run actual retrieval, return ranked chunks with scores |
| `inspect_chunk` | `{rag_id, chunk_id, neighbors}` | Read chunk + surrounding chunks for context |
| `mark_chunk_quality` | `{chunk_id, score, note?}` | Set quality_score |
| `add_chunk_note` | `{chunk_id, note}` | Annotate |
| `find_gaps` | `{rag_id, eval_qa_pairs[]}` | Run retrieval, identify queries with no good match |
| `eval_retrieval` | `{rag_id, qa_pairs[], metrics[]}` | Compute recall@k, MRR, nDCG |
| `compose_rag_v` | `{name, file_ids[], chunk_size, chunk_overlap, embedder_id, reranker_id?, hybrid}` | Rebuild corpus, register in DB |

---

## 6. Training — consumes Dataset, produces Run

No chat on this page. Pure config + progress.

- **Pick dataset** from dropdown (lists `datasets` table for this project)
- **Pick base model** from dropdown (any registered local GGUF / safetensors)
- **Config**: LoRA rank, LR, epochs, batch, seq len, merge-on-save, system prompt override
- **Run** → writes `runs` row + adapter dir + merged dir + GGUF export
- **Live log** via SSE or polling
- **Manifest** per run: dataset used (with manifest hash), base model, hyperparams, loss curve, start/end time

Already exists; minor changes for dataset linkage + manifest hash.

---

## 7. Benchmark — tests for any model

### 7.1 DB schema

```sql
CREATE TABLE benchmark_reports (
  id              TEXT PRIMARY KEY,
  project_id      TEXT NOT NULL,
  run_id          TEXT REFERENCES runs(id),        -- nullable for base-model-only tests
  model_path      TEXT NOT NULL,
  suite           TEXT NOT NULL,                   -- "mmlu" | "perplexity" | "dataset_eval" | "rag_eval"
  judge_kind      TEXT,                            -- "self" | "external:<provider_id>"
  judge_provider  TEXT,
  num_samples     INTEGER NOT NULL,
  overall_score   REAL,
  per_metric_json TEXT NOT NULL,
  per_sample_json TEXT,                            -- bounded; truncate if huge
  started_at      REAL NOT NULL,
  ended_at        REAL NOT NULL,
  manifest_path   TEXT NOT NULL
);
```

### 7.2 Three test modes

1. **Generic capability** — perplexity, MMLU-style, HellaSwag etc. (`real_benchmarks.py` exists, expand.)
2. **Dataset eval** — sample N Q&A pairs from a dataset the run was trained on, ask the model, judge with self OR external LLM.
3. **RAG eval** — sample N Q&A pairs from a RAG corpus, run retrieval, ask model with retrieved context, judge retrieval + answer quality.

### 7.3 LLM Judge

- **Self-judge**: same loaded model, given reference answer, asked to score the prediction 0-5 on accuracy + relevance.
- **External judge**: configured OpenAI-compat endpoint (`/api/providers` row), runs same prompt, returns structured score.

### 7.4 Tool suite — Benchmark chat

Optional chat on the Benchmark page that helps analyze results: *"You scored 0.62 on dataset_eval. Here are 5 examples where you got it wrong — what patterns do you see?"* Tools: `get_benchmark_report`, `list_run_benchmarks`, `compare_runs`, `inspect_sample`.

---

## 8. Chat harness — backend-owned

### 8.1 Architecture

```
Frontend (any page that wants chat)
    ↓
Generic chat widget JS (~150 lines)
    ↓ POST {messages, mode, page_context}
Backend route /api/projects/{pid}/chat
    ↓
ChatHarness (mode-aware):
  - system prompt registry (mode → prompt + tool list)
  - tool registry (mode → list of {name, schema, handler})
  - tool-calling loop (already exists in data_prep_chat.py; generalize)
  - history persistence (chat_sessions table)
    ↓
ModelManager (single source of model truth — see §9)
```

### 8.2 Generic chat widget

Same widget on every page that has a chat. Frontend posts:

```json
{
  "mode": "data_prep" | "rag" | "benchmark" | "plain",
  "messages": [...],
  "project_id": "...",
  "context": {
    "selected_file_ids": ["..."],
    "dataset_id": "...",
    "rag_id": "...",
    ...
  }
}
```

Backend injects the right system prompt + tools + context slice. Returns `{reply, tool_calls[], rounds, session_id}` with SSE streaming later.

### 8.3 Tool registry pattern

```python
@tool_registry.register(mode="data_prep")
def list_files(project_id: str, folder_id: str | None = None) -> dict:
    """List files in the project's library. Use this before anything else."""
    ...

@tool_registry.register(mode="data_prep")
def read_file(project_id: str, file_id: str) -> dict:
    """Read a file's converted text. Use after list_files."""
    ...
```

Mode binding is explicit. A tool can be registered for multiple modes:

```python
@tool_registry.register(mode=["data_prep", "rag"])
def list_files(...):
    ...
```

### 8.4 Chat sessions (history persistence)

```sql
CREATE TABLE chat_sessions (
  id          TEXT PRIMARY KEY,
  project_id  TEXT NOT NULL,
  mode        TEXT NOT NULL,           -- "data_prep" | "rag" | "benchmark" | "plain"
  title       TEXT,
  messages    TEXT NOT NULL,           -- JSON array of OpenAI-style messages
  created_at  REAL NOT NULL,
  updated_at  REAL NOT NULL
);
```

UI: chat history sidebar per mode, search across sessions, fork a session at any point.

---

## 9. Model loading — single source of truth

### 9.1 Unified ModelManager

The current setup has **two parallel systems**:
- Global inference engine (loaded via `/api/chat-v2/load`)
- ModelManager (loaded via `/api/providers/{pid}/load`)

These should be **one**. Pick the ModelManager as the source of truth, deprecate the global engine. Move the loader config UI from `/inference` to the ModelManager's `load()` parameters.

### 9.2 `/inference` page

- One model slot, global to the app.
- Load button → modal with all loader params (n_ctx, n_gpu_layers, n_batch, n_threads, seed, RoPE, flash_attn, mmap, mlock) → POST `/api/providers/{pid}/load` for chosen provider.
- Unload button.
- Live status: `idle` / `loading` (with progress % if available) / `loaded` / `error`.
- Streaming chat (SSE) to the loaded model — no project, no tools, no RAG.

### 9.3 Status pill (app-wide)

Persistent header on every page. Polls `/api/providers` every 5s. Shows:
- `○ No model loaded` (red) → click opens `/inference`
- `◐ Loading Qwen3-8B (47%)` (amber)
- `● Qwen3.8-27B-abliterated-Q4_K_M` (green) → click opens `/inference`

Already half-built. Needs to be elevated to every page template.

---

## 10. Implementation stages

Each stage is independently shippable. Stages build on each other.

### Stage 1 — File library foundation (the bedrock)
- DB migrations for `file_folders`, `project_files`, `file_versions`, `file_conversions`, `folder_membership`
- Disk layout: `files/raw/{pdfs,imgs,csvs,docs,code,other}/{id}_{stem}.{ext}` + `files/converted/{user_folder}/{stem} (converted {YYYY-MM-DD}).md` + trash dirs
- Routes: `POST /api/projects/{pid}/files/upload` (with bulk-upload + dedup report), `GET /api/projects/{pid}/files`, `GET /api/projects/{pid}/files/{fid}`, `PATCH /api/projects/{pid}/files/{fid}` (metadata only; raw is immutable), `DELETE /api/projects/{pid}/files/{fid}` (soft to trash), `POST /api/projects/{pid}/files/{fid}/restore`, `POST /api/projects/{pid}/files/{fid}/convert`, `GET /api/projects/{pid}/files/{fid}/versions`, `GET /api/projects/{pid}/files/{fid}/raw?version=N` (re-download)
- Routes: `GET/POST/PATCH/DELETE /api/projects/{pid}/folders`
- Background conversion pipeline (reuse existing DataPrepRunner)
- UI: file library on Data Prep page with folders, MIME-routed RAW structure, upload, viewer/editor (raw preview when supported, else converted), re-download button, dedup notification
- **No refactor of routes yet** — old pages still work

### Stage 2 — Datasets
- DB migrations for `datasets`, `dataset_source_files`, `dataset_includes`, `qa_pairs`
- Disk layout: `datasets/{id}/data.jsonl` + `manifest.json`
- Routes for dataset CRUD + lint + compose
- UI: dataset registry on Data Prep page

### Stage 3 — RAG refactor
- DB migrations for `rag_chunks`, `rag_source_files`
- Update `rags` table with manifest_path, current_version
- Disk layout per §5.1
- UI: build/rebuild on RAG page uses file picker

### Stage 4 — Chat harness
- DB migrations for `chat_sessions`
- Generalize `data_prep_chat.py` → `chat_harness.py` with mode-aware tool registry
- Generic chat widget JS (drop into any page)
- Move chat from `/projects/{pid}/chat` into Data Prep + RAG pages
- Delete `/projects/{pid}/chat` and `/projects/{pid}/agentic` routes

### Stage 5 — Expanded tool suites
- Data Prep: update_qa_pair, delete_qa_pair, bulk_importance, deduplicate, lint_dataset, compose_dataset, tag_pairs
- RAG: search_corpus, inspect_chunk, mark_chunk_quality, add_chunk_note, find_gaps, eval_retrieval, compose_rag_v
- All registered via the `@tool_registry.register` decorator

### Stage 6 — Training link
- Update `runs` to reference `dataset_id` + manifest hash
- On Run start, write `runs/{run_id}/manifest.json` with dataset lineage
- Drift detection on file picker for next run

### Stage 7 — Benchmark expansion
- DB migrations for `benchmark_reports`
- Dataset eval mode + RAG eval mode
- LLM judge (self + external)
- Compare runs table

### Stage 8 — Model manager unification
- Deprecate global inference engine
- Move loader config UI to `/inference`
- Promote status pill to every page

### Stage 9 — Polish + verify
- End-to-end test with Qwen3-8B on fan-dragon
- Cleanup, performance pass, docs

---

## 11. What I'd ship first

Recommend Stage 1 (file library + folders + dual storage + provenance) as the next PR. Everything else builds on it.

Stages 2-3 can run in parallel as separate PRs after Stage 1 lands.

Stage 4 (chat harness) is the chunkiest and most disruptive — schedule it when Genor has bandwidth for a longer test cycle.

---

## 12. Decisions (locked)

1. **Folder nesting** — flat at the user-organisation level. The browser itself shows user-named folders ("Contracts", "Handbook", "Eval Sets") with drag-to-move between them.

2. **Auto-separation on upload by MIME type** — at upload time, files are routed into type-segregated RAW folders:
   - `files/raw/pdfs/`, `files/raw/imgs/`, `files/raw/csvs/`, `files/raw/docs/`, `files/raw/code/`, `files/raw/other/`
   - Files in `raw/` are **immutable** — cannot be moved, only deleted (soft to trash). This guarantees the audit trail: a raw file at hash `abc` is the same bytes forever.
   - Converted files live in user-named folders under `files/converted/{user_folder}/` with names like `{original_stem} (converted {YYYY-MM-DD}).md` so a CLI user can `ls` and immediately trace back to the source doc.

3. **Soft delete with 7-day purge**:
   - Files move to `files/raw/.RAW_TRASH/` or `files/converted/.CONVERTED_TRASH/`
   - DB row gets `deleted_at` timestamp
   - A scheduled task (cron or systemd timer, configurable) purges anything older than 7 days
   - Manual cleanup: a `/api/projects/{pid}/files/trash/purge?older_than_days=N` endpoint + an `fts files trash` CLI subcommand for ad-hoc runs
   - Until purged, file is restorable via UI ("Trash" view with Restore button)

4. **File dedup** — sha256 hash of raw bytes:
   - On upload, compute hash. If `project_files.raw_hash` already exists for this project, **notify** the user with the duplicate's filename + path, and **skip** (do not write a new version).
   - On bulk upload (e.g. 100 files), defer the report until all files are processed. Final report shape:
     ```json
     {
       "uploaded": 87,
       "converted_ok": 84,
       "converted_failed": 3,
       "duplicates_skipped": 10,
       "duplicates": [
         {"filename": "...", "duplicate_of": "Q3-contracts.pdf", "raw_hash": "..."}
       ],
       "errors": [{"filename": "...", "stage": "upload|convert", "message": "..."}]
     }
     ```

5. **JSONL format** (clarified with Genor): the **training data file** fed to the trainer. Each line is one Q&A pair. The exact JSON shape per line depends on the format:
   - **ShareGPT** (default): `{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}`
   - **Alpaca**: `{"instruction": "...", "input": "...", "output": "..."}`
   - **OpenAI chat**: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
   
   Each saved dataset produces a `.jsonl` file with thousands of these lines + a sibling `manifest.json` with full provenance. The trainer reads this file directly.

6. **External LLM judge** — anything that speaks `/v1/chat/completions` OpenAI-compat. Reuse the existing `/api/providers` row system: user adds a provider with `kind: openai_compat`, gets `base_url` + `api_key` + `model_id`, the benchmark route uses it for judging. Anthropic-format not needed for v1.

7. **Self-eval on tiny models** — allowed regardless of model size. User's responsibility. Note in UI: *"Judging with the same model is biased toward the model's own style. For trustworthy eval use an external judge."*

---

## 13. Implementation stages

Each stage is independently shippable. Stages build on each other.
