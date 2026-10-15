# Finetune Studio — design and contributor notes

Local WebUI (FastAPI + Jinja, Python 3.12, uv venv `.venv/`) and CLI `fts` that turn your own files into knowledge once,
then train a model on it (LoRA adapter or merged model) or serve it as a **Smart Brain** (retrieval, often called
RAG) — plus testing, export and chat. This page is the public map; install and deployment are in
[INSTALL.md](INSTALL.md) and [DEPLOYMENT.md](DEPLOYMENT.md).

## The flow

```
upload ─► file library (bytes, versions, folders, trash)
           │ parsed ONCE ─► knowledge.sources: parsed text + passages (+ Vision readings, hand edits)
           ▼
  mining ─► Q&A pairs ─► dataset ─► 2 train: LoRA adapter / merged model ─► 3 test ─► 4 use it
  passages ─► 2 Smart Brain index (semantic + keyword search, rank fusion, reranking) ─► chat with sources,
              encrypted bundle, standalone MCP + HTTP server
```

Every project page shows one flow bar: `1 Files & Knowledge (files · pairs) → 2 train | 2 Smart Brain → 3 test →
4 use it`.

## Layout

- `src/finetune_studio/`
  - `webui/` — app, routes, templates, static (no build step)
  - `cli/` — the `fts` commands
  - `knowledge/` — `sources` (each file's one parse), `index` (Smart Brain index from the passages), `search`
    (facts + passages), `migrate` (one-time storage moves)
  - `smart_brain/` — index store and query, embedders/rerankers, standalone server, encrypted export
  - `data/` — parsers, data prep, file storage, mining
  - `training/`, `models/`, `testing/`, `benchmarks/`, `compare/`, `inference/`, `supervisor/`
  - `db/` — all SQLite access; `accel/` — vendor-neutral GPU layer
- `tests/` — pytest, one file per concern; `e2e_*.py` + `run_qa.sh` are live/browser suites (they can change data:
  they need `FTS_ALLOW_LIVE_E2E=1` and `FTS_BASE`)
- `scripts/` — GPU detect/plan/install (`accel_plan.py`), diagnostics, codemap, dataset tools
- Runtime state (never committed): `data/`, `models/`, `output/`, `.tmp/`, and `$FTS_ROOT` (default `~/.finetune-studio`)

## Rules

- **One parse per file.** A source is a file-library file; its parse is a `file_conversions` row plus
  `files/<sha12>/{parsed.txt,parsed.json,chunks/}`. Never parse a file a second time: edits and Vision readings go
  through `knowledge.sources.set_text`, Reparse is `sources.parse(force=True)`, the Smart Brain index reads the
  passages (`knowledge.index`).
- **Errors are honest:** a real 4xx/5xx with a reason, never a 200 `{error}` or a silent fallback.
- **Device code goes through `finetune_studio.accel`** — never hard-code `cuda`, a device map, `bf16=True` or
  bitsandbytes. The GPU stack choice lives only in `scripts/accel_plan.py`.
- **Install only via `install.sh` / `update.sh`** (the uv venv has no pip module); every resolve passes the torch
  constraints file so torch never drifts to another CUDA build.
- **Data-prep paths** go through `data.fs.paths.resolve_in_project` / `resolve_within`, never a raw user path.
- **Smart Brain export is AES-256-GCM encrypted by default**; the passphrase only travels in POST bodies and is never
  stored or logged; the shipped server binds 127.0.0.1 unless a token is set. Its public names stay stable (env
  `RAG_*`, `rag.config.json`, MCP tools `rag_search` / `rag_info`).
- **Hosts and paths come from config**, never from code: `FTS_ROOT`, `FTS_BASE` (live suites), `FTS_DB`.
- **New code goes in the module that already owns the concern**; a new feature is a new module with its own test file.
  `make codemap` refreshes the symbol map; `make codemap-check` fails on drift.

## Checks

- Tests: `.venv/bin/python -m pytest -q -p no:cacheprovider --ignore=tests/test_vram.py`
- Lint: `.venv/bin/ruff check src/ scripts/` (must stay clean)
- GPU paths are proven on real hardware; CI runs the CPU suite in shards (`scripts/ci_shard.py`).
