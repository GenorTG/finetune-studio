# Finetune Studio — Architecture

> Self-hosted fine-tuning studio for local LLMs: file library → RAG prep → agentic
> Q&A mining → LoRA/QLoRA training → GGUF export → benchmarks → single-model
> inference chat. FastAPI + Jinja2 SSR with a terminal (tmux/neovim) aesthetic.
> ~21k LOC Python, one SQLite DB, zero cloud dependencies.

## 1. Entry points

| Entry | Command | What it does |
|---|---|---|
| CLI | `fts …` / `finetune-studio …` (`pyproject [project.scripts]` → `finetune_studio.cli:main`) | 19 subcommands: `webui`, `train`, `convert`, `benchmark`, `rag`, `files`, `analyze`, `augment`, `compare`, `models`, `optimize`, `test`, `suite`, `validate`, `validate_hallucination`, `vram`, `analyze`… (`cli/commands/*.py`, registered in `cli/_registry.py`) |
| Web | `fts webui` → `uvicorn finetune_studio.webui.app:app --host 0.0.0.0 --port 7860` | The studio itself (systemd user service on fan-dragon) |

## 2. Layer map

```
src/finetune_studio/
├── config.py            Settings dataclass + env overrides (single source of defaults)
├── db/                  SQLite (stdlib sqlite3, WAL)
│   ├── connection.py    SCHEMA (16 tables), migrations via CREATE IF NOT EXISTS
│   ├── system_updates.py  update-pipeline rows (queued|running|done|error|cancelled)
│   ├── datasets.py      project_datasets + HF-dataset counting
│   └── reviews.py       data_review rows
├── data/
│   ├── fs/              Stage 1A file library: paths.py (FTS_ROOT layout),
│   │                    file_library.py (742 LOC — upload/dedup/folders/trash/versions)
│   ├── parsers/ + parsers.py   MIME → text (pdf/docx/xlsx/csv/xml/images→OCR)
│   ├── ocr.py           image OCR pass
│   ├── prep/            prep jobs: chunk→embed→LLM Q&A drafts (runner.py, export.py)
│   ├── rag_portable/    portable RAG store: embedders, bm25, rerankers, query, store
│   ├── project_filesystem.py  facade re-exporting data.fs.* (stable import surface)
│   └── shared_models.py, converter.py, organizer.py, validator.py, rag_eval.py
├── models/
│   ├── registry.py      GGUF/safetensors discovery (scan_models)
│   ├── manager.py       ModelManager (legacy — slated for deletion in Stage 8)
│   ├── providers.py     provider rows (local_gguf / external_api) + OpenAI-compat calls
│   └── loader.py        model info (arch, size, quant) for UI
├── training/
│   ├── engine.py        TrainingEngine: TRL SFTTrainer, LoRA/QLoRA/full, on_update hook
│   ├── data.py          JSONL → datasets.Dataset
│   ├── data_quality.py, data_augmentation.py, config_optimizer.py
│   ├── hallucination_guard.py, knowledge_preservation.py
│   ├── vram/profile.py  VRAM estimation before launch
│   └── monitor.py       live loss/step tracking → db.update_run
├── testing/
│   ├── inference.py     ★ InferenceEngine — THE single global model instance.
│   │                    GGUF (llama-cpp-python) or HF (transformers). Idle auto-unload
│   │                    (FTS_IDLE_TIMEOUT, default 1800s). Chat-template aware.
│   └── suite.py         prompt/response test suites
├── benchmarks/          real_benchmarks (public sets), tool_calling, scoring, samplers
├── templates/           chat-template renderer (GGUF metadata / Jinja2) — Stage 8 core
├── compare/engine.py    A-vs-B model comparison (requests → external APIs)
└── webui/
    ├── app.py           ★ composition root: singletons + router mounts (see §3)
    ├── routes/          one module per domain (see §3)
    ├── templates/       Jinja2 pages (base.html = session bar + global pill)
    └── static/          app.css (terminal theme, breakpoints 480/700/900/1100/1280)
                         js/: spa, activity, palette, tutorial, settings, sprites, thinking
```

★ = load-bearing files. Everything else composes around them.

## 3. HTTP surface (app.py mounts — the URL truth table)

| Prefix | Router module | Serves |
|---|---|---|
| `/` (HTML) | `pages.py` | dashboard, projects, project tabs, `/inference`, `/settings`, `/models/explore` (14 page routes) |
| `/api/models` | `routes/models.py` | model list/info/load/unload + **VRAM diagnostics** (`_gpu_snapshot`, `_vram_hint`) |
| `/api/inference` | `models.inference_router` | **status / load / unload / chat / memory-estimate** — the global engine |
| `/api/chat-v2` | `chat_v2.py` | Test-mode project chat (RAG-augmented) |
| `/api` (self-prefixed) | `data_prep.py` | `/api/projects/{pid}/data-prep/*` sources/qa/runs/export + HTML page |
| `/api` (self-prefixed) | `data_prep_chat.py` | `/api/projects/{pid}/data-prep/chat` — **agent loop**: list_sources → read_source → create_qa_pairs |
| `/api` | `file_library.py` | `/api/projects/{pid}/files/*` + `/folders/*` (Stage 1A) |
| `/api` | `exports.py`, `datasets.py`, `hf_models.py`, `updates.py` | run exports, dataset registry, HF downloads, **system update pipeline** |
| `/api/training` | `training.py` | run start/status/events (SSE) |
| `/api/testing`, `/api/compare`, `/api/benchmarks` | `testing.py`, `comparison.py`, `benchmarks.py` | suites, A/B, benchmarks |
| `/api/projects` | `projects.py`, `rag.py` | CRUD + RAG corpora |
| `/api/agentic` | `agentic.py` | tool-using agent (legacy surface, Stage 5 absorbs) |
| `/api/system` | `system.py` | resources (RAM/VRAM), update endpoints via updates |
| `/api/activity` | `activity.py` | live task feed (drawer polls every 2s) |
| `/api/data*` | `data.py`, `data_editor.py`, `quality.py` | uploads, editor, quality scoring |

**Route-ordering rule:** specific paths (`/files/trash`) must register BEFORE
catch-alls (`/files/{fid}`) — FastAPI matches in declaration order.

## 4. Runtime state (app.py)

```python
training_engine   = TrainingEngine()      # one active run; on_update → db persistence
inference_engine  = InferenceEngine()     # THE single loaded model (Stage 8 rule)
discovered_models = []                    # rescanned at lifespan startup
```

- **Single-model rule:** exactly one model in VRAM at a time. Tabs differ only by
  which tools/system-prompt they hand it (Stage 8 unifies this fully — see
  `REFACTOR-SPEC.md`). `ModelManager` is legacy; chat falls back to matching
  `inference_engine.model_path` against provider rows.
- **Lazy imports:** heavy ML libs (torch/transformers/trl/peft/llama_cpp) are
  imported *inside functions* — `uvicorn` boots in ~2s despite the ML stack.
  Don't "clean up" these into top-level imports.

## 5. Data flow (project lifecycle)

```
upload ──► files/raw/<mime-family>/<sha256>   (immutable, deduped, auto-MIME folder)
       ──► parse ──► qa/sources/<id>.json     (chunked ~500 tok)
       ──► Q&A drafts:  prep job (batch, LLM)  OR  agent chat (tool-calling)
       ──► qa/pairs/<id>.json                  (status: pending|approved|rejected)
       ──► export JSONL ──► project_datasets   (training-ready)
       ──► training (LoRA/QLoRA/full) ──► output/<run>/adapter + merged
       ──► GGUF export (llama.cpp CLI) ──► *.gguf
       ──► benchmarks (base vs trained) ──► benchmark_runs
       ──► inference validation (needle tests, chat, tool-calling)
```

Soft delete everywhere: `files/.RAW_TRASH/`, `.CONVERTED_TRASH/`, 7-day purge
(`fts files purge-trash`). Versions in `file_versions`, conversions in
`file_conversions` — see `REFACTOR-SPEC.md` for the 7 locked decisions.

## 6. Persistence

- **SQLite** at `Settings.db_path` (default `data/finetune_studio.db`), WAL mode,
  schema applied idempotently in `db/connection.py` (16 tables: core runs/projects
  + Stage 1A file library set + `system_updates`).
- **Disk is truth for content** (files, JSONL, GGUFs, qa/pairs); the DB stores
  metadata/indexes only. `FTS_ROOT` env (default `~/.finetune-studio`) roots all
  project dirs: `projects/<pid>/{files,qa,datasets,exports}` (`data/fs/paths.py`).

## 7. Frontend conventions

- `base.html`: session bar (horizontal tmux-style tab strip — **never** a vertical
  rail; mobile keeps it horizontal + scrollable), global model pill (polls
  `/api/providers` **and** `/api/inference/status` every 3s), activity drawer,
  command palette (Ctrl-K), SPA link interception (`data-link`).
- Cache-bust: static assets carry `?v=N` in base.html — bump N on every css/js change.
- Terminal aesthetic is intentional: `▐` section markers, `[ BUTTON ]` labels,
  `┌── NO_DATA ──┐` boxes. Not debug artifacts.

## 8. Config & env

| Knob | Where | Default |
|---|---|---|
| host/port | `config.Settings` | 0.0.0.0:7860 |
| LoRA rank / epochs / batch / max_seq | `config.Settings` | 64 / 4 / 2 / 2048 |
| `FTS_ROOT` | `data/fs/paths.py` | `~/.finetune-studio` |
| `FTS_IDLE_TIMEOUT` | `testing/inference.py` | 1800 s auto-unload |
| `LLAMA_CPP_DIR` | `install.sh` / exports | project-local `.llama.cpp/` |
| `FTS_UPDATE_SCRIPT` | `routes/updates.py` | repo-root `update.sh` |
| `FTS_SKIP_UPDATE=1` | `routes/updates.py` | test mode — canned log, no subprocess |

## 9. Related docs

- `REFACTOR-SPEC.md` — the 7 locked architectural decisions + Stage 1–9 plan
- `DEPENDENCIES.md` — what each Python dep is for, where it's imported
- `DEPLOYMENT.md` — install, service, **update pipeline** (API + Settings UI)
- `.agent/AGENT-WORKFLOW.md` (gitignored, local) — agentic working playbook
