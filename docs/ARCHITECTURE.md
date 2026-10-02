# Finetune Studio — Architecture

> Self-hosted fine-tuning studio for local LLMs: file library → RAG prep → agentic
> Q&A mining → LoRA/QLoRA training → GGUF export → benchmarks → single-model
> inference chat. FastAPI + Jinja2 SSR with a terminal (tmux/neovim) aesthetic.
> 40,475 lines across 212 tracked Python modules in `src/finetune_studio/`
> (257 tracked files under `src/`, including UI/assets); 527 tracked files
> across the repository. One SQLite DB. The app is self-hosted; model downloads, OCR
> language-data bootstrap, and optional integrations can contact external
> services.
>
> For module detail, see **§10 Module reference**. Backend Python modules were
> documented in 11 lane docs; frontend assets and operations are now covered
> by `docs/modules/webui-frontend.md`; operational commands live in
> `docs/INSTALL.md`, `docs/DEPLOYMENT.md`, and the repo `AGENTS.md`. This does
> not mean every tracked test/fixture has received a
> line-by-line audit. This file stays the 10,000-foot map; read the owning
> module doc and current implementation before changing behavior.

## 1. Entry points

| Entry | Command | What it does |
|---|---|---|
| CLI | `fts …` / `finetune-studio …` (`pyproject [project.scripts]` → `finetune_studio.cli:main`) | 17 registered subcommands: `models`, `train`, `test`, `suite`, `validate`, `convert`, `webui`, `rag`, `compare`, `benchmark`, `analyze`, `augment`, `optimize`, `validate-hallucination`, `rag-test`, `vram`, `files` (`cli/_registry.py`) |
| Web | `fts webui` → `uvicorn finetune_studio.webui.app:app --host 0.0.0.0 --port 7860` | The studio itself (optional systemd user service) |

## 2. Layer map

```
src/finetune_studio/
├── config.py            Mutable Settings/RAGSettings dataclasses + process singleton
├── db/                  SQLite (stdlib sqlite3, WAL)
│   ├── connection.py    SCHEMA (23 tables), migrations via CREATE IF NOT EXISTS
│   ├── system_updates.py  update-pipeline rows (queued|running|done|error|cancelled)
│   ├── datasets.py      project_datasets + HF-dataset counting
│   └── reviews.py       data_review rows
├── data/
│   ├── fs/              Stage 1A file library: paths.py (FTS_ROOT layout),
│   │                    file_library.py (upload/dedup/folders/trash/versions)
│   ├── parsers/ + parsers.py   MIME → text (pdf/docx/xlsx/csv/xml/images→OCR)
│   ├── ocr.py           image OCR pass
│   ├── prep/            prep jobs: chunk→embed→LLM Q&A drafts (runner.py, export.py)
│   ├── rag_portable/    portable RAG store: embedders, bm25, rerankers, query, store
│   ├── project_filesystem.py  facade re-exporting data.fs.* (stable import surface)
│   └── shared_models.py, converter.py, organizer.py, validator.py, rag_eval.py
├── models/
│   ├── registry.py      GGUF/safetensors discovery (scan_models)
│   ├── manager.py       ModelManager — owns the one shared InferenceEngine
│   │                    via its `.engine` property (see §4; NOT legacy, live)
│   ├── llama_loader.py  ★ the one true `llama_cpp.Llama()` builder — every
│   │                    GGUF load (provider, engine, helper) goes through
│   │                    `load_llama_gguf()`; GH-AAA no-mixed-offload contract
│   ├── providers.py     provider rows (local_gguf / external_api) + OpenAI-compat calls
│   └── loader.py        model info (arch, size, quant) for UI — live, used by
│                        routes/models.py + routes/pages.py
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
├── benchmarks/          real_benchmarks.py (public MMLU/GSM8K/HellaSwag sets,
│                        real + offline), suite_defs.py (WebUI suite discovery),
│                        comparison.py (the live A-vs-B model comparator)
├── templates/           chat-template renderer (GGUF metadata / Jinja2) — the
│                        canonical Jinja2 rendering path, single source of truth
├── compare/             retired 2026-10-01 — was a fully dead, unwired
│                        duplicate of benchmarks/comparison.py's ModelComparator;
│                        deleted, `__init__.py` kept as a pointer docstring
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
| `/api/system` | `system.py` | resources (RAM/VRAM), update endpoints via updates |
| `/api/activity` | `activity.py` | live task feed via SSE (`GET /api/activity/events`); `GET /api/activity` is the one-shot snapshot fallback |
| `/api/data*` | `data.py`, `data_editor.py`, `quality.py` | uploads, editor, quality scoring |

**Route-ordering rule:** specific paths (`/files/trash`) must register BEFORE
catch-alls (`/files/{fid}`) — FastAPI matches in declaration order.

### Network and file-access boundary

The FastAPI composition currently installs activity/CORS/proxy middleware but
does not install authentication middleware or route-level authentication
dependencies. The CLI and Unix launcher default to `0.0.0.0`, so network
isolation or an authenticated reverse proxy must be provided outside this app
when it is reachable by untrusted clients. That deployment layer is not
verified by this source map.

`POST /api/projects/{pid}/data-prep/sources` has a legacy `data_path` form
that reads the supplied absolute path and copies its contents into the
project's content-addressed store; unlike its `file_id` form, it does not
check project-file ownership. The shipped UI uses `file_id`, but a test
explicitly preserves external absolute-path ingestion. Treat this as an
unresolved trust-boundary issue; do not assume project IDs or the session-bar
UI are authorization. See `docs/modules/webui-routes-workflow.md`.

## 4. Runtime state (app.py)

```python
training_engine   = TrainingEngine()          # one active run; on_update → db persistence
inference_engine  = get_manager().engine      # THE single loaded model
discovered_models = []                        # rescanned at lifespan startup
```

- **Single-model rule:** exactly one model in VRAM at a time. `inference_engine`
  IS `ModelManager`'s own `.engine` — the same object, not two coordinated
  trackers (merged 2026-09-30; `models/manager.py` is live, not legacy). A
  model loaded via any path (data-prep helper, Testing tab, chat) is
  immediately visible to every other path that reads `inference_engine`. A
  short-lived *second* `InferenceEngine()` is legitimately constructed by a
  few one-shot benchmark/judge call sites — always only safe when the caller
  frees the persistent engine's VRAM first via
  `models.llama_loader.unload_all_models()` (GH-AAA contract).
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
  schema applied idempotently in `db/connection.py` (23 tables: core runs/projects
  + Stage 1A file library set + `system_updates`).
- **Disk is truth for content** (files, JSONL, GGUFs, qa/pairs); the DB stores
  metadata/indexes only. `FTS_ROOT` env (default `~/.finetune-studio`) roots all
  project dirs: `projects/<pid>/{files,qa,datasets,exports}` (`data/fs/paths.py`).

## 7. Frontend conventions

- `base.html`: session bar (desktop tmux-style tab strip; mobile uses the compact
  overflow menu), global model pill (polls
  `/api/providers` **and** `/api/inference/status` every 3s), activity drawer
  (live via SSE `/api/activity/events`, silent poll fallback only if EventSource
  fails), command palette (Ctrl-K), SPA link interception (`data-link`).
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

## 10. Module reference (2026-10-01 audit and continuation)

The 11 backend docs below cover Python modules by cohesive subsystem.
`webui-frontend.md` covers the CSS, shared JavaScript, and all page templates;
`ops-and-packaging.md` covers installers, launchers, update scripts, and
deployment helpers. `docs/CODEMAP.md` is the generated symbol map, not proof
that a file was read. The test-suite audit remains incomplete; see HANDOFF.

| Doc | Covers |
|---|---|
| [`modules/data-parsers.md`](modules/data-parsers.md) | `data/parsers/` + `data/parsers.py`, `ocr.py`, `converter.py`, `sentence_transformer_local.py`, `validator.py` — file-format extraction |
| [`modules/data-prep.md`](modules/data-prep.md) | `data/prep/` — QA-pair mining, coverage audit, dataset export |
| [`modules/data-fs.md`](modules/data-fs.md) | `data/fs/` — project filesystem: files, chunks, parsed content, QA records |
| [`modules/rag.md`](modules/rag.md) | `data/rag_portable/` (live) + `rag/` (legacy, still used in 3 places) — both RAG stacks, fully reconciled |
| [`modules/webui-routes-core.md`](modules/webui-routes-core.md) | `webui/routes/{data,projects,models,exports,chat_v2,file_library,...}.py` — project lifecycle, data, settings, models |
| [`modules/webui-routes-workflow.md`](modules/webui-routes-workflow.md) | `webui/routes/{training,data_prep,data_prep_chat,benchmarks,testing,rag,pages,activity}.py` |
| [`modules/training.md`](modules/training.md) | `training/` — `TrainingEngine`, GGUF export, VRAM estimation, augmentation/guard modules |
| [`modules/db-models.md`](modules/db-models.md) | `db/` (SQLite CRUD) + `models/` (loading, registry, providers) |
| [`modules/testing-engine.md`](modules/testing-engine.md) | `testing/` — `InferenceEngine`, suite builders/runners, judging, scoring |
| [`modules/benchmarks-compare.md`](modules/benchmarks-compare.md) | `benchmarks/` (public MMLU/GSM8K/HellaSwag + comparison) — `compare/` retired as a dead duplicate |
| [`modules/core-cli-entrypoints.md`](modules/core-cli-entrypoints.md) | `webui/app.py` + non-route helpers, `cli/`, top-level `config.py`/`naming.py`/`__init__.py`, `templates/` |
| [`modules/webui-frontend.md`](modules/webui-frontend.md) | `webui/static/{css,js}/` and the Jinja page templates; lifecycle, navigation, UI contracts |
| [`modules/ops-and-packaging.md`](modules/ops-and-packaging.md) | Installers, launchers, service unit, update scripts, packaging and QA runner |

**Known gaps not fixed in the 2026-10-01 pass** (flagged by multiple lanes, need a product decision rather than a blind fix — see `HANDOFF.md`):
- Three overlapping export-listing routes (`training.py`, `exports.py` ×2) — pick one, delete the others.
- `webui/routes/projects.py`'s `/rags/{rid}/*` (legacy RAGManager) and `webui/routes/comparison.py`'s `/api/compare/*` have no frontend caller — migrate onto `rag_portable`/delete, or document as intentional back-compat.
- `config.py`'s `Settings.rag_store_path`/`rag_embedding_model` duplicate `Settings.rag.store_path`/`embedding_model` with nothing syncing them.
