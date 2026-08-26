# Finetune Studio — WebUI Expansion Plan

## Audit (2026-08-26)

Backend is ~80% done. WebUI is ~10% done. UI exposes 5 thin stubs (Dashboard, Models, Training, Data, Testing). Backend modules that have NO UI surface yet:

- RAG (manager, ingest, store, query)
- Training engine (runs, progress, settings)
- Benchmarks (comparison, real_benchmarks, scoring, samplers, tool_calling)
- Quality / validation / dedup
- Comparison (model A vs model B)
- MCP tools
- Agentic loop
- Inference (chat)

## Gaps (from Chris's feedback)

1. **Data prep** — only approve/reject-style. Needs inline manual edit (per-row).
2. **Missing UI features** — training runs history, benchmarks, agentic tool playground, model comparison, RAG management, project scoping.
3. **Project → multi-RAG architecture** — currently RAG is a single global store. Needs: Project has many RAGs (legal, social, paperwork, …). Trainable. Combinable at inference.
4. **Training runs with versioning + metadata** — needs: run history per project, persisted settings + base model + data + pre/post benchmark scores, compare runs, promote to production.
5. **Tooltips + manual** — zero. Inline help on every input + a Help page with full docs.

## Architecture (ponytail: minimal, stdlib, no over-engineer)

### Data layer
- **One SQLite DB** (`finetune.db`, stdlib `sqlite3`) — single source of truth for projects/rags/runs/benchmarks. No ORM, raw SQL + thin helpers.
- **JSONL on disk** for training data — unchanged.
- **Vector stores** stay on disk under `~/.local/share/finetune-studio/rags/{project_id}/{rag_id}/`.

### New entities
```
projects        (id, name, description, base_model, system_prompt, created_at)
project_rags    (id, project_id, name, description, tags, store_path, doc_count, created_at)
training_runs   (id, project_id, name, base_model, data_path, rag_ids_json,
                 settings_json, system_prompt, status, started_at, finished_at,
                 output_path, metrics_json, parent_run_id, notes)
benchmark_runs  (id, run_id, suite_name, scores_json, time_ms, ran_at)
```

### New routes
- `/api/projects` (list, create, get, rename, delete)
- `/api/projects/{id}/rags` (CRUD + ingest + query)
- `/api/projects/{id}/runs` (list, create, get, delete, restore)
- `/api/projects/{id}/runs/{rid}/benchmark` (run + persist)
- `/api/data/{project_id}/{dataset}/rows` (list, get, patch, approve, reject, delete)
- `/api/agentic/tools` (list MCP tools)
- `/api/agentic/run` (run agentic step with selected tools)

### New tabs
1. **Projects** — list, create, select. Click into project.
2. **Project** (detail) — RAGs (CRUD + ingest/query), Runs (history), Production model.
3. **Data Prep** — moved into project. Inline row editor. Approve/reject. Quality dashboard.
4. **Training** — config form, runs history with metadata, diff between runs, pre/post benchmark deltas, promote button.
5. **Benchmarks** — pick run + suite, run, show scores, compare across runs.
6. **Inference Chat** — pick project (multi-RAG toggle per RAG), load model, chat, RAG sources inline.
7. **Agentic Tools** — load model, enable tools (RAG/calc/web), show tool trace, test cases.
8. **Help** — markdown docs + inline tooltips everywhere.

### Tooltip system
- `data-tip="…"` attribute on any element → global JS renders floating tooltip on hover/focus.
- Add `data-tip` to every form label, button, key concept.

## Slices

| Slice | What | Who |
|-------|------|-----|
| A — Foundation | SQLite schema + db.py + Project/RAG/Run CRUD routes + Projects tab + Project detail tab | me (this session) |
| B — Data prep editor | row list, inline edit, approve/reject, save back to JSONL | sub-agent |
| C — Benchmarks tab | run suites against runs, persist scores, side-by-side compare | sub-agent |
| D — Agentic tools tab | tool calling playground with MCP tools | sub-agent |
| E — Help + tooltips | tooltip system + Help page + add tooltips everywhere | sub-agent |
| F — Inference chat v2 | project-aware, multi-RAG toggle, sources inline, system prompt editor | sub-agent |

## Order of work

Foundation (A) unblocks all others — must come first.
Then B, C, D, E, F can run in parallel.

## Non-goals (YAGNI)

- No multi-user / auth. Single-user studio on local machine.
- No cloud sync. Local SQLite.
- No version control of data. Just JSONL on disk + last-modified tracking.
- No fancy diff view. Just table comparison for runs.
- No plugin system. Just core features.
