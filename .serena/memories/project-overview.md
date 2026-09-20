# finetune-studio — project memory

Local fine-tune + data-prep WebUI (Python/FastAPI, `src/finetune_studio/`). Dev on **genorbox1** (this repo), GPU runs on **fan-dragon** (`/home/genortg/finetune-studio`, systemd **user** unit `finetune-studio`, port 7860).

## Workflow (law)
- Read `docs/WORKPLAN.md` (order is law) → `HANDOFF.md` (≤120 lines, rewrite never append) → `docs/CODEMAP.md` (symbol map; regenerate with `make codemap`, commit with any symbol change).
- Deploy: commit+push → `ssh fan-dragon 'bash -c "cd /home/genortg/finetune-studio && bash update.sh"'`. Never edit files on fan-dragon; never `make run` on genorbox1.
- Tests: `.venv/bin/python -m pytest tests/<file> -v --tb=short` (venv is uv-managed). Lint: `.venv/bin/python -m ruff check <changed files>` — never `make lint` (swallows failures).
- Visual changes: browser-screenshot verify on `http://fan-dragon:7860`, HTTP codes are not enough.

## Architecture anchors
- Routes: `src/finetune_studio/webui/routes/` (training.py `start_training` POST /api/training/start; rag.py; pages.py wizard route; versions.py). Templates: `webui/templates/` — wizard `project_wizard.html`, base `base.html` (cache-bust `app.css?v=NN` on CSS edits).
- `.app` grid rows are explicit: session-bar=1, breadcrumb=2, subnav=3, main=4 (collision here once rendered the sticky bar mid-page).
- RAG portable package: `data/rag_portable/` — `mcp_package.py` builds tarball (server.py + install.sh + **setup.sh** guided deploy: port/service/MCP-ENTRY.txt; `--uninstall`); `include_models=true` bundles embedder+reranker (~1.4 GB, compresslevel=1).
- Model-path resolution: `training/merge_base.py` (`_find_in_hub_cache`, `_find_in_app_hf_models`, `resolve_merge_base`); training start now guards hub downloads via `routes/training.py::_resolve_model_path` (allow_download=true to override).
- DB: `db/__init__.py` re-exports a curated subset only — some symbols live on submodules (`db.datasets.get_dataset_by_path`); call sites must import bare names.

## Gotchas (hard-won)
- Training runs in-process: never fork dataset workers (`UNSLOTH_DATASET_NUM_PROC=0` pinned at import in `training/engine.py`).
- Keep unsloth OUT of the server process for inference (patch pollution → gibberish); train in subprocess, infer with plain transformers.
- Merge nf4 LoRA onto the 16-bit sibling base via PeftModel, not `save_pretrained_merged`.
- fan-dragon shell is fish: wrap ssh commands in `bash -c "..."` with escaped `\$`.
- GPU: check `nvidia-smi --query-compute-apps` before runs; never kill foreign processes (ComfyUI etc.).
- Never compare strict-% across different bench suite instruments; human-grade judging per `docs/judging/PROTOCOL.md`.
- FastAPI uploads use `files=` (plural); browser-tool uploads go under `/tmp/openclaw/uploads/`; no native alert()/confirm() in the WebUI (freezes the tool) — use `fts.notify`.
