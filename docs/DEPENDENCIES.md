# Python Dependencies — What Uses What

> Verified by import scan of `src/` + `tests/` + `scripts/` on 2026-09-11.
> "lazy" = imported inside functions (the codebase's deliberate boot-time pattern —
> keeps `uvicorn` startup ~2s despite the ML stack). Do not hoist these to top level.

## Core runtime (`[project.dependencies]`)

| Package | Used by | Purpose |
|---|---|---|
| `fastapi` | all `webui/routes/*` (38 top-level imports) | HTTP API + SSR page routes |
| `uvicorn[standard]` | `cli/commands/webui.py` | ASGI server (`fts webui`) |
| `jinja2` | `templates/renderer.py`, `webui/routes/pages.py` | HTML page templates + chat-template rendering |
| `python-multipart` | webui upload routes | `UploadFile` form parsing (implicit dep of FastAPI file uploads) |
| `pydantic` | `data_prep`, `hf_models`, `quality`, `rag` routes | request/response models |
| `torch` | `testing/inference.py`, `webui/{pages,system,exports}`, `training/*` (via TRL) | HF-model inference, device/VRAM probes (lazy) |
| `transformers` | `testing/inference.py`, `rag/store.py`, `data/shared_models.py`, `rag_portable/constants.py`, `training/engine.py` (lazy) | HF tokenizers/models, chat templates, merge |
| `trl` | `training/engine.py`, `training/vram/profile.py` | `SFTTrainer` LoRA/QLoRA fine-tuning (lazy) |
| `peft` | `training/engine.py`, `vram/profile.py`, `pages.py` | LoRA adapters, adapter merge (lazy) |
| `datasets` | `benchmarks/real_benchmarks.py`, `db/{datasets,reviews,connection}`, `training/data.py` | JSONL→Dataset, benchmark sets, counting |
| `accelerate` | **no direct import** — required at runtime by `trl`/`transformers` Trainer | distributed/amp backend. **Keep it.** |
| `safetensors` | `testing/inference.py`, `data/shared_models.py`, `cli` | checkpoint format, GGUF↔HF paths |
| `huggingface-hub` | `webui/routes/hf_models.py` | model downloads + resume (`hf download`) |
| `sentence-transformers` | `rag/store.py`, `data/shared_models.py`, `rag_portable/{embedders,rerankers}.py` | embeddings (`all-MiniLM-L6-v2` default) + rerank |
| `chromadb` | `rag/store.py` | legacy vector store (portable store in `rag_portable/` is the current path — slated for removal with Stage 3) |
| `sentencepiece` | `webui/routes/exports.py` | tokenizer export for GGUF conversion |
| `gguf` | `testing/inference.py`, `templates/*` | GGUF metadata parsing (arch/quant/template) |
| `requests` | `compare/engine.py`, `models/providers.py` | external OpenAI-compat API calls (judge + provider) |
| `aiofiles` | `webui/routes/data.py` | async upload writes |
| `rich` | **0 imports** | leftover — removal candidate |
| `aiosqlite` | **0 imports** (`db/` uses stdlib `sqlite3`) | leftover — removal candidate |
| `tiktoken` | **0 imports** | leftover — removal candidate |

## Not in pyproject — installed by `install.sh` (GPU-aware)

| Package | Why it's special |
|---|---|
| `llama-cpp-python` | CUDA variant depends on the host driver — `install.sh` maps driver→CUDA (≥555→cu132 prebuilt wheel, older→cu124/121/118, CPU fallback) and installs the matching wheel. Used by `testing/inference.py`, `templates/renderer.py`, `benchmarks/samplers.py`, `webui/{data_prep_chat,pages,exports}`, `models/{manager,providers,loader}` (9 modules, all lazy). A plain pyproject pin would break cross-host installs. |
| llama.cpp **CLI** (`llama-quantize`, `convert_hf_to_gguf.py`) | built by `install.sh`/`update.sh` into `LLAMA_CPP_DIR` (project-local `.llama.cpp/`); needed by the merge→GGUF export endpoint. |

## Optional extras

```bash
pip install -e '.[bitsandbytes]'   # 4-bit QLoRA base loading
pip install -e '.[dev]'            # pytest, pytest-asyncio, ruff
```

## Python floor

- Developed/tested on **Python 3.13**.
- `requires-python` per `pyproject.toml`; TRL 0.12+ and transformers 4.40+ pin the floor.

## Cleanup summary (when you touch pyproject)

Remove: `rich`, `aiosqlite`, `tiktoken` (0 imports, verified 2026-09-11).
Keep despite no direct import: `accelerate` (Trainer runtime), `python-multipart`
(FastAPI uploads), `sentencepiece` (tokenizer export).
`chromadb` stays until Stage 3 (RAG refactor) retires `rag/store.py`.
