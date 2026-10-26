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
| `pydantic` | `data_prep`, `hf_models`, `quality`, `brain` routes | request/response models |
| `torch` | `testing/inference.py`, `webui/{pages,system,exports}`, `training/*` (via TRL) | HF-model inference, device/VRAM probes (lazy) |
| `transformers` | `testing/inference.py`, `data/shared_models.py`, `smart_brain/constants.py`, `training/engine.py` (lazy) | HF tokenizers/models, chat templates, merge |
| `trl` | `training/engine.py`, `training/vram/profile.py` | `SFTTrainer` LoRA/QLoRA fine-tuning (lazy) |
| `peft` | `training/engine.py`, `vram/profile.py`, `pages.py` | LoRA adapters, adapter merge (lazy) |
| `datasets` | `benchmarks/real_benchmarks.py`, `db/{datasets,reviews,connection}`, `training/data.py` | JSONL→Dataset, benchmark sets, counting |
| `accelerate` | **no direct import** — required at runtime by `trl`/`transformers` Trainer | distributed/amp backend. **Keep it.** |
| `safetensors` | `testing/inference.py`, `data/shared_models.py`, `cli` | checkpoint format, GGUF↔HF paths |
| `huggingface-hub` | `webui/routes/hf_models.py` | model downloads + resume (`hf download`) |
| `sentence-transformers` | `data/shared_models.py`, `smart_brain/{embedders,rerankers}.py` | Smart Brain embeddings (`multilingual-e5-large` default) + reranking |
| `sentencepiece` | `webui/routes/exports.py` | tokenizer export for GGUF conversion |
| `gguf` | `testing/inference.py`, `templates/*` | GGUF metadata parsing (arch/quant/template) |
| `requests` | `compare/engine.py`, `models/providers.py` | external OpenAI-compat API calls (judge + provider) |
| `httpx` | `webui/routes/data_prep_chat.py`, FastAPI `TestClient` | Guide / agent chat API-provider calls; was only transitive via chromadb until 2026-10-14 |
| `aiofiles` | `webui/routes/data.py` | async upload writes |
| `rich` | **0 imports** | leftover — removal candidate |
| `aiosqlite` | **0 imports** (`db/` uses stdlib `sqlite3`) | leftover — removal candidate |
| `tiktoken` | **0 imports** | leftover — removal candidate |

## Not in pyproject — installed by `install.sh` (GPU-aware)

| Package | Why it's special |
|---|---|
| `llama-cpp-python` | GPU backend depends on the host — `scripts/accel_plan.py` (used by `install.sh`/`update.sh`) picks the abetlen prebuilt CUDA wheel (cu132/cu130/cu12x) or a source build with `GGML_CUDA`/`GGML_HIP`/`GGML_SYCL`/`GGML_VULKAN`/`GGML_METAL`; CPU only when no GPU exists. Install it with `--extra-index-url` + `--index-strategy first-index` (never `--index-url`, which silently lets PyPI win with the CPU sdist) and `--no-cache` for source builds (uv does not key its build cache on `CMAKE_ARGS`). Used by `testing/inference.py`, `templates/renderer.py`, `benchmarks/samplers.py`, `webui/{data_prep_chat,pages,exports}`, `models/{manager,providers,loader}` (9 modules, all lazy). A plain pyproject pin would break cross-host installs. |
| llama.cpp **CLI** (`llama-quantize`, `convert_hf_to_gguf.py`) | built by `install.sh`/`update.sh` into `LLAMA_CPP_DIR` (project-local `.llama.cpp/`); needed by the merge→GGUF export endpoint. |
| `flash-linear-attention`, `causal-conv1d` | Qwen3.5 / Qwen3-Next linear-attention kernels; transformers uses them whenever they import (no app import). `accel_plan.py install linear-attn` (run by every installer): NVIDIA sm_75+ on Linux, `causal-conv1d` built with `--no-build-isolation --no-cache` by an nvcc of torch's CUDA major; each probed on the GPU and uninstalled if it fails. `accel/kernels.py` hides them at run time on a device that cannot run them (Pascal, CPU, MPS, XPU). Opt out: `FTS_LINEAR_ATTN=0`. |

## Optional extras

```bash
# install.sh already adds bitsandbytes (4-bit QLoRA base loading) where the hardware supports it.
# Dev tools (pytest, pytest-asyncio, ruff), resolved against the installed torch:
uv pip install --python .venv/bin/python -c .venv/torch-constraints.txt -e '.[dev]'
```

## Python floor

- Developed and tested on **Python 3.12** (CI and the dev host); 3.13 is accepted by `requires-python`
  but not tested.
- `requires-python` per `pyproject.toml`; TRL 0.12+ and transformers 4.40+ pin the floor.

## Cleanup summary (when you touch pyproject)

Remove: `rich`, `aiosqlite`, `tiktoken` (0 imports, verified 2026-09-11).
Keep despite no direct import: `accelerate` (Trainer runtime), `python-multipart`
(FastAPI uploads), `sentencepiece` (tokenizer export).
`chromadb` was removed 2026-10-14 with the legacy `rag/` package (the Smart Brain is `smart_brain/`).
