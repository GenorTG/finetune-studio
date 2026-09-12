# Attributions

Finetune Studio is built on the shoulders of open-source giants. Every package
below is used under its declared license. Thank you to the maintainers.

Versions listed are what this build was developed and tested against
(Sept 2026). Lower versions may also work; see `pyproject.toml` for minimum
constraints.

---

## Core (Python)

| Package | Version | License | What we use it for | Project |
|---|---|---|---|---|
| **torch** | 2.13.0 | BSD-3-Clause | Tensor ops, GPU kernels, training | https://pytorch.org |
| **transformers** | 5.15.0 | Apache-2.0 | Model loading, generation, training | https://github.com/huggingface/transformers |
| **peft** | (latest) | Apache-2.0 | LoRA / QLoRA adapters | https://github.com/huggingface/peft |
| **trl** | (latest) | Apache-2.0 | SFTTrainer for instruction tuning | https://github.com/huggingface/trl |
| **accelerate** | (latest) | Apache-2.0 | Multi-GPU / mixed precision | https://github.com/huggingface/accelerate |
| **datasets** | (latest) | Apache-2.0 | Streaming dataset loading | https://github.com/huggingface/datasets |
| **safetensors** | 0.8.0 | Apache-2.0 | Safe model weight format | https://github.com/huggingface/safetensors |
| **tokenizers** | 0.22.2 | Apache-2.0 | Fast text tokenization | https://github.com/huggingface/tokenizers |
| **huggingface_hub** | 1.28.0 | Apache-2.0 | Model downloads, caching | https://github.com/huggingface/huggingface_hub |
| **sentence-transformers** | 6.0.0 | Apache-2.0 | Embedding models for RAG | https://github.com/UKPLab/sentence-transformers |
| **chromadb** | 1.5.9 | Apache-2.0 | Vector database for RAG | https://github.com/chroma-core/chroma |
| **gguf** | (latest) | MIT | GGUF model format reader | https://github.com/ggerganov/ggml |
| **llama-cpp-python** | 0.3.35 | MIT | Local GGUF inference (CPU+GPU) | https://github.com/abetlen/llama-cpp-python |
| **bitsandbytes** | (latest) | MIT | 4-bit/8-bit quantization for QLoRA | https://github.com/TimDettmers/bitsandbytes |

## Web framework

| Package | Version | License | What we use it for | Project |
|---|---|---|---|---|
| **fastapi** | 0.141.1 | MIT | HTTP API + page routes | https://github.com/tiangolo/fastapi |
| **uvicorn** | 0.52.4 | BSD-3-Clause | ASGI server | https://github.com/encode/uvicorn |
| **starlette** | 1.6.0 | BSD-3-Clause | ASGI toolkit (FastAPI dep) | https://github.com/encode/starlette |
| **jinja2** | 3.1.6 | BSD-3-Clause | Server-side templating | https://github.com/pallets/jinja |
| **aiosqlite** | (latest) | MIT | Async SQLite (project DB) | https://github.com/omnilib/aiosqlite |
| **python-multipart** | (latest) | Apache-2.0 | Form / file upload parsing | https://github.com/Kludex/python-multipart |
| **pydantic** | 2.13.2 | MIT | Data validation | https://github.com/pydantic/pydantic |
| **aiofiles** | 25.1.0 | Apache-2.0 | Async file I/O | https://github.com/Tinche/aiofiles |
| **rich** | 14.2.0 | MIT | Pretty terminal output | https://github.com/Textualize/rich |

## Data processing

| Package | Version | License | What we use it for | Project |
|---|---|---|---|---|
| **numpy** | 2.5.2 | BSD-3-Clause | Numerical arrays | https://github.com/numpy/numpy |
| **pandas** | (latest) | BSD-3-Clause | Tabular data (eval datasets) | https://github.com/pandas-dev/pandas |
| **scikit-learn** | 1.9.0 | BSD-3-Clause | Benchmark scoring | https://github.com/scikit-learn/scikit-learn |
| **scipy** | 1.18.0 | BSD-3-Clause | Statistical tests | https://github.com/scipy/scipy |
| **requests** | 2.33.1 | Apache-2.0 | HTTP client | https://github.com/psf/requests |

## Testing / QA

| Package | Version | License | What we use it for | Project |
|---|---|---|---|---|
| **playwright** | (latest) | Apache-2.0 | Live-browser UI tests | https://github.com/microsoft/playwright |

## Frontend (browser, no build step)

| Asset | Source | License | What |
|---|---|---|---|
| **JetBrains Mono** | Google Fonts | OFL-1.1 | UI body font |
| **Share Tech Mono** | Google Fonts | OFL-1.1 | Headers, terminal vibe |
| **VT323** | Google Fonts | OFL-1.1 | Status displays, captions |
| **Press Start 2P** | Google Fonts | OFL-1.1 | Pixel-art accents |

All served via Google Fonts CDN; not bundled with the app.

---

## Generated assets

The pixel-art sprite icons in `static/js/sprites.js` were authored for this
project and are released under the same license as the rest of the code
(Finetune Studio Non-Commercial and Research License v1.0). No third-party sprite assets are used.

---

## License of this project

Finetune Studio itself is released under the **Finetune Studio Non-Commercial and Research License v1.0**. See `LICENSE`.

This is a *source-available* license: personal, scientific, and non-commercial use is free. Commercial use and commercial model training require a separate paid license. See `docs/LEGAL.md` for the full legal analysis.

---

## Updates

If you bump a dependency, please update the version column above. Run
`python -m pip list` after `pip install -e .` and refresh.
