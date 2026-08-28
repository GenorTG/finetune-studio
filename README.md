<div align="center">

# ⚡ Finetune Studio

**Full-stack LLM fine-tuning, inference, and evaluation — all in one app.**

Train custom models, run inference with vision support, benchmark quality, and manage your entire ML workflow from a single WebUI.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GPU: CUDA/ROCm/CPU](https://img.shields.io/badge/GPU-CUDA%20%7C%20ROCm%20%7C%20CPU-orange.svg)](#installation)

---

</div>

## 🚀 What is Finetune Studio?

Finetune Studio is a self-hosted, GPU-aware ML platform that covers the full lifecycle of language model development:

- **Train** — SFT (supervised fine-tuning) with LoRA/QLoRA, full parameter, and RLHF/DPO. Real-time loss curves, VRAM profiling, automatic checkpoint management.
- **Inference** — Load any GGUF or Safetensors model with an LM Studio-style loader. Vision/multimodal support, auto-unload on idle, persistent settings per model.
- **Evaluate** — Run MMLU, HellaSwag, ARC, TruthfulQA, GSM8K, Winogrande benchmarks. Tool-calling evaluation. Side-by-side model comparison.
- **Data** — Import, validate, convert, and organize training datasets. Built-in RAG (retrieval-augmented generation) with ChromaDB.
- **Projects** — Organize everything into project workspaces with their own data, training runs, and results.

All wrapped in a clean, dark-themed WebUI with real-time training monitoring and GPU status.

## ✨ Features

### 🧠 Training Engine
- **SFT / QLoRA / LoRA / DPO** — all major fine-tuning methods
- **VRAM profiler** — predict memory usage before training starts
- **Real-time monitoring** — live loss curves, step progress, GPU utilization
- **Auto-checkpointing** — resume from the last checkpoint automatically
- **Multi-GPU** — distributed training via Accelerate

### 💬 Inference
- **GGUF + Safetensors** — load any format, auto-detects model type
- **LM Studio-style loader** — GPU layers slider, memory estimation bar, advanced settings
- **Vision/multimodal** — auto-detects mmproj files, image upload + chat
- **Auto-unload** — configurable idle timeout, manual unload button
- **Chat template engine** — supports ChatML, Llama, Gemma, Qwen, and custom templates
- **Per-model settings** — remember context length, GPU layers, etc. per model

### 📊 Benchmarks
- **Standard suites** — MMLU, HellaSwag, ARC-C, TruthfulQA, GSM8K, Winogrande
- **Tool calling** — evaluate function-calling accuracy
- **Side-by-side** — compare two models head-to-head

### 📁 Data Pipeline
- **Import** — JSONL, CSV, Alpaca, ShareGPT, and more
- **Validation** — schema checking, format conversion, deduplication
- **RAG** — ingest documents into ChromaDB for retrieval-augmented training data

### 🎯 Smart Model Discovery
- Scans `models/gguf/` and `models/safetensors/` automatically
- Reads GGUF metadata for proper names, architecture, layer counts
- Detects vision capability via mmproj presence
- Filters out junk (tokenizers, mmprojectors <0.5GB)

## 📦 Installation

### Quick Start (Recommended)

```bash
git clone https://github.com/GenorTG/finetune-studio.git
cd finetune-studio
bash install.sh   # Auto-detects GPU, installs correct wheels
bash run.sh       # Starts the WebUI on http://localhost:7860
```

### GPU-Aware Install

`install.sh` automatically detects your hardware:

| GPU | PyTorch | llama-cpp-python |
|-----|---------|-----------------|
| NVIDIA (CUDA) | `cu124` / `cu130` wheels | abetlen CUDA wheels |
| AMD (ROCm) | `rocm6.x` wheels | source build |
| Intel (XPU) | Intel extension | source build |
| CPU only | CPU wheels | source build |

```bash
bash install.sh          # Auto-detect
bash install.sh --cpu    # Force CPU
make install             # Same as install.sh
make install-cpu         # CPU only
```

### Manual Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[gguf]"
```

### Requirements

- Python 3.12 or 3.13
- NVIDIA GPU with CUDA 12.4+ (recommended) or CPU-only
- 16GB+ RAM (32GB+ recommended for 7B+ models)

## 🖥️ Usage

### WebUI

```bash
bash run.sh
# → http://localhost:7860
```

### CLI

```bash
finetune-studio          # Launch WebUI
finetune-studio train    # Run training
finetune-studio test     # Run test suite
```

### Project Workflow

1. **Create a project** → Dashboard → "+ New Project"
2. **Add data** → Project → Data tab → Import JSONL/CSV
3. **Configure training** → Training tab → Set hyperparameters
4. **Train** → Click "Start Training" → Watch live loss curves
5. **Evaluate** → Testing tab → Run benchmarks
6. **Infer** → Inference page → Load model → Chat

## 🏗️ Architecture

```
finetune-studio/
├── src/finetune_studio/
│   ├── webui/          # FastAPI + Jinja2 WebUI
│   │   ├── routes/     # API endpoints & page routes
│   │   ├── templates/  # Jinja2 HTML templates
│   │   └── static/     # CSS, JS, favicon
│   ├── training/       # Training engine (SFT, DPO, LoRA)
│   ├── testing/        # Inference engine + benchmark runner
│   ├── benchmarks/     # MMLU, HellaSwag, ARC, etc.
│   ├── data/           # Dataset import, validation, conversion
│   ├── templates/      # Chat template renderer
│   ├── models/         # Model registry & discovery
│   ├── rag/            # RAG pipeline (ChromaDB + embeddings)
│   ├── config.py       # Configuration
│   └── db.py           # SQLite database
├── tests/              # Unit + integration tests
├── install.sh          # GPU-aware installer
├── run.sh              # Launcher
└── pyproject.toml      # Package metadata
```

## 🔧 Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FTS_HOST` | `0.0.0.0` | WebUI bind host |
| `FTS_PORT` | `7860` | WebUI port |
| `FTS_IDLE_TIMEOUT` | `1800` | Auto-unload timeout (seconds) |
| `FTS_MODELS_DIR` | `./models` | Model scan directory |
| `FTS_DB_PATH` | `./data/finetune.db` | SQLite database path |

## 🤖 Supported Models

Any model in GGUF or Safetensors format:

- **Qwen** — 3.8-27B, 3.8-Flash, 2.5-VL, etc.
- **Llama** — 3.x, 3.1, 3.2
- **Mistral** — 7B, 8x7B
- **Gemma** — 2B, 7B, 9B
- **Phi** — 3, 4
- **GPT-2 / GPT-NeoX** — classic models
- Any HuggingFace model compatible with `transformers`

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Python, FastAPI, Uvicorn |
| **Frontend** | Jinja2, Tailwind CSS, HTMX |
| **Training** | PyTorch, TRL, PEFT, Accelerate |
| **Inference** | llama-cpp-python (GGUF), transformers (Safetensors) |
| **Database** | SQLite + aiosqlite |
| **RAG** | ChromaDB, sentence-transformers |
| **Build** | Hatchling, pip, Make |

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built on top of these incredible open-source projects:

| Project | What it does | License |
|---------|-------------|---------|
| [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) | GGUF inference | MIT |
| [TRL](https://github.com/huggingface/trl) | Training (SFT, DPO, RLHF) | Apache-2.0 |
| [PEFT](https://github.com/huggingface/peft) | LoRA, QLoRA adapters | Apache-2.0 |
| [Transformers](https://github.com/huggingface/transformers) | Model loading & tokenization | Apache-2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) | Deep learning framework | BSD-3 |
| [FastAPI](https://github.com/fastapi/fastapi) | Web framework | MIT |
| [Tailwind CSS](https://tailwindcss.com) | Utility-first CSS | MIT |
| [HTMX](https://htmx.org) | HTML-first interactivity | BSD-2 |
| [ChromaDB](https://github.com/chroma-core/chroma) | Vector database | Apache-2.0 |
| [Accelerate](https://github.com/huggingface/accelerate) | Multi-GPU training | Apache-2.0 |
| [gguf](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py) | GGUF metadata reading | MIT |
| [Rich](https://github.com/Textualize/rich) | Terminal formatting | MIT |

---

<div align="center">

**Made with ❤️ for the open-source ML community**

</div>
