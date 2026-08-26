# Finetune Studio — Spec

## Overview
Full-featured model training app with Python WebUI, built on fan-dragon.
Uses UV for package management, FastAPI for backend, Jinja2+HTMX for frontend.

## Target Machine
- **fan-dragon**: RTX 3090 24GB, AMD Ryzen 9 5900X, 64GB RAM, Python 3.13, fish shell
- **Existing env**: `/home/genortg/1TB-SAMSUNG/Comfy/comfy-env/` has torch 2.10, unsloth, transformers, trl
- **Project dir**: `/home/genortg/finetune-studio/`

## Architecture
```
finetune-studio/
├── pyproject.toml              # UV project config
├── README.md
├── install.sh                  # Bootstrap: installs UV, creates venv, installs deps
├── run.sh                      # Start the app
├── src/
│   └── finetune_studio/
│       ├── __init__.py
│       ├── __main__.py         # python -m finetune_studio
│       ├── config.py           # Settings (paths, defaults)
│       ├── models/
│       │   ├── __init__.py
│       │   ├── loader.py       # Load HF models (safetensors), GGUF via llama-cpp-python
│       │   └── registry.py     # Scan directories, list available models
│       ├── training/
│       │   ├── __init__.py
│       │   ├── engine.py       # Training loop (unsloth SFTTrainer)
│       │   ├── data.py         # Dataset loading, formatting, tokenization
│       │   └── monitor.py      # SSE progress events (loss, lr, step, eta)
│       ├── data/
│       │   ├── __init__.py
│       │   ├── organizer.py    # Scan, categorize, dedup training data
│       │   ├── validator.py    # Validate JSONL/JSON format
│       │   └── converter.py    # Convert between formats (CSV↔JSONL, etc.)
│       ├── testing/
│       │   ├── __init__.py
│       │   ├── inference.py    # Load model, generate response
│       │   └── suite.py        # Run test suites, score results
│       └── webui/
│           ├── __init__.py
│           ├── app.py          # FastAPI app
│           ├── routes/
│           │   ├── __init__.py
│           │   ├── pages.py    # HTML page routes
│           │   ├── models.py   # Model API routes
│           │   ├── training.py # Training API routes
│           │   ├── data.py     # Data management routes
│           │   └── testing.py  # Testing API routes
│           ├── static/
│           │   ├── css/
│           │   │   └── app.css
│           │   └── js/
│           │       └── app.js  # HTMX + SSE handling
│           └── templates/
│               ├── base.html
│               ├── index.html
│               ├── models.html
│               ├── training.html
│               ├── data.html
│               └── testing.html
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_models.py
│   └── test_training.py
└── data/                       # Default training data dir
    └── .gitkeep
```

## Features

### 1. Model Management
- Scan directories for models (safetensors, GGUF)
- Load model info (name, size, architecture, parameters)
- Select model for training or inference
- Support: HuggingFace format, GGUF (via llama-cpp-python)

### 2. Training Data Management
- Upload files (JSONL, JSON, CSV, TXT)
- Validate format (check required fields: messages, text, etc.)
- Preview data (show first N examples)
- Organize: categorize, tag, dedup
- Convert between formats
- Split into train/validation sets

### 3. Training Engine
- Configure: base model, LoRA rank, learning rate, epochs, batch size, etc.
- Start/stop training
- Realtime progress via SSE (loss curve, current step, ETA, learning rate)
- Save checkpoints
- Export to GGUF after training

### 4. Model Testing (Inference)
- Load any model (safetensors or GGUF)
- Interactive chat interface
- Run test suites: pre-defined question sets, score results
- Compare models side-by-side
- Export test results

### 5. WebUI
- Dashboard: overview of models, training jobs, data
- Models page: browse, select, view details
- Training page: configure, start, monitor progress
- Data page: upload, validate, organize
- Testing page: chat, test suites, compare

## Tech Stack
- **Package manager**: UV
- **Backend**: FastAPI + uvicorn
- **Frontend**: Jinja2 + HTMX + Tailwind CSS (CDN)
- **Training**: unsloth + transformers + trl
- **Inference**: transformers (safetensors) + llama-cpp-python (GGUF)
- **Progress**: Server-Sent Events (SSE)
- **Database**: SQLite (for model registry, training history)

## Dependencies (pyproject.toml)
```toml
[project]
name = "finetune-studio"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "jinja2>=3.1.0",
    "python-multipart>=0.0.9",
    "htmx>=0.0.1",
    "torch>=2.1.0",
    "unsloth>=2026.1.0",
    "transformers>=4.40.0",
    "trl>=0.8.0",
    "peft>=0.10.0",
    "datasets>=2.18.0",
    "accelerate>=0.28.0",
    "bitsandbytes>=0.43.0",
    "llama-cpp-python>=0.3.0",
    "safetensors>=0.4.0",
    "huggingface-hub>=0.20.0",
    "rich>=13.0.0",
    "pydantic>=2.0.0",
    "aiosqlite>=0.19.0",
]
```

## Install Flow
1. SSH to fan-dragon
2. `cd /home/genortg/finetune-studio && bash install.sh`
3. install.sh:
   - Checks Python version (>=3.11)
   - Installs UV if not present
   - Creates venv with UV
   - Installs dependencies
   - Creates data/ directory
4. `bash run.sh` or `uv run finetune-studio`
5. Opens browser to http://localhost:7860

## Run Flow
1. Start FastAPI server on port 7860
2. Scan model directories on startup
3. Load training history from SQLite
4. Serve WebUI
5. Accept training jobs via API
6. Stream progress via SSE
