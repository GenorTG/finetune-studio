# Finetune Studio

Full-featured model training app with WebUI + CLI + RAG + Model Comparison.

## Requirements
- **Python 3.12 or 3.13** (auto-detected)
- **CUDA GPU** (RTX 3090 recommended, GTX 1070+ works)
- **UV** package manager (installed automatically)

## Install

### Linux / macOS
```bash
cd finetune-studio
bash install.sh

# Force Python version:
PYTHON_VERSION=3.12 bash install.sh
```

### Windows
```powershell
cd finetune-studio
.\install.ps1
```

## Run

### WebUI
```bash
bash run.sh
# → http://localhost:7860
```

### CLI
```bash
finetune-studio --help
# or short alias:
fts --help
```

## CLI Commands

### `fts models` — List discovered models
```bash
fts models                    # Table view
fts models --json             # JSON output
```

### `fts train` — Start training
```bash
fts train /path/to/model /path/to/data.jsonl \
    --output ./output \
    --lr 8e-5 \
    --epochs 4 \
    --batch 2 \
    --lora-rank 64 \
    --system-prompt "You are a helpful assistant."
```

### `fts test` — Interactive model testing
```bash
fts test /path/to/model
fts test /path/to/model.gguf --max-tokens 256 --temperature 0.5
```

### `fts suite` — Run test suites
```bash
fts suite /path/to/model /path/to/suite.json
fts suite /path/to/model /path/to/suite.json --json
```

### `fts validate` — Validate training data
```bash
fts validate data.jsonl
fts validate file1.jsonl file2.json file3.csv
```

### `fts convert` — Convert data formats
```bash
fts convert data.jsonl json
fts convert data.csv jsonl --system-prompt "You are helpful."
```

### `fts webui` — Start WebUI server
```bash
fts webui --port 7860
fts webui --host 127.0.0.1 --reload
```

---

## RAG (Retrieval-Augmented Generation)

### `fts rag ingest` — Ingest documents
```bash
fts rag ingest /path/to/documents          # Ingest directory
fts rag ingest /path/to/file.pdf           # Ingest single file
fts rag ingest ./docs --chunk-size 256     # Custom chunk size
```

**Supported formats:** TXT, MD, PDF, DOCX, CSV, JSON, JSONL, PY, JS, TS, HTML, CSS

### `fts rag query` — Query RAG store
```bash
fts rag query "What projects has the company completed?"
fts rag query "terrain data" --top-k 3
fts rag query "budget info" --json
```

### `fts rag list` — List indexed documents
```bash
fts rag list
fts rag list --json
```

### `fts rag stats` — Store statistics
```bash
fts rag stats
```

### `fts rag remove` — Remove document
```bash
fts rag remove <document_id>
```

### `fts rag clear` — Clear store
```bash
fts rag clear --confirm
```

### `fts rag-test` — RAG-enhanced inference
```bash
fts rag-test /path/to/model "What is the project timeline?" --top-k 5
fts rag-test /path/to/model "budget details" --system-prompt "Answer based on documents."
```

---

## Model Comparison

### `fts compare` — Compare models/APIs
```bash
# Compare two local models
fts compare model1=/path/to/model1 model2=/path/to/model2 suite.json

# Compare local model vs API
fts compare local=/path/to/model api=https://api.openai.com/v1/chat/completions suite.json

# With options
fts compare m1=/path/to/model1 m2=/path/to/model2 suite.json \
    --max-tokens 256 \
    --temperature 0.3 \
    --runs 3 \
    --report comparison.txt \
    --json
```

### Test Suite Format
```json
[
    {
        "name": "identity_check",
        "messages": [{"role": "user", "content": "Who are you?"}],
        "expected_keywords": ["Krzysztof", "IT"],
        "forbidden_keywords": ["I don't know"],
        "category": "identity"
    }
]
```

---

## Features
- **Model Management**: Browse, load, manage models (safetensors + GGUF)
- **Training Data**: Upload, validate, organize, convert (JSONL/JSON/CSV/TXT)
- **Training Engine**: Configure LoRA/LR/epochs, start/stop, realtime SSE progress
- **Model Testing**: Interactive chat + automated test suites with scoring
- **RAG**: Ingest documents, query with context, manage vector store
- **Comparison**: Compare models/APIs, score results, generate reports
- **CLI**: Full command-line interface for scripting and automation

## Tech Stack
- **UV** — package manager with lock file (reproducible installs)
- **FastAPI** + **Jinja2** + **HTMX** + **Tailwind CSS** (CDN)
- **Unsloth** / **transformers** / **trl** — training
- **ChromaDB** — vector store (local, persistent)
- **sentence-transformers** — embeddings (all-MiniLM-L6-v2)
- **llama-cpp-python** — GGUF inference (optional)

## Project Structure
```
finetune-studio/
├── pyproject.toml
├── uv.lock
├── install.sh / .ps1 / .bat
├── run.sh / .bat
├── src/finetune_studio/
│   ├── cli.py              # CLI interface
│   ├── config.py
│   ├── models/             # Model management
│   ├── training/           # Training engine
│   ├── data/               # Data validation/conversion
│   ├── testing/            # Inference + test suites
│   ├── rag/                # RAG (ingest, store, query, manager)
│   ├── compare/            # Model comparison (engine, scorer, reporter)
│   └── webui/              # FastAPI app + templates
└── tests/
```
