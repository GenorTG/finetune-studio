# Finetune Studio

Full-featured model training app with WebUI + CLI + RAG + Model Comparison.

GPU-aware install: auto-detects NVIDIA/AMD/Intel, installs CUDA wheels, GGUF included by default.

## Requirements

- **Python 3.12+** (auto-detected)
- **CUDA GPU** (RTX 3090 recommended, GTX 1070+ works)
- **UV** package manager (installed automatically)

## Install

```bash
cd finetune-studio
bash install.sh

# Flags:
bash install.sh --cpu        # force CPU-only
bash install.sh --no-gguf    # skip llama-cpp-python
bash install.sh --check      # verify install
```

## Run

```bash
bash run.sh          # → http://localhost:7860
```

### CLI

```bash
fts --help
fts models                       # list models
fts train /path/to/model data.jsonl --output ./out
fts test /path/to/model
fts suite /path/to/model suite.json
fts validate data.jsonl
fts convert data.csv jsonl
fts webui --port 7860
```

### RAG

```bash
fts rag ingest ./docs            # ingest documents
fts rag query "What is X?"      # query
fts rag list                     # list indexed docs
fts rag stats                    # store stats
```

### Model Comparison

```bash
fts compare m1=/path/to/model1 m2=/path/to/model2 suite.json
fts compare local=/path/to/model api=https://api.openai.com/v1/chat/completions suite.json
```

## Features

- **Model Management**: Browse, load, manage models (safetensors + GGUF)
- **Training Data**: Upload, validate, organize, convert (JSONL/JSON/CSV/TXT)
- **Training Engine**: Configure LoRA/LR/epochs, start/stop, realtime SSE progress
- **Model Testing**: Interactive chat + automated test suites with scoring
- **RAG**: Ingest documents, query with context, manage vector store
- **Comparison**: Compare models/APIs, score results, generate reports
- **CLI**: Full command-line interface for scripting and automation

## Tech Stack

| Component | Project | License |
|-----------|---------|---------|
| Web framework | [FastAPI](https://github.com/tiangolo/fastapi) | MIT |
| Template engine | [Jinja2](https://github.com/pallets/jinja) | BSD-3 |
| ASGI server | [Uvicorn](https://github.com/encode/uvicorn) | BSD-3 |
| Training | [PyTorch](https://github.com/pytorch/pytorch) | BSD-3 |
| Training | [TRL](https://github.com/huggingface/trl) | Apache-2.0 |
| Training | [PEFT](https://github.com/huggingface/peft) | Apache-2.0 |
| Training | [Accelerate](https://github.com/huggingface/accelerate) | Apache-2.0 |
| Model hub | [Hugging Face Hub](https://github.com/huggingface/huggingface_hub) | Apache-2.0 |
| Tokenizers | [Transformers](https://github.com/huggingface/transformers) | Apache-2.0 |
| Vector store | [ChromaDB](https://github.com/chroma-core/chroma) | Apache-2.0 |
| Embeddings | [sentence-transformers](https://github.com/UKPLab/sentence-transformers) | Apache-2.0 |
| GGUF inference | [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) | MIT |
| CUDA wheels | [abetlen CUDA builds](https://abetlen.github.io/llama-cpp-python/whl/) | MIT |
| Data validation | [Pydantic](https://github.com/pydantic/pydantic) | MIT |
| Data serialization | [Datasets](https://github.com/huggingface/datasets) | Apache-2.0 |
| Model format | [SafeTensors](https://github.com/huggingface/safetensors) | Apache-2.0 |
| Rich output | [Rich](https://github.com/Textualize/rich) | MIT |
| Async SQLite | [aiosqlite](https://github.com/omnilib/aiosqlite) | MIT |

## Project Structure

```
finetune-studio/
├── pyproject.toml
├── install.sh / .ps1 / .bat
├── run.sh / .bat
├── src/finetune_studio/
│   ├── cli.py              # CLI interface
│   ├── config.py
│   ├── models/             # Model management
│   ├── training/           # Training engine
│   ├── data/               # Data validation/conversion
│   ├── testing/            # Inference + test suites
│   ├── rag/                # RAG (ingest, store, query)
│   ├── compare/            # Model comparison
│   └── webui/              # FastAPI app + templates
└── tests/
```

## License

MIT
