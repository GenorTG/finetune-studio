# Portable Inference Server with RAG

## What You Need

### Hardware
- **GPU**: RTX 3090 24GB (recommended) or RTX 3080 16GB minimum
- **RAM**: 32GB+ (64GB for large models)
- **Storage**: 50GB+ (model + embeddings + docs)
- **OS**: Linux (Ubuntu 22.04+), Windows, macOS

### Software
- Python 3.12 or 3.13
- CUDA 12.x (for GPU inference)
- That's it — everything else is bundled

### Model
- GGUF format (recommended — single file, portable)
- Or safetensors (HuggingFace format)

### Documents for RAG
- PDF, DOCX, TXT, MD, CSV, JSON, code files
- Project documentation, terrain data, specs, etc.

## Portable Stack

### Option 1: Docker (Recommended for Production)
```bash
docker run -d \
  --gpus all \
  -v /path/to/models:/app/models \
  -v /path/to/rag-data:/app/rag_data \
  -v /path/to/config.yaml:/app/config.yaml \
  -p 8080:8080 \
  finetune-studio-server
```

### Option 2: Python (Development/Testing)
```bash
git clone https://github.com/.../inference-server
cd inference-server
pip install .
inference-server --config config.yaml
```

### Option 3: Standalone Binary (Future)
- PyInstaller/cx_Freeze bundle
- No Python required on target machine
- Single executable + model file

## Configuration (config.yaml)
```yaml
server:
  host: "0.0.0.0"
  port: 8080

model:
  path: "./models/model.gguf"
  n_gpu_layers: 99
  n_ctx: 8192

rag:
  enabled: true
  store_path: "./rag_data/store"
  embedding_model: "all-MiniLM-L6-v2"
  chunk_size: 512
  top_k: 5
  min_score: 0.3

inference:
  max_tokens: 1024
  temperature: 0.7
  top_p: 0.9
  repeat_penalty: 1.05

api:
  key: ""  # Optional API key
  cors: true
```

## API Endpoints

### OpenAI-Compatible
```
POST /v1/chat/completions
POST /v1/completions
GET  /v1/models
```

### RAG-Enhanced
```
POST /v1/rag/query          # Query with RAG context
POST /v1/rag/ingest         # Add documents to RAG
GET  /v1/rag/documents      # List indexed documents
DELETE /v1/rag/documents/{id}  # Remove document
```

### Management
```
GET  /health                # Health check
GET  /stats                 # Server statistics
POST /reload                # Reload model/RAG
```

## Deployment Flow

1. **Setup server** (one-time):
   ```bash
   # Clone and install
   git clone .../inference-server
   cd inference-server
   pip install .

   # Download model
   wget -O models/model.gguf https://huggingface.co/.../model.Q4_K_M.gguf

   # Add documents
   cp -r /path/to/docs/* rag_data/documents/

   # Configure
   cp config.example.yaml config.yaml
   # Edit config.yaml with model path, API key, etc.

   # Ingest documents into RAG
   inference-server --ingest rag_data/documents/
   ```

2. **Run server**:
   ```bash
   inference-server --config config.yaml
   # → http://localhost:8080
   ```

3. **Use API**:
   ```bash
   # Standard inference
   curl http://localhost:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "What projects?"}]}'

   # RAG-enhanced inference
   curl http://localhost:8080/v1/rag/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What projects?", "model": "model-id"}'
   ```

4. **Update documents** (no retraining):
   ```bash
   # Add new documents
   inference-server --ingest /path/to/new-docs/

   # Or via API
   curl -X POST http://localhost:8080/v1/rag/ingest \
     -F "file=@new_document.pdf"
   ```

## What Makes This Portable

1. **Single config file** — point to model + RAG dir
2. **Docker image** — run anywhere with GPU
3. **No external dependencies** — no cloud APIs, no internet needed
4. **Persistent RAG** — ChromaDB store survives restarts
5. **OpenAI-compatible** — drop-in for existing integrations
6. **Auto-ingestion** — ingest on startup or via API
7. **Health checks** — monitoring-ready

## Cost Estimate (Self-Hosted)

| Component | Cost |
|-----------|------|
| Server (RTX 3090) | ~15,000-20,000 PLN one-time |
| Electricity | ~200-400 PLN/month |
| Internet | Existing |
| Maintenance | Your time |
| **Total Year 1** | ~20,000-25,000 PLN |
| **Total Year 2+** | ~2,400-4,800 PLN/year |

vs. Cloud API:
| Provider | Cost (1M tokens/day) |
|----------|---------------------|
| OpenAI GPT-4 | ~3,000-5,000 PLN/month |
| Claude | ~2,000-4,000 PLN/month |
| **Break-even** | ~6-12 months |
