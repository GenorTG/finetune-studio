# RAG + Model Comparison — Architecture

## Real Estate Use Case
- **RAG**: Dynamic document queries (projects, docs, terrain data)
- **Fine-tuned model**: Baseline persona, consistent style, domain knowledge
- **Hybrid**: RAG for current data, model for personality/style
- **Retrain**: Only when persona changes or quality degrades (not for every doc update)

## New Modules

### `rag/` — Retrieval-Augmented Generation
- `ingest.py` — document ingestion, chunking, embedding (sentence-transformers)
- `store.py` — vector store management (ChromaDB, local, no server)
- `query.py` — retrieval + context injection into prompts
- `manager.py` — add/remove/update/index documents

### `compare/` — Model Comparison
- `engine.py` — run same prompts through multiple models/APIs
- `scorer.py` — score outputs (BLEU, ROUGE, keyword match, LLM-as-judge)
- `report.py` — generate comparison reports

## CLI Additions
```
fts rag ingest <dir>         — ingest documents into RAG store
fts rag query <question>     — query RAG
fts rag list                 — list indexed documents
fts rag remove <doc_id>      — remove document from index
fts compare <model1> <model2> <suite>  — compare two models
fts compare <model> <api_url> <suite>  — compare model vs API
```

## RAG Stack (local-first)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, local, fast)
- **Vector Store**: ChromaDB (embedded, no server, persistent)
- **Chunking**: 512 tokens, 50 overlap
- **Retrieval**: Top-k similarity, optional MMR for diversity
- **Context**: Inject top-k chunks into system prompt before inference

## Comparison Scoring
- **Keyword match**: expected/forbidden keywords
- **BLEU/ROUGE**: n-gram overlap with reference answers
- **Length penalty**: too short or too long answers
- **LLM-as-judge**: use a larger model to score relevance (optional)
