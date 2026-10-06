---
id: rag
title: RAG index page
page: rag
keywords: rag retrieval index search embedder chunk overlap rerank hybrid bm25 build quick index corpus citation ground chat export bundle
---
## Purpose
The no-training flow. Documents are chunked, embedded and indexed; at question time the model looks up relevant passages, so answers can quote the source and new files only need a rebuild. Minutes, not hours; model weights never change.

## Workflow
1. Upload on Files first.
2. Click **⚡ Quick index**: it parses any unparsed library files and builds the corpus (first run downloads the embedder, ~2.2 GB).
3. **Search test**: ask a fact question per document type; the top hit must be the right passage and show its source filename and score.
4. **Ask the model (grounded)** to chat over the corpus.
5. Do this before exporting a dataset if you want retrieved-context rows in the training data.

## Key controls
- `#quick-index-btn` — one-click parse + build.
- `#b-embedder` — embedding model (multilingual-e5-large default; bge-large-en for English-only; all-MiniLM-L6-v2 is small and weaker).
- `#b-chunk` — chunk size (default 400).
- `#b-overlap` — overlap between chunks (default 80).
- `#build-btn` — build; `#rebuild-btn` wipes and rebuilds (needed after changing chunk size, overlap or embedder).
- `#s-rerank-enabled` — cross-encoder rerank on/off.
- `#s-hybrid-enabled` — hybrid (embedding + BM25 keyword) retrieval on/off.
- `#s-rerank-top-n` — how many candidates to rerank (default 50).
- `#q-text` — search test box.
- `#rag-chat` — grounded chat section.

## What to check
Documents/chunks counters after build; the search test returns the correct passage with a source filename. The RAG export bundle is AES-256-GCM encrypted.

## Common mistakes
Changing chunk size or embedder without a rebuild; using RAG for facts you expect the model to recall without a corpus attached; using training where citations are required.
