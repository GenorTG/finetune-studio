---
id: brain
title: Smart Brain page
page: brain
keywords: smart brain rag retrieval index search embedder passages chunk rerank hybrid keyword semantic bm25 fusion build quick index citation ground chat stale rebuild export bundle standalone server answer from files no training
---
## Purpose
Step 2 without training. The Smart Brain answers from the project's files: it finds the passages that answer a question and the model quotes them, so answers cite their source and new or edited files only need a rebuild. Minutes, not hours; model weights never change. It reads the same parse training uses (Files & Knowledge), including Vision-reader readings and hand edits. Methods: semantic search (embeddings) + keyword search (BM25), combined by rank fusion, then reranking (cross-encoder); an answer gate declines when nothing found covers the question. Often called RAG.

## Workflow
1. Upload in Files & Knowledge first; each readable file is parsed on arrival.
2. Click **⚡ Quick index**: it parses any file that is not parsed yet and builds the index from the files' passages, the same ones mining reads (first run downloads the embedder, ~2.2 GB).
3. **Index status** says *up to date* or *rebuild needed* (files changed since the build, new files, removed files).
4. **Search test**: ask a fact question per document type; the top hit must be the right passage and show its source filename and score.
5. **Ask the model (grounded)** to chat with sources.
6. Build it before exporting a dataset if you want retrieved-context rows in the training data.

## Key controls
- `#quick-index-btn` — one-click parse + build.
- `#b-embedder` — embedding model (multilingual-e5-large default; bge-large-en for English-only; all-MiniLM-L6-v2 is small and weaker).
- `#build-btn` — build; `#rebuild-btn` wipes and rebuilds (needed after changing the embedder).
- `#b-fact-cards` — also index approved Q&A pairs as fact cards (off by default).
- `#s-rerank-enabled` — reranking on/off.
- `#s-hybrid-enabled` — keyword search + rank fusion on/off (off = semantic search only).
- `#s-rerank-top-n` — how many candidates to rerank (default 50).
- `#q-text` — search test box.
- `#brain-chat` — grounded chat section.

## What to check
Index status *up to date*; documents/chunks counters after build; the search test returns the correct passage with a source filename. The exported bundle (.ftsrag) and the standalone server package are AES-256-GCM encrypted by default.

## Common mistakes
Changing the embedder without a rebuild; editing a file and not rebuilding (status shows *rebuild needed*); using the Smart Brain for facts the model should recall with no index attached; using training where citations are required.
