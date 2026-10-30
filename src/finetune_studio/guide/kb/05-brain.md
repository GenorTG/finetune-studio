---
id: brain
title: Smart Brain page
page: brain
keywords: smart brain rag retrieval generation answer llm vectors vector embeddings layers pipeline index search embedder passages chunk rerank hybrid keyword semantic bm25 fusion build quick index citation ground chat stale rebuild export bundle standalone server answer from files no training
---
## Purpose
Step 2 without training. The Smart Brain answers from the project's files: it finds the passages that answer a question and the model quotes them, so answers cite their source and new or edited files only need a rebuild. Minutes, not hours; model weights never change. It reads the same parse training uses (Files & Knowledge), including Vision-reader readings and hand edits. Methods: semantic search (embeddings) + keyword search (BM25), combined by rank fusion, then reranking (cross-encoder); an answer gate declines when nothing found covers the question. Often called RAG.

## Workflow
1. Upload in Files & Knowledge first; each readable file is parsed on arrival.
2. Click **⚡ Quick index**: it parses any file that is not parsed yet and builds the index from the files' passages, the same ones mining reads (first run downloads the embedder, ~2.2 GB).
While it builds, the bar shows the passages embedded, the time taken and the time left; **Stop build** ends it and keeps the old index. A large embedder (e5-large, bge-large, Qwen3-Embedding) on the CPU can take many minutes.
3. **Index status** says *up to date* or *rebuild needed* (files changed since the build, new files, removed files).
4. **Search test**: ask a fact question per document type; the top hit must be the right passage and show its source filename and score.
5. **Ask the model (grounded)** to chat with sources. When a question reaches its answer through another item ("who manages the site where pump X is installed?": the first search finds the pump's site, never that site's manager), the model may ask for one follow-up search before answering; the chat shows it as *Also searched*. On the Quillmoor benchmark this lifted two-step questions from 10 % to 46 % with the API model.
6. Build it before exporting a dataset if you want retrieved-context rows in the training data.
7. **Studio to studio**: *Download / export the index* writes an encrypted `.ftsrag` bundle (passphrase shown once); the other studio opens **Import an index from another studio**, picks the file and types the passphrase. The imported index says where it came from; a rebuild there indexes that project's own files instead. CLI: `fts brain bundle` / `fts brain import`.
8. **Standalone server export**: pick the search layers it ships, each a separate switch. **Meaning search** (vectors: finds passages that say the same in other words; needs the embedder inside via *Include models*, or an embeddings API where it runs), **Word search** (BM25 keyword index: exact names, codes, numbers; no model), **Rank fusion** (RRF; automatic when both searches are on), **Reranker** (a cross-encoder rescoring pass; needs *Include models*) and **Answer generation**, the "G" in RAG (the server also writes the answer with any OpenAI-compatible chat model: endpoint and model name here, the API key only where it runs). The **Pipeline** line shows how a question travels through the package; the package ships only the files its layers use. The same switches exist as `fts brain export --no-semantic/--no-keyword/--no-reranker/--answer`.

## Key controls
- `#quick-index-btn` — one-click parse + build.
- `#b-embedder` — embedding model (multilingual-e5-large default; bge-large-en for English-only; all-MiniLM-L6-v2 is small and weaker).
- `#build-btn` — build; `#rebuild-btn` wipes and rebuilds from scratch.
- `#build-stop-btn` — stop a running build; it ends after the current batch and the old index stays as it was.
- `#reembed-btn` — after changing the embedder: keep the passages, replace only their vectors (quicker than a rebuild; needs an index).
- `#b-fact-cards` — also index approved Q&A pairs as fact cards (off by default).
- `#s-reranker` — the reranker model. A new index gets **bge-reranker-v2-m3** on a machine with a GPU (multilingual, most accurate: on the Quillmoor benchmark it found the right passage in the top 5 for 89 % of questions against 79 % for MiniLM, at about 0.2 s per search) and **ms-marco-MiniLM-L-6-v2** without one (small and fast on a CPU, English). Changing it applies to the next search; no rebuild needed.
- `#s-rerank-enabled` — reranking on/off.
- `#s-hybrid-enabled` — keyword search + rank fusion on/off (off = semantic search only).
- `#s-rerank-top-n` — how many candidates to rerank (default 50).
- `#q-text` — search test box.
- `#brain-chat` — grounded chat section.
- `#brain-import-panel` — import an encrypted `.ftsrag` bundle from another studio (works in a new project with no index).
- `#mcp-layers` — standalone export: the search layers, one switch each.
- `#mcp-sem` — export meaning search (vectors).
- `#mcp-kw` — export word search (BM25).
- `#mcp-rerank` — export the reranker (needs Include models).
- `#mcp-answer` — answer generation (the "G" in RAG) with an LLM endpoint.
- `#mcp-pipeline` — how a question travels through the package.

## What to check
Index status *up to date*; documents/chunks counters after build; the search test returns the correct passage with a source filename. The exported bundle (.ftsrag) and the standalone server package are AES-256-GCM encrypted by default.

## Common mistakes
Changing the embedder without a rebuild; editing a file and not rebuilding (status shows *rebuild needed*); using the Smart Brain for facts the model should recall with no index attached; using training where citations are required.
