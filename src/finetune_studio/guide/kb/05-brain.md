---
id: brain
title: Smart Brain page
page: brain
keywords: smart brain rag retrieval generation answer llm vectors vector embeddings layers pipeline index search embedder passages chunk rerank hybrid keyword semantic bm25 fusion build quick index citation ground chat stale rebuild export bundle standalone server answer from files no training
---
## Purpose
The main path of a project, step 2. The Smart Brain answers from the project's files: it finds the passages that answer a question and the model quotes them, so answers cite their source and new or edited files only need a rebuild. Minutes, not hours; model weights never change. It reads the same parse training uses (Files & Knowledge), including Vision-reader readings and hand edits. Methods: semantic search (embeddings) + keyword search (BM25), combined by rank fusion, then reranking (cross-encoder); an answer gate declines when nothing found covers the question. Often called RAG.

## Workflow
1. Add files: drop them on this page (**＋ Add files**) or use Files & Knowledge; each readable file is parsed on arrival.
2. Click **Build Smart Brain** (**Update Smart Brain** after files change): it parses any file that is not parsed yet and builds the index from the files' passages, the same ones mining reads (first run downloads the embedder, ~2.2 GB).
While it builds, the bar shows the passages embedded, the time taken and the time left; **Stop build** ends it and keeps the old index. A large embedder (e5-large, bge-large, Qwen3-Embedding) on the CPU can take many minutes.
3. The panel at the top says the state in words: *Ready*, *Needs an update* (files changed since the build, new files, removed files), or what is still missing.
4. **Test a search** (under Manage): ask a fact question per document type; the top hit must be the right passage and show its source filename and score.
5. **Ask your documents** to chat with sources; follow-up questions keep the conversation, and **New conversation** clears it. Each answer lists the files it used: open one to read the passages. With no chat model loaded, load one in the same panel. When a question reaches its answer through another item ("who manages the site where pump X is installed?": the first search finds the pump's site, never that site's manager), the model may ask for one follow-up search before answering; the chat shows it as *Also searched*. On the Quillmoor benchmark this lifted two-step questions from 10 % to 84 % with the API model and the default reranker.
6. Build it before exporting a dataset if you want retrieved-context rows in the training data.
7. **Studio to studio**: *Build the package* writes an encrypted `.ftsrag` bundle (passphrase shown once); the other studio opens **Import an index from another studio**, picks the file and types the passphrase. The imported index says where it came from; a rebuild there indexes that project's own files instead. CLI: `fts brain bundle` / `fts brain import`.
8. **Standalone server export**: pick the search layers it ships, each a separate switch. **Meaning search** (vectors: finds passages that say the same in other words; needs the embedder inside via *Include models*, or an embeddings API where it runs), **Word search** (BM25 keyword index: exact names, codes, numbers; no model), **Rank fusion** (RRF; automatic when both searches are on), **Reranker** (a cross-encoder rescoring pass; needs *Include models*) and **Answer generation**, the "G" in RAG (the server also writes the answer with any OpenAI-compatible chat model: endpoint and model name here, the API key only where it runs). The **Pipeline** line shows how a question travels through the package; the package ships only the files its layers use. The same switches exist as `fts brain export --no-semantic/--no-keyword/--no-reranker/--answer`.

## Key controls
- `#brain-hero` — the current state and the one next action.
- `#brain-lint` — Knowledge check: errors (unreadable files, real keys) and items to review (almost no text, garbled, duplicates, out-of-date versions, documents that disagree on a value, possible passwords, index out of date). Review items can be ignored; the check needs no model. Conflict detection needs a built index.
- `#brain-identity` — who it speaks for: name, about you or your company, personality and how to answer. Used in every answer and in Testing and packed into the export (MCP `rag_profile`, HTTP `/profile`). Facts still come only from it and the files; no rebuild needed.
- `#quick-index-btn` — Build / Update Smart Brain: one-click parse + build.
- `#brain-add-files-btn` — add files without leaving the page (`#brain-upload-input`).
- `#chat-new-btn` — start a new conversation.
- `#chat-model-select` — pick a local model to answer with when none is loaded.
- `#b-embedder` — embedding model (multilingual-e5-large default; bge-large-en for English-only; all-MiniLM-L6-v2 is small and weaker).
- `#build-btn` (under **Build options**) — build; `#rebuild-btn` wipes and rebuilds from scratch.
- `#build-stop-btn` — stop a running build; it ends after the current batch and the old index stays as it was.
- `#reembed-btn` — after changing the embedder: keep the passages, replace only their vectors (quicker than a rebuild; needs an index).
- `#b-fact-cards` — also index approved Q&A pairs as fact cards (off by default).
- `#s-reranker` — the reranker model. A new index gets **bge-reranker-v2-m3** on a machine with a GPU (multilingual, most accurate: on the Quillmoor benchmark it found the right passage in the top 5 for 89 % of questions against 79 % for MiniLM, at about 0.2 s per search) and **ms-marco-MiniLM-L-6-v2** without one (small and fast on a CPU, English). Changing it applies to the next search; no rebuild needed.
- `#s-rerank-enabled` — reranking on/off.
- `#s-hybrid-enabled` — keyword search + rank fusion on/off (off = semantic search only).
- `#s-rerank-top-n` — how many candidates to rerank (default 50).
- `#q-text` — Test a search box.
- `#brain-chat` — grounded chat section.
- `#chat-answer-with` — who writes the chat's answer: **the loaded chat model** (top-bar pill) or **the AI assistant**: the API connection of its *Smart Brain chat* job (Settings → AI assistant), so grounded chat works without loading a local model, e.g. to compare the local answer against an outside provider or on a machine without a GPU. The default is always the local model; the AI assistant is never picked for you, even when no model is loaded (the page then says: load a model in Inference, or choose the AI assistant). Choosing it sends your question and the retrieved passages to that provider. On the Quillmoor benchmark the API model with follow-up search scored 96.1 %.
- `#chat-follow-up` — lets the model ask for one follow-up search before answering (on by default). Measured gain with models that follow the instruction (API model 88.4 → 96.1 %, Qwen 3.5 4B 88.8 → 91.1 %); none with Llama 3.2 3B or Gemma 3 4B, and Gemma abstained less, so turn it off for small models that over-use it.
- `#brain-import-panel` — import an encrypted `.ftsrag` bundle from another studio (works in a new project with no index).
- `#mcp-layers` — standalone export: the search layers, one switch each.
- `#mcp-sem` — export meaning search (vectors).
- `#mcp-kw` — export word search (BM25).
- `#mcp-rerank` — export the reranker (needs Include models).
- `#mcp-answer` — answer generation (the "G" in RAG) with an LLM endpoint.
- `#mcp-pipeline` — how a question travels through the package.

## What to check
The top panel says *Ready*; documents and passages counters after build; the search test returns the correct passage with a source filename. The exported bundle (.ftsrag) and the standalone server package are AES-256-GCM encrypted by default.

## Common mistakes
Changing the embedder without a rebuild; editing a file and not updating (the panel shows *Needs an update*); using the Smart Brain for facts the model should recall with no index attached; using training where citations are required.
