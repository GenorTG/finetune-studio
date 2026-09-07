"""`fts rag-test` — one-shot Q&A with RAG context."""
from __future__ import annotations

import json
import os
import sys


def cmd_rag_test(args) -> None:
    from finetune_studio.rag.query import RAGConfig, RAGQuery
    from finetune_studio.rag.store import VectorStore
    from finetune_studio.testing.inference import InferenceEngine

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    engine = InferenceEngine()
    print(f"Loading {args.model}...")
    engine.load(args.model)

    store = VectorStore(args.store)
    rag = RAGQuery(store, RAGConfig(top_k=args.top_k))

    messages = rag.augment_prompt(args.question, args.system_prompt, args.top_k)
    response = engine.generate(messages, max_tokens=args.max_tokens)

    if args.json:
        sources = rag.retrieve(args.question, args.top_k)
        print(json.dumps({
            "response": response,
            "sources": [{"text": r.text[:200], "score": r.score, "source": r.metadata.get("source")} for r in sources],
        }, indent=2))
    else:
        print(f"\nAI: {response}")
        sources = rag.retrieve(args.question, args.top_k)
        if sources:
            print(f"\n--- Sources ({len(sources)} chunks) ---")
            for r in sources:
                print(f"  [{r.score:.3f}] {r.metadata.get('source', 'unknown')}")

    engine.unload()
