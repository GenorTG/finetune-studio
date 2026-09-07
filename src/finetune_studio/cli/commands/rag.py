"""`fts rag {ingest,query,list,remove,stats,clear}` — RAG store operations."""
from __future__ import annotations

import json
import os
import sys


def cmd_rag(args) -> None:
    if args.rag_command is None:
        print("Usage: finetune-studio rag {ingest,query,list,remove,stats,clear}")
        sys.exit(1)

    from finetune_studio.rag.manager import RAGManager
    manager = RAGManager(args.store)

    if args.rag_command == "ingest":
        if os.path.isdir(args.path):
            result = manager.ingest_directory(args.path, args.chunk_size, args.overlap)
        else:
            result = manager.ingest_file(args.path, args.chunk_size, args.overlap)
        print(json.dumps(result, indent=2))

    elif args.rag_command == "query":
        results = manager.store.search(args.question, top_k=args.top_k)
        if args.json:
            print(json.dumps([{
                "text": r.text, "score": round(r.score, 3),
                "source": r.metadata.get("source", "unknown"),
                "document_id": r.document_id,
            } for r in results], indent=2))
        else:
            if not results:
                print("No results found.")
            for i, r in enumerate(results):
                print(f"\n--- Result {i+1} (score: {r.score:.3f}) ---")
                print(f"Source: {r.metadata.get('source', 'unknown')}")
                print(f"Doc: {r.document_id}")
                print(f"{r.text[:300]}{'...' if len(r.text) > 300 else ''}")

    elif args.rag_command == "list":
        docs = manager.list_documents()
        if args.json:
            print(json.dumps(docs, indent=2))
        else:
            if not docs:
                print("No documents in RAG store.")
            for doc in docs:
                print(f"  {doc['document_id']}: {doc['chunk_count']} chunks, sources: {doc['sources']}")

    elif args.rag_command == "remove":
        result = manager.remove_document(args.document_id)
        print(json.dumps(result, indent=2))

    elif args.rag_command == "stats":
        stats = manager.stats()
        print(json.dumps(stats, indent=2))

    elif args.rag_command == "clear":
        if not args.confirm:
            print("This will clear ALL RAG data. Use --confirm to proceed.")
            sys.exit(1)
        manager.clear()
        print("RAG store cleared.")
