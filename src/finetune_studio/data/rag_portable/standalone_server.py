#!/usr/bin/env python3
"""Self-contained RAG corpus server — runs without Finetune Studio installed.

Searches a corpus directory produced by Finetune Studio:
    manifest.json  chunks.jsonl  vectors.npy  vectors.idx.json  bm25.json

Modes
    --http [PORT]   REST API:   GET /health   GET /search?q=...&top_k=5
    --mcp           MCP stdio server (JSON-RPC 2.0 over stdin/stdout).
                    Exposes one tool: rag_search(query, top_k).

Query embedding (semantic search) uses any OpenAI-compatible
/v1/embeddings endpoint — LM Studio, Ollama, OpenAI, vLLM, ...:
    RAG_EMBED_BASE_URL   e.g. http://localhost:1234/v1  (unset = keyword-only)
    RAG_EMBED_MODEL      default: the model the corpus was built with
    RAG_EMBED_API_KEY    optional bearer token

With no endpoint configured the server still works: keyword (BM25) search.

Dependencies: numpy only (install.sh sets up a venv with it).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[\-'][A-Za-z0-9]+)*|[\u00A0-\uFFFF]+", re.UNICODE)
PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "fts-rag-mcp"
SERVER_VERSION = "1.0.0"


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class Corpus:
    """Loads the corpus files once; scores queries with BM25 + dense + RRF."""

    def __init__(self, corpus_dir: Path = CORPUS_DIR):
        self.dir = Path(corpus_dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        self.chunks = [
            json.loads(line)
            for line in (self.dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        with open(self.dir / "bm25.json", encoding="utf-8") as f:
            self.bm25 = json.load(f)
        self.vectors = np.load(self.dir / "vectors.npy").astype(np.float32)
        self.idx_map = json.loads((self.dir / "vectors.idx.json").read_text(encoding="utf-8"))
        self.chunk_ids = [c["id"] for c in self.chunks]
        rs = self.manifest.get("rag_settings") or {}
        self.hybrid = bool(rs.get("hybrid_enabled", True))
        self.rrf_k = int(rs.get("rrf_k", 60))
        emb = (self.manifest.get("embedding_model") or {}).get("name") or ""
        self.embed_model = os.environ.get("RAG_EMBED_MODEL") or emb.split(":")[-1] or emb
        self.embed_base = (os.environ.get("RAG_EMBED_BASE_URL") or "").rstrip("/")

    def bm25_scores(self, query: str) -> np.ndarray:
        n = self.bm25.get("doc_count", len(self.chunks))
        scores = np.zeros(max(n, len(self.chunks)), dtype=np.float32)
        k1 = self.bm25.get("k1", 1.5)
        b = self.bm25.get("b", 0.75)
        avgdl = self.bm25.get("avgdl") or 1.0
        doc_lens = self.bm25.get("doc_lens") or [1] * n
        terms = self.bm25.get("terms") or {}
        df = self.bm25.get("df") or {}
        for term in set(tokenize(query)):
            postings = terms.get(term)
            if not postings:
                continue
            d = df.get(term, len(postings))
            idf = math.log(1 + (n - d + 0.5) / (d + 0.5))
            for doc_id, tf in postings:
                dl = doc_lens[doc_id] if doc_id < len(doc_lens) else avgdl
                scores[doc_id] += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
        return scores

    def embed_query(self, query: str) -> np.ndarray | None:
        """Embed the query via an OpenAI-compatible endpoint; None if unavailable."""
        if not self.embed_base:
            return None
        body = json.dumps({"input": [query], "model": self.embed_model}).encode()
        req = urllib.request.Request(
            f"{self.embed_base}/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        key = os.environ.get("RAG_EMBED_API_KEY")
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            vec = np.asarray(data["data"][0]["embedding"], dtype=np.float32)
        except (urllib.error.URLError, KeyError, ValueError, TimeoutError) as e:
            print(f"[warn] embedder unavailable ({e}); falling back to keyword-only", file=sys.stderr)
            return None
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else None

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query = (query or "").strip()
        if not query:
            return []
        rankings: list[list[str]] = []
        dense_scores = None
        q = self.embed_query(query)
        if q is not None and self.vectors.ndim == 2 and self.vectors.shape[0] == len(self.chunks):
            if q.shape[0] == self.vectors.shape[1]:
                dense_scores = self.vectors @ q
                rankings.append([self.chunk_ids[i] for i in np.argsort(-dense_scores)])
            else:
                print(f"[warn] query dim {q.shape[0]} != corpus dim {self.vectors.shape[1]}; "
                      "skipping semantic search", file=sys.stderr)
        bm25 = self.bm25_scores(query)
        rankings.append([self.chunk_ids[i] for i in np.argsort(-bm25)])

        rrf: dict[str, float] = {}
        for ranking in rankings:
            for rank, cid in enumerate(ranking, start=1):
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        ordered = sorted(rrf.items(), key=lambda kv: -kv[1])[: max(1, top_k)]

        hits = []
        for rank, (cid, score) in enumerate(ordered, start=1):
            i = self.idx_map.get(cid)
            if i is None or i >= len(self.chunks):
                continue
            c = self.chunks[i]
            hit = {
                "rank": rank,
                "chunk_id": cid,
                "score": round(float(score), 6),
                "text": c.get("text", ""),
                "filename": c.get("filename") or c.get("source") or "",
                "document_id": c.get("document_id", ""),
                "chunk_index": c.get("chunk_index", 0),
            }
            if dense_scores is not None:
                hit["dense_score"] = round(float(dense_scores[i]), 6)
            hit["bm25_score"] = round(float(bm25[i]), 6)
            hits.append(hit)
        return hits

    def info(self) -> dict:
        return {
            "corpus": self.manifest.get("name", ""),
            "chunks": len(self.chunks),
            "documents": self.manifest.get("documents", 0),
            "embed_model": self.embed_model or "(none)",
            "mode": "hybrid" if self.embed_base else "keyword-only",
            "embed_endpoint": self.embed_base or "(not set)",
        }


# ── HTTP mode ─────────────────────────────────────────────────────────


def serve_http(corpus: Corpus, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def _json(self, obj, code=200):
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            from urllib.parse import parse_qs, urlparse

            u = urlparse(self.path)
            if u.path == "/health":
                self._json({"ok": True, **corpus.info()})
            elif u.path == "/search":
                qs = parse_qs(u.query)
                q = (qs.get("q") or [""])[0]
                top_k = int((qs.get("top_k") or ["5"])[0])
                self._json({"query": q, "hits": corpus.search(q, top_k)})
            else:
                self._json({"error": "not found — try /health or /search?q=..."}, 404)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                hits = corpus.search(str(body.get("query", "")), int(body.get("top_k", 5)))
                self._json({"hits": hits})
            except (ValueError, json.JSONDecodeError) as e:
                self._json({"error": str(e)}, 400)

        def log_message(self, fmt, *args):
            print(f"[http] {fmt % args}", file=sys.stderr)

    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[rag-server] {corpus.info()} — listening on http://0.0.0.0:{port}/search?q=...",
          file=sys.stderr)
    srv.serve_forever()


# ── MCP stdio mode ────────────────────────────────────────────────────

_TOOLS = [{
    "name": "rag_search",
    "description": ("Search this knowledge corpus (your project's documents, "
                    "split into chunks and indexed). Returns the most relevant "
                    "passages with their source file names."),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look up."},
            "top_k": {"type": "integer", "description": "How many passages (default 5).",
                      "default": 5},
        },
        "required": ["query"],
    },
}, {
    "name": "rag_info",
    "description": "Corpus stats: name, chunk/document counts, search mode.",
    "inputSchema": {"type": "object", "properties": {}},
}]


def mcp_responder(corpus: Corpus):
    def respond(msg: dict) -> dict | None:
        method = msg.get("method")
        mid = msg.get("id")
        if method == "initialize":
            return {"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            }}
        if method and method.startswith("notifications/"):
            return None
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": mid, "result": {"tools": _TOOLS}}
        if method == "tools/call":
            params = msg.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            try:
                if name == "rag_search":
                    out = corpus.search(str(args.get("query", "")),
                                        int(args.get("top_k", 5)))
                elif name == "rag_info":
                    out = corpus.info()
                else:
                    return {"jsonrpc": "2.0", "id": mid,
                            "error": {"code": -32602, "message": f"unknown tool {name!r}"}}
                return {"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text",
                                 "text": json.dumps(out, ensure_ascii=False)}],
                    "isError": False,
                }}
            except Exception as e:  # noqa: BLE001 — MCP must not die on one call
                return {"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": f"error: {e}"}],
                    "isError": True,
                }}
        if mid is not None:
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32601, "message": f"method not found: {method}"}}
        return None
    return respond


def serve_mcp(corpus: Corpus) -> None:
    respond = mcp_responder(corpus)
    print(f"[rag-mcp] {corpus.info()} — stdio ready", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        out = respond(msg)
        if out is not None:
            sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
            sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Standalone Finetune-Studio RAG corpus server")
    ap.add_argument("--corpus", default=str(CORPUS_DIR), help="corpus directory")
    ap.add_argument("--http", nargs="?", const=8899, type=int, default=None,
                    help="serve REST API on PORT (default 8899)")
    ap.add_argument("--mcp", action="store_true", help="serve MCP over stdio")
    ap.add_argument("--query", help="run one search and print JSON, then exit")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args(argv)

    try:
        corpus = Corpus(Path(args.corpus))
    except (OSError, KeyError, ValueError) as e:
        print(f"cannot load corpus at {args.corpus}: {e}", file=sys.stderr)
        return 2

    if args.query:
        print(json.dumps(corpus.search(args.query, args.top_k), ensure_ascii=False, indent=2))
        return 0
    if args.http is not None:
        serve_http(corpus, args.http)
        return 0
    serve_mcp(corpus)
    return 0


if __name__ == "__main__":
    sys.exit(main())
