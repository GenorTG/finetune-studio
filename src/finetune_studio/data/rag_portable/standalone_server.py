#!/usr/bin/env python3
"""Self-contained RAG corpus server — runs without Finetune Studio installed.

Searches a corpus produced by Finetune Studio.  Two on-disk layouts:

    corpus/corpus.enc   ENCRYPTED container (default; format v2, see
                        rag_container.py).  Needs the passphrase at launch:
                        --keyfile PATH | RAG_PASSPHRASE | RAG_KEYFILE | prompt.
                        Decrypted IN MEMORY ONLY — no temp files, no caches.
    corpus/manifest.json chunks.jsonl vectors.npy vectors.idx.json bm25.json
                        legacy / --no-encrypt plaintext layout.

Modes
    --http [PORT]   REST API:   GET /health   GET /search?q=...&top_k=5
                    Binds 127.0.0.1 by default.  A non-loopback --host
                    REQUIRES an auth token (Authorization: Bearer ...).
    --mcp           MCP stdio server (JSON-RPC 2.0 over stdin/stdout).
                    Exposes rag_search(query, top_k) and rag_info().
    --query TEXT    one search, JSON on stdout.
    --print-config  show the effective merged config (secrets redacted).
    --save-config   persist the given flags into rag.config.json.

Configuration precedence:  CLI flags > env (RAG_*) > rag.config.json > defaults.
    host/RAG_HOST  port/RAG_PORT  top_k/RAG_TOP_K  device/RAG_DEVICE
    reranker/RAG_RERANKER  auth_token/RAG_AUTH_TOKEN  embed_model/RAG_EMBED_MODEL
    embed_base_url/RAG_EMBED_BASE_URL  embed_api_key/RAG_EMBED_API_KEY
    config path: --config / RAG_CONFIG / <this dir>/rag.config.json

Embedding, best available source first:
    1. corpus/embedder/  — a bundled model (public weights, not encrypted)
    2. RAG_EMBED_BASE_URL — any OpenAI-compatible /v1/embeddings endpoint
    3. none of the above — keyword (BM25) search only

Dependencies: numpy always; cryptography for encrypted packages;
sentence-transformers only when the package bundles models.
"""
from __future__ import annotations

import argparse
import hmac
import io
import ipaddress
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
CORPUS_DIR = HERE / "corpus"
ENC_NAME = "corpus.enc"
CONFIG_NAME = "rag.config.json"
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[\-'][A-Za-z0-9]+)*|[\u00A0-\uFFFF]+", re.UNICODE)
PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "fts-rag-mcp"
SERVER_VERSION = "2.0.0"
MAX_BODY = 1 << 20
MAX_TOP_K = 100

DEFAULT_CONFIG: dict = {
    "host": "127.0.0.1", "port": 8899, "top_k": 5, "device": "auto",
    "reranker": True, "auth_token": "", "embed_model": "",
    "embed_base_url": "", "embed_api_key": "",
}
ENV_KEYS = {
    "host": "RAG_HOST", "port": "RAG_PORT", "top_k": "RAG_TOP_K",
    "device": "RAG_DEVICE", "reranker": "RAG_RERANKER",
    "auth_token": "RAG_AUTH_TOKEN", "embed_model": "RAG_EMBED_MODEL",
    "embed_base_url": "RAG_EMBED_BASE_URL", "embed_api_key": "RAG_EMBED_API_KEY",
}
SECRET_KEYS = frozenset({"auth_token", "embed_api_key"})


def _resolve_device(requested: str) -> str:
    """``auto`` = GPU first (CUDA/ROCm, XPU, MPS), CPU only when none is usable.

    Self-contained on purpose: this file ships without finetune_studio.
    An explicit GPU request that torch cannot serve falls back to ``auto``
    with a warning rather than crashing the server at first search.
    """
    req = (requested or "auto").strip().lower()
    try:
        import torch
    except Exception:  # noqa: BLE001 — no torch: sentence-transformers can't load anyway
        return "cpu"
    def probe() -> str:
        try:
            if torch.cuda.is_available():      # also true for ROCm builds
                return "cuda"
            xpu = getattr(torch, "xpu", None)
            if xpu is not None and xpu.is_available():
                return "xpu"
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
        except Exception as e:  # noqa: BLE001 — a broken backend must not take the server down
            print(f"[warn] GPU probe failed ({e}); using cpu", file=sys.stderr)
        return "cpu"
    if req in ("", "auto"):
        return probe()
    if req == "cpu":
        return "cpu"
    best = probe()
    if best == "cpu" or req.split(":")[0] != best:
        print(f"[warn] device {requested!r} unavailable; using {best!r}", file=sys.stderr)
        return best
    return req


class ConfigError(Exception):
    """Bad configuration / refusal to start (reported, exit code 2)."""


def _coerce(key: str, value: object) -> object:
    try:
        if key in ("port", "top_k"):
            return int(value)  # type: ignore[call-overload]
        if key == "reranker":
            if isinstance(value, str):
                low = value.strip().lower()
                if low in ("1", "true", "yes", "on"):
                    return True
                if low in ("0", "false", "no", "off", ""):
                    return False
                raise ValueError(value)
            return bool(value)
        return str(value).strip() if key != "auth_token" else str(value)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"invalid value for {key!r}: {value!r}") from e


def resolve_config(flags: Mapping[str, object], env: Mapping[str, str],
                   file_cfg: Mapping[str, object]) -> tuple[dict, dict]:
    """Merge flags > env > file > defaults. Returns (config, origin-per-key).

    ``flags`` values of ``None`` mean "not given".  Env values that are empty
    strings are treated as unset (so ``RAG_EMBED_BASE_URL=`` does not mask the
    config file).
    """
    cfg: dict = {}
    origin: dict = {}
    for key, default in DEFAULT_CONFIG.items():
        val, src = default, "default"
        if key in file_cfg and file_cfg[key] is not None:
            val, src = file_cfg[key], "file"
        ev = env.get(ENV_KEYS[key])
        if ev not in (None, ""):
            val, src = ev, "env"
        if flags.get(key) is not None:
            val, src = flags[key], "flag"
        cfg[key] = _coerce(key, val)
        origin[key] = src
    if not 1 <= cfg["port"] <= 65535:
        raise ConfigError(f"port out of range: {cfg['port']}")
    if not 1 <= cfg["top_k"] <= MAX_TOP_K:
        raise ConfigError(f"top_k must be 1..{MAX_TOP_K}")
    return cfg, origin


def load_config_file(path: str | None, env: Mapping[str, str]) -> tuple[dict, Path | None]:
    explicit = path or env.get("RAG_CONFIG") or ""
    p = Path(explicit) if explicit else HERE / CONFIG_NAME
    if not p.is_file():
        if explicit:
            raise ConfigError(f"config file not found: {p}")
        return {}, None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ConfigError(f"cannot read config {p}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"config {p} must be a JSON object")
    unknown = sorted(set(data) - set(DEFAULT_CONFIG))
    if unknown:
        print(f"[warn] ignoring unknown config keys: {unknown}", file=sys.stderr)
    return {k: v for k, v in data.items() if k in DEFAULT_CONFIG}, p


def redact(cfg: Mapping[str, object]) -> dict:
    return {k: ("***" if k in SECRET_KEYS and v else v) for k, v in cfg.items()}


def is_loopback(host: str) -> bool:
    if host.strip().lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip().strip("[]")).is_loopback
    except ValueError:
        return False


def check_bind_security(cfg: Mapping[str, object]) -> None:
    """Refuse to expose the corpus on the network without authentication."""
    if not is_loopback(str(cfg["host"])) and not cfg["auth_token"]:
        raise ConfigError(
            f"refusing to bind {cfg['host']!r} without an auth token: a "
            "non-loopback host exposes the corpus to the network. Set "
            "--auth-token / RAG_AUTH_TOKEN / auth_token in rag.config.json, "
            "or use --host 127.0.0.1.")


def tokens_equal(supplied: str, expected: str) -> bool:
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def read_passphrase(keyfile: str | None, env: Mapping[str, str]) -> str:
    """--keyfile > RAG_PASSPHRASE > RAG_KEYFILE > interactive prompt."""
    kf = keyfile or None
    if kf is None and env.get("RAG_PASSPHRASE"):
        return env["RAG_PASSPHRASE"]
    kf = kf or env.get("RAG_KEYFILE") or None
    if kf:
        try:
            text = Path(kf).read_text(encoding="utf-8")
        except OSError as e:
            raise ConfigError(f"cannot read keyfile {kf}: {e}") from e
        text = text.rstrip("\r\n")
        if not text:
            raise ConfigError(f"keyfile {kf} is empty")
        return text
    try:
        open("/dev/tty", "r+").close()
    except OSError:
        raise ConfigError(
            "this package is encrypted and no passphrase was provided: set "
            "RAG_PASSPHRASE, pass --keyfile PATH, or run from a terminal") from None
    import getpass
    pw = getpass.getpass("RAG package passphrase: ")
    if not pw:
        raise ConfigError("empty passphrase")
    return pw


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _load_container_module():
    try:
        import rag_container
    except ImportError:
        sys.path.insert(0, str(HERE))
        import rag_container
    return rag_container


class Corpus:
    """Loads the corpus once; scores queries with BM25 + dense + RRF.

    Encrypted packages are decrypted straight into memory (``bytes`` ->
    ``json.loads`` / ``np.load(BytesIO)``): nothing plaintext touches disk.
    """

    def __init__(self, corpus_dir: Path = CORPUS_DIR, cfg: Mapping | None = None,
                 passphrase: str | None = None):
        self.dir = Path(corpus_dir)
        self.cfg = dict(cfg) if cfg is not None else resolve_config({}, os.environ, {})[0]
        enc = self.dir / ENC_NAME
        self.encrypted = enc.is_file()
        self._container = None
        if self.encrypted:
            if passphrase is None:
                raise ConfigError("encrypted package: a passphrase is required")
            self._container = _load_container_module().ContainerReader(enc, passphrase)
            read = self._container.read
        else:
            def read(name: str) -> bytes:
                return (self.dir / name).read_bytes()
        self.manifest = json.loads(read("manifest.json").decode("utf-8"))
        self.chunks = [
            json.loads(line)
            for line in read("chunks.jsonl").decode("utf-8").splitlines()
            if line.strip()
        ]
        self.bm25 = json.loads(read("bm25.json").decode("utf-8"))
        self.vectors = np.load(io.BytesIO(read("vectors.npy")),
                               allow_pickle=False).astype(np.float32)
        self.idx_map = json.loads(read("vectors.idx.json").decode("utf-8"))
        self.chunk_ids = [c["id"] for c in self.chunks]
        rs = self.manifest.get("rag_settings") or {}
        self.hybrid = bool(rs.get("hybrid_enabled", True))
        self.rrf_k = int(rs.get("rrf_k", 60))
        emb = (self.manifest.get("embedding_model") or {}).get("name") or ""
        self.embed_model = self.cfg["embed_model"] or emb.split(":")[-1] or emb
        self.embed_base = str(self.cfg["embed_base_url"]).rstrip("/")
        self.device = _resolve_device(str(self.cfg["device"]))
        self.top_k = int(self.cfg["top_k"])
        self._local_embedder = None
        self._local_embedder_failed = False
        self._reranker = None
        self._reranker_failed = False
        self.rerank_enabled = bool(self.cfg["reranker"]) and bool(rs.get("rerank_enabled", False))
        self.rerank_top_n = int(rs.get("rerank_top_n", 50))

    def read_source(self, name: str) -> str:
        """Lazily decrypt one original document (``sources/<name>``)."""
        if self._container is not None:
            return self._container.read(f"sources/{name}").decode("utf-8")
        return (self.dir / "sources" / name).read_text(encoding="utf-8")

    @staticmethod
    def _looks_like_model(d: Path) -> bool:
        """A real HF snapshot dir, not a leftover placeholder."""
        if not d.is_dir():
            return False
        names = {p.name for p in d.iterdir()}
        return bool({"config.json", "sentence_bert_config.json"} & names
                    or {n for n in names if n.endswith((".safetensors", ".bin"))})

    def local_embedder(self):
        """SentenceTransformer loaded from corpus/embedder/, or None."""
        if self._local_embedder is not None or self._local_embedder_failed:
            return self._local_embedder
        d = self.dir / "embedder"
        if not self._looks_like_model(d):
            return None
        try:
            from sentence_transformers import SentenceTransformer
            self._local_embedder = SentenceTransformer(
                str(d), device=self.device)
            print(f"[rag-server] local embedder loaded from {d}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 — degrade, never crash the server
            self._local_embedder_failed = True
            print(f"[warn] bundled embedder failed ({e}); "
                  "falling back to API/keyword search", file=sys.stderr)
            return None
        return self._local_embedder

    def local_reranker(self):
        """CrossEncoder loaded from corpus/reranker/, or None."""
        if (self._reranker is not None or self._reranker_failed
                or not self.rerank_enabled):
            return self._reranker
        d = self.dir / "reranker"
        if not self._looks_like_model(d):
            return None
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(
                str(d), device=self.device)
            print(f"[rag-server] local reranker loaded from {d}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            self._reranker_failed = True
            print(f"[warn] bundled reranker failed ({e}); skipping rerank", file=sys.stderr)
            return None
        return self._reranker

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
        """Embed the query: bundled model → API endpoint → None (keyword-only)."""
        model = self.local_embedder()
        if model is not None:
            v = np.asarray(model.encode([query], normalize_embeddings=True),
                           dtype=np.float32)[0]
            return v
        if not self.embed_base:
            return None
        body = json.dumps({"input": [query], "model": self.embed_model}).encode()
        req = urllib.request.Request(
            f"{self.embed_base}/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        key = self.cfg["embed_api_key"]
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

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        query = (query or "").strip()
        if not query:
            return []
        top_k = self.top_k if top_k is None else min(max(1, int(top_k)), MAX_TOP_K)
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
                "rrf_score": round(float(score), 6),
                "text": c.get("text", ""),
                "filename": c.get("filename") or c.get("source") or "",
                "document_id": c.get("document_id", ""),
                "chunk_index": c.get("chunk_index", 0),
            }
            if dense_scores is not None:
                hit["dense_score"] = round(float(dense_scores[i]), 6)
            hit["bm25_score"] = round(float(bm25[i]), 6)
            hits.append(hit)

        reranker = self.local_reranker()
        if reranker is not None and len(hits) > 1:
            pool = hits[: self.rerank_top_n]
            try:
                ce = reranker.predict([(query, h["text"]) for h in pool])
                for h, s in zip(pool, ce):
                    h["ce_score"] = round(float(s), 6)
                pool.sort(key=lambda h: -h["ce_score"])
                hits = pool + hits[len(pool):]
                for rank, h in enumerate(hits, start=1):
                    h["rank"] = rank
            except Exception as e:  # noqa: BLE001
                print(f"[warn] rerank failed ({e}); keeping RRF order", file=sys.stderr)
        return hits

    def info(self) -> dict:
        if self.local_embedder() is not None:
            mode = "offline-semantic (bundled embedder)"
        elif self.embed_base:
            mode = "hybrid (embeddings API)" if self.hybrid else "semantic (embeddings API)"
        else:
            mode = "keyword-only"
        if self.local_reranker() is not None:
            mode += " + rerank"
        return {
            "corpus": self.manifest.get("name", ""),
            "chunks": len(self.chunks),
            "documents": self.manifest.get("documents", 0),
            "encrypted": self.encrypted,
            "embed_model": self.embed_model or "(none)",
            "mode": mode,
            "embed_endpoint": self.embed_base or "(not set)",
        }


# ── HTTP mode ─────────────────────────────────────────────────────────


def make_handler(corpus: Corpus, token: str = ""):
    """Request handler class. ``token`` non-empty => Bearer auth required."""

    class Handler(BaseHTTPRequestHandler):
        def _json(self, obj, code=200, extra=None):
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            """Bearer check (constant time). Without a token the server is
            loopback-only; reject foreign Host headers (DNS rebinding)."""
            if token:
                auth = self.headers.get("Authorization") or ""
                if auth[:7].lower() == "bearer " and tokens_equal(auth[7:].strip(), token):
                    return True
                self._json({"error": "unauthorized"}, 401,
                           {"WWW-Authenticate": 'Bearer realm="rag"'})
                return False
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0] \
                if not (self.headers.get("Host") or "").startswith("[") \
                else (self.headers.get("Host") or "").split("]")[0] + "]"
            if host and not is_loopback(host):
                self._json({"error": "forbidden host"}, 403)
                return False
            return True

        def do_GET(self):
            if not self._authorized():
                return
            u = urlparse(self.path)
            if u.path == "/health":
                self._json({"ok": True, **corpus.info()})
            elif u.path == "/search":
                qs = parse_qs(u.query)
                q = (qs.get("q") or [""])[0]
                try:
                    top_k = int(qs["top_k"][0]) if qs.get("top_k") else None
                    self._json({"query": q, "hits": corpus.search(q, top_k)})
                except ValueError as e:
                    self._json({"error": str(e)}, 400)
            else:
                self._json({"error": "not found — try /health or /search?q=..."}, 404)

        def do_POST(self):
            if not self._authorized():
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY:
                    self._json({"error": "request body too large"}, 413)
                    return
                body = json.loads(self.rfile.read(length) or b"{}")
                top_k = body.get("top_k")
                hits = corpus.search(str(body.get("query", "")),
                                     int(top_k) if top_k is not None else None)
                self._json({"hits": hits})
            except (ValueError, json.JSONDecodeError) as e:
                self._json({"error": str(e)}, 400)

        def log_request(self, code="-", size="-"):  # path only: never the query text
            print(f"[http] {self.command} {urlparse(self.path).path} {code}", file=sys.stderr)

        def log_message(self, fmt, *args):  # error lines may echo raw request text
            return

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True


class _Server6(_Server):
    import socket as _socket
    address_family = _socket.AF_INET6


def serve_http(corpus: Corpus, cfg: Mapping) -> None:
    check_bind_security(cfg)
    host, port = str(cfg["host"]).strip("[]"), int(cfg["port"])
    cls = _Server6 if ":" in host else _Server
    srv = cls((host, port), make_handler(corpus, str(cfg["auth_token"])))
    auth = "bearer-token auth ON" if cfg["auth_token"] else "no auth (loopback only)"
    print(f"[rag-server] {corpus.info()} — listening on http://{host}:{port}/search?q=... "
          f"({auth})", file=sys.stderr)
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
            "top_k": {"type": "integer", "description": "How many passages (server default, usually 5)."},
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
                                        int(args["top_k"]) if args.get("top_k") is not None else None)
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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Standalone Finetune-Studio RAG corpus server "
                    "(flags > env RAG_* > rag.config.json > defaults)")
    ap.add_argument("--corpus", default=str(CORPUS_DIR), help="corpus directory")
    ap.add_argument("--config", help="path to rag.config.json (or RAG_CONFIG)")
    ap.add_argument("--http", nargs="?", const=0, type=int, default=None, metavar="PORT",
                    help="serve REST API (optionally on PORT)")
    ap.add_argument("--mcp", action="store_true", help="serve MCP over stdio")
    ap.add_argument("--query", help="run one search and print JSON, then exit")
    ap.add_argument("--host", help="bind address (default 127.0.0.1)")
    ap.add_argument("--port", type=int, help="HTTP port (default 8899)")
    ap.add_argument("--top-k", dest="top_k", type=int, help="results per search (default 5)")
    ap.add_argument("--device", help="auto (GPU first, default) | cpu | cuda | cuda:N | xpu | mps for bundled models")
    ap.add_argument("--reranker", dest="reranker", action="store_true", default=None,
                    help="enable the bundled reranker")
    ap.add_argument("--no-reranker", dest="reranker", action="store_false",
                    help="disable the reranker")
    ap.add_argument("--auth-token", dest="auth_token",
                    help="bearer token for HTTP (prefer RAG_AUTH_TOKEN: argv is visible)")
    ap.add_argument("--embed-model", dest="embed_model")
    ap.add_argument("--embed-base-url", dest="embed_base_url")
    ap.add_argument("--keyfile", help="file holding the package passphrase")
    ap.add_argument("--print-config", action="store_true",
                    help="show the effective merged config (secrets redacted) and exit")
    ap.add_argument("--save-config", action="store_true",
                    help="write the given flags into rag.config.json and exit")
    return ap


_FLAG_KEYS = ("host", "port", "top_k", "device", "reranker", "auth_token",
              "embed_model", "embed_base_url")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    flags = {k: getattr(args, k) for k in _FLAG_KEYS}
    if args.http:  # --http PORT
        flags["port"] = args.http
    env = os.environ
    try:
        file_cfg, cfg_path = load_config_file(args.config, env)
        cfg, origin = resolve_config(flags, env, file_cfg)
        corpus_path = Path(args.corpus)
        encrypted = (corpus_path / ENC_NAME).is_file()

        if args.print_config:
            print(json.dumps({"config": redact(cfg), "origin": origin,
                              "config_file": str(cfg_path) if cfg_path else None,
                              "corpus": str(corpus_path), "encrypted": encrypted},
                             indent=2))
            return 0
        if args.save_config:
            target = cfg_path or (Path(args.config) if args.config else HERE / CONFIG_NAME)
            given = {k: v for k, v in flags.items()
                     if v is not None and k not in SECRET_KEYS}
            target.write_text(json.dumps({**file_cfg, **given}, indent=2) + "\n",
                              encoding="utf-8")
            print(f"saved {sorted(given)} to {target}", file=sys.stderr)
            return 0

        # Security gate first: no passphrase prompt / decrypt for a refused start.
        if args.http is not None:
            check_bind_security(cfg)
        passphrase = read_passphrase(args.keyfile, env) if encrypted else None
        corpus = Corpus(corpus_path, cfg, passphrase)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    except (OSError, KeyError, ValueError) as e:
        print(f"cannot load corpus at {args.corpus}: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        if type(e).__name__ in ("ContainerError", "WrongPassphraseOrTampered"):
            print(f"cannot open encrypted package: {e}", file=sys.stderr)
            return 3
        raise

    if args.query:
        print(json.dumps(corpus.search(args.query), ensure_ascii=False, indent=2))
        return 0
    if args.http is not None:
        serve_http(corpus, cfg)
        return 0
    serve_mcp(corpus)
    return 0


if __name__ == "__main__":
    sys.exit(main())
