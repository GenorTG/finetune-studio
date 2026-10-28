#!/usr/bin/env python3
"""Self-contained RAG corpus server — runs without Finetune Studio installed.

Searches a corpus produced by Finetune Studio.  Two on-disk layouts:

    corpus/corpus.enc   ENCRYPTED container (default; format v2, see
                        brain_container.py).  Needs the passphrase at launch:
                        --keyfile PATH | RAG_PASSPHRASE | RAG_KEYFILE | prompt.
                        Decrypted IN MEMORY ONLY — no temp files, no caches.
    corpus/manifest.json chunks.jsonl vectors.npy vectors.idx.json bm25.json
                        legacy / --no-encrypt plaintext layout.

Modes
    --http [PORT]   REST API:   GET /health   GET /search?q=...&top_k=5[&source_prefix=pl-]
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
    query_prefix/RAG_QUERY_PREFIX (prepended to the embedded query; Qwen3: "Instruct: ...\\nQuery:")
    dense_extra/RAG_DENSE_EXTRA (also return the N best passages by meaning alone)
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
SERVER_NAME = "fts-brain-mcp"
SERVER_VERSION = "2.0.0"
MAX_BODY = 1 << 20
MAX_TOP_K = 100

DEFAULT_CONFIG: dict = {
    "host": "127.0.0.1", "port": 8899, "top_k": 10, "device": "auto",
    "reranker": True, "auth_token": "", "embed_model": "",
    "embed_base_url": "", "embed_api_key": "", "query_prefix": "", "dense_extra": 0,
    "semantic": True, "keyword": True, "answer": False,
    "llm_base_url": "", "llm_model": "", "llm_api_key": "", "answer_max_tokens": 512,
    "llm_headers": {}, "llm_body": {},
}
ENV_KEYS = {
    "host": "RAG_HOST", "port": "RAG_PORT", "top_k": "RAG_TOP_K",
    "device": "RAG_DEVICE", "reranker": "RAG_RERANKER",
    "auth_token": "RAG_AUTH_TOKEN", "embed_model": "RAG_EMBED_MODEL",
    "embed_base_url": "RAG_EMBED_BASE_URL", "embed_api_key": "RAG_EMBED_API_KEY",
    "query_prefix": "RAG_QUERY_PREFIX", "dense_extra": "RAG_DENSE_EXTRA",
    "semantic": "RAG_SEMANTIC", "keyword": "RAG_KEYWORD", "answer": "RAG_ANSWER",
    "llm_base_url": "RAG_LLM_BASE_URL", "llm_model": "RAG_LLM_MODEL", "llm_api_key": "RAG_LLM_API_KEY",
    "answer_max_tokens": "RAG_ANSWER_MAX_TOKENS",
    "llm_headers": "RAG_LLM_HEADERS", "llm_body": "RAG_LLM_BODY",
}
SECRET_KEYS = frozenset({"auth_token", "embed_api_key", "llm_api_key"})
BOOL_KEYS = frozenset({"reranker", "semantic", "keyword", "answer"})
JSON_KEYS = frozenset({"llm_headers", "llm_body"})
USER_AGENT = "fts-brain-server/2.0"   # some API gateways (Cloudflare) refuse Python's default urllib agent
SESSION_ID = os.urandom(16).hex()     # "{session}" in an llm header (the OpenCode Go gateway wants one per client)

# The grounded prompt of the studio's Smart Brain chat (smart_brain/prompt.py; this file ships on its own, so the
# text is copied here and tests/test_brain_export_layers.py pins the two together).
DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer using ONLY the context below. "
    "If the answer isn't in the context, say so. Quote the source filename in [brackets] when relevant."
)
CONTEXT_MARKER = "CONTEXT:\n"
_BLOCK_SEP = "\n\n---\n\n"
ANSWER_CONTEXT_CHARS = 24000
_STOP = frozenset(["what", "which", "when", "where", "whom", "whose", "does", "that", "this", "with", "from", "have",
                   "about", "tell", "there", "their", "into"])


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
        if key in ("port", "top_k", "dense_extra", "answer_max_tokens"):
            return int(value)  # type: ignore[call-overload]
        if key == "query_prefix":   # "\n" in an env value / unit file = a newline (Qwen3: "Instruct: …\nQuery:")
            return str(value).replace("\\n", "\n")
        if key in JSON_KEYS:   # a JSON object: from rag.config.json as-is, from the env as a JSON string
            obj = json.loads(value) if isinstance(value, str) else value
            if not isinstance(obj, dict):
                raise ValueError(value)
            return obj
        if key in BOOL_KEYS:
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
    if not 0 <= cfg["dense_extra"] <= MAX_TOP_K:
        raise ConfigError(f"dense_extra must be 0..{MAX_TOP_K}")
    if not 16 <= cfg["answer_max_tokens"] <= 32768:
        raise ConfigError("answer_max_tokens must be 16..32768")
    if not (cfg["semantic"] or cfg["keyword"]):
        raise ConfigError("meaning search and word search are both off: turn on at least one")
    cfg["llm_base_url"] = str(cfg["llm_base_url"]).rstrip("/")
    if cfg["answer"] and not (cfg["llm_base_url"] and cfg["llm_model"]):
        raise ConfigError("answer generation needs an LLM endpoint: set llm_base_url and llm_model "
                          "(--llm-base-url / --llm-model or RAG_LLM_BASE_URL / RAG_LLM_MODEL)")
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
        import brain_container
    except ImportError:
        sys.path.insert(0, str(HERE))
        import brain_container
    return brain_container


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
            has = self._container.__contains__
        else:
            def read(name: str) -> bytes:
                return (self.dir / name).read_bytes()

            def has(name: str) -> bool:
                return (self.dir / name).is_file()
        self.manifest = json.loads(read("manifest.json").decode("utf-8"))
        self.chunks = [
            json.loads(line)
            for line in read("chunks.jsonl").decode("utf-8").splitlines()
            if line.strip()
        ]
        self.chunk_ids = [c["id"] for c in self.chunks]
        self.idx_map = {cid: i for i, cid in enumerate(self.chunk_ids)}
        # Search layers: a layer runs when the package ships its files AND the config leaves it on. A flag can turn
        # a shipped layer off, never on one that is not in the package.
        self.shipped = {"semantic": has("vectors.npy") and has("vectors.idx.json"), "keyword": has("bm25.json")}
        self.semantic = bool(self.cfg["semantic"]) and self.shipped["semantic"]
        self.keyword = bool(self.cfg["keyword"]) and self.shipped["keyword"]
        for layer, label in (("semantic", "meaning search (no vectors)"), ("keyword", "word search (no BM25 index)")):
            if self.cfg[layer] and not self.shipped[layer]:
                print(f"[warn] {label} is not in this package; it stays off", file=sys.stderr)
        if not (self.semantic or self.keyword):
            raise ConfigError("no search layer can run: this package ships "
                              + (", ".join(k for k, v in self.shipped.items() if v) or "none")
                              + " and the config turns the rest off")
        self.bm25 = json.loads(read("bm25.json").decode("utf-8")) if self.keyword else {}
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        if self.semantic:
            self.vectors = np.load(io.BytesIO(read("vectors.npy")), allow_pickle=False).astype(np.float32)
            self.idx_map = json.loads(read("vectors.idx.json").decode("utf-8"))
        rs = self.manifest.get("rag_settings") or {}
        self.rrf_k = int(rs.get("rrf_k", 60))
        emb = (self.manifest.get("embedding_model") or {}).get("name") or ""
        self.embed_model = self.cfg["embed_model"] or emb.split(":")[-1] or emb
        self.embed_base = str(self.cfg["embed_base_url"]).rstrip("/")
        self.device = _resolve_device(str(self.cfg["device"]))
        self.top_k = int(self.cfg["top_k"])
        self.query_prefix = str(self.cfg.get("query_prefix") or "")
        self.dense_extra = int(self.cfg.get("dense_extra") or 0)
        self._local_embedder = None
        self._local_embedder_failed = False
        self._reranker = None
        self._reranker_failed = False
        self.rerank_enabled = bool(self.cfg["reranker"]) and bool(rs.get("rerank_enabled", False))
        self.rerank_top_n = int(rs.get("rerank_top_n", 50))
        if self.semantic and not self.keyword and not self.embed_base \
                and not self._looks_like_model(self.dir / "embedder"):
            # Meaning search alone with nothing to embed the question: every search would come back empty.
            raise ConfigError("meaning search is the only layer on, but there is no bundled embedder and no "
                              "RAG_EMBED_BASE_URL to embed questions with")

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
            v = np.asarray(model.encode([self.query_prefix + query], normalize_embeddings=True),
                           dtype=np.float32)[0]
            return v
        if not self.embed_base:
            return None
        body = json.dumps({"input": [self.query_prefix + query], "model": self.embed_model}).encode()
        req = urllib.request.Request(
            f"{self.embed_base}/embeddings",
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
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

    def search(self, query: str, top_k: int | None = None, source_prefix: str = "",
               dense_extra: int | None = None) -> list[dict]:
        """Hybrid search; ``source_prefix`` keeps only passages whose file name starts with it
        (e.g. ``pl-`` for a per-language corpus). The reranker scores the best ``rerank_top_n``
        candidates, not only the returned ``top_k``. ``dense_extra`` (default: config) appends up to that many
        of the best passages by meaning alone that the fused ranking left out (``"via": "dense"``): a short
        question whose words match nothing ranks its answer low in the keyword half."""
        query = (query or "").strip()
        if not query:
            return []
        top_k = self.top_k if top_k is None else min(max(1, int(top_k)), MAX_TOP_K)
        allowed = np.arange(len(self.chunks))
        if source_prefix:
            allowed = np.array([i for i, c in enumerate(self.chunks)
                                if (c.get("filename") or c.get("source") or "").startswith(source_prefix)],
                               dtype=int)
            if allowed.size == 0:
                return []
        rankings: list[list[str]] = []
        dense_scores = None
        q = self.embed_query(query) if self.semantic else None
        if q is not None and self.vectors.ndim == 2 and self.vectors.shape[0] == len(self.chunks):
            if q.shape[0] == self.vectors.shape[1]:
                dense_scores = self.vectors @ q
                rankings.append([self.chunk_ids[i] for i in allowed[np.argsort(-dense_scores[allowed])]])
            else:
                print(f"[warn] query dim {q.shape[0]} != corpus dim {self.vectors.shape[1]}; "
                      "skipping semantic search", file=sys.stderr)
        bm25 = self.bm25_scores(query) if self.keyword else None
        if bm25 is not None:
            # Only passages sharing a term get a keyword rank: ranking the zero-score rest (arbitrary order) pushed
            # a passage found by meaning alone below ones that matched just "you" / "are" / "w".
            hit = allowed[bm25[allowed] > 0]
            rankings.append([self.chunk_ids[i] for i in hit[np.argsort(-bm25[hit])]])

        rrf: dict[str, float] = {}
        for ranking in rankings:
            for rank, cid in enumerate(ranking, start=1):
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        reranker = self.local_reranker()
        n_candidates = max(top_k, self.rerank_top_n) if reranker is not None else top_k
        ordered = sorted(rrf.items(), key=lambda kv: -kv[1])[: max(1, n_candidates)]

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
            if bm25 is not None:
                hit["bm25_score"] = round(float(bm25[i]), 6)
            hits.append(hit)

        if reranker is not None and len(hits) > 1:
            pool = hits[: self.rerank_top_n]
            try:
                ce = reranker.predict([(query, h["text"]) for h in pool])
                for h, s in zip(pool, ce):
                    h["ce_score"] = round(float(s), 6)
                pool.sort(key=lambda h: -h["ce_score"])
                hits = pool + hits[len(pool):]
            except Exception as e:  # noqa: BLE001
                print(f"[warn] rerank failed ({e}); keeping RRF order", file=sys.stderr)
        hits = hits[:top_k]
        extra = self.dense_extra if dense_extra is None else min(max(0, int(dense_extra)), MAX_TOP_K)
        if extra and dense_scores is not None:
            have = {h["chunk_id"] for h in hits}
            for i in allowed[np.argsort(-dense_scores[allowed])]:
                if extra <= 0:
                    break
                c = self.chunks[i]
                if c["id"] in have:
                    continue
                extra_hit = {"chunk_id": c["id"], "score": 0.0, "rrf_score": 0.0, "via": "dense",
                             "text": c.get("text", ""), "filename": c.get("filename") or c.get("source") or "",
                             "document_id": c.get("document_id", ""), "chunk_index": c.get("chunk_index", 0),
                             "dense_score": round(float(dense_scores[i]), 6)}
                if bm25 is not None:
                    extra_hit["bm25_score"] = round(float(bm25[i]), 6)
                hits.append(extra_hit)
                extra -= 1
        for rank, h in enumerate(hits, start=1):
            h["rank"] = rank
        return hits

    def layers(self) -> dict:
        """What actually runs (``shipped`` = in the package; a shipped layer can still be off or unusable)."""
        if not self.semantic:
            sem = "off" if self.shipped["semantic"] else "not in this package"
        elif self.local_embedder() is not None:
            sem = "on (bundled embedder)"
        elif self.embed_base:
            sem = "on (embeddings API)"
        else:
            sem = "unavailable: no bundled embedder and no RAG_EMBED_BASE_URL"
        kw = "on" if self.keyword else ("off" if self.shipped["keyword"] else "not in this package")
        sem_on = sem.startswith("on")
        rerank = ("on (bundled cross-encoder)" if self.local_reranker() is not None else
                  "off" if not self.rerank_enabled else "unavailable: no reranker model in this package")
        return {"semantic": sem, "keyword": kw, "fusion": "on (RRF)" if sem_on and self.keyword else "off",
                "reranker": rerank,
                "answer": f"on ({self.cfg['llm_model']})" if self.cfg["answer"] else "off"}

    def info(self) -> dict:
        lay = self.layers()
        parts = [n for n, on in (("semantic", lay["semantic"].startswith("on")),
                                 ("keyword", lay["keyword"] == "on")) if on] or ["none"]
        mode = " + ".join(parts) + (" (RRF)" if len(parts) == 2 else "")
        if lay["reranker"].startswith("on"):
            mode += " + rerank"
        if self.cfg["answer"]:
            mode += " + answer"
        return {
            "corpus": self.manifest.get("name", ""),
            "chunks": len(self.chunks),
            "documents": self.manifest.get("documents", 0),
            "encrypted": self.encrypted,
            "embed_model": self.embed_model or "(none)",
            "mode": mode,
            "layers": lay,
            "embed_endpoint": self.embed_base or "(not set)",
        }

    def answer(self, question: str, top_k: int | None = None) -> dict:
        """The generation half of RAG: retrieve, then ask the configured chat model to answer from the passages."""
        if not self.cfg["answer"]:
            raise ValueError("answer generation is off in this server (start it with --answer)")
        question = (question or "").strip()
        if not question:
            raise ValueError("question required")
        hits = self.search(question, top_k)
        sources = [{"rank": h["rank"], "filename": h["filename"], "chunk_id": h["chunk_id"]} for h in hits]
        weak = weak_match(question, hits)
        if not hits:
            return {"question": question, "answer": "Nothing in this corpus matches the question.",
                    "sources": [], "weak": True, "model": None}
        messages = [{"role": "system", "content": build_system_prompt(format_context_blocks(hits))},
                    {"role": "user", "content": question}]
        body = json.dumps({"model": self.cfg["llm_model"], "messages": messages,
                           "max_tokens": int(self.cfg["answer_max_tokens"]), "temperature": 0.2,
                           **self.cfg["llm_body"]}).encode()
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT,
                   **{str(k): str(v).replace("{session}", SESSION_ID) for k, v in self.cfg["llm_headers"].items()}}
        req = urllib.request.Request(f"{self.cfg['llm_base_url']}/chat/completions", data=body,
                                     headers=headers, method="POST")
        if self.cfg["llm_api_key"]:
            req.add_header("Authorization", f"Bearer {self.cfg['llm_api_key']}")
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode())
            text = data["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode("utf-8", "replace")
            raise RuntimeError(f"answer model at {self.cfg['llm_base_url']} said {e.code}: {detail}") from e
        except (urllib.error.URLError, KeyError, IndexError, ValueError, TimeoutError) as e:
            raise RuntimeError(f"answer model unavailable at {self.cfg['llm_base_url']}: {e}") from e
        return {"question": question, "answer": strip_thinking(text), "sources": sources, "weak": weak,
                "model": self.cfg["llm_model"]}


def format_context_blocks(results: list[dict], max_chars: int = ANSWER_CONTEXT_CHARS) -> str:
    """Numbered, source-cited passage blocks (same layout as the studio's ``prompt.format_context_blocks``)."""
    blocks: list[str] = []
    total = 0
    for r in results:
        score = r.get("ce_score") if "ce_score" in r else r.get("rrf_score", 0)
        block = f"[{r['rank']}] (source: {r.get('filename') or r.get('source') or 'source'}, score {score:.3f})\n{r['text']}"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return _BLOCK_SEP.join(blocks)


def build_system_prompt(context: str) -> str:
    return f"{DEFAULT_SYSTEM_PROMPT}\n\n{CONTEXT_MARKER}{context}"


def weak_match(query: str, hits: list[dict]) -> bool:
    """True when the passages cover too few of the question's content words (the studio's answer-gate check)."""
    terms = {w for w in re.findall(r"[a-z0-9]{4,}", query.lower()) if w not in _STOP}
    if not terms or not hits:
        return False
    blob = " ".join(str(h.get("text", "")) for h in hits).lower()
    return sum(1 for t in terms if t in blob) / len(terms) < 0.6


def strip_thinking(text: str) -> str:
    """Drop a reasoning model's <think>…</think> block from the answer."""
    return re.sub(r"(?s)^.*?</think>\s*", "", text).strip() if "</think>" in text else text.strip()


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
            elif u.path == "/ask":
                q = (parse_qs(u.query).get("q") or [""])[0]
                self._answer(q, None)
            elif u.path == "/search":
                qs = parse_qs(u.query)
                q = (qs.get("q") or [""])[0]
                try:
                    top_k = int(qs["top_k"][0]) if qs.get("top_k") else None
                    prefix = (qs.get("source_prefix") or [""])[0]
                    extra = int(qs["dense_extra"][0]) if qs.get("dense_extra") else None
                    self._json({"query": q, "hits": corpus.search(q, top_k, prefix, extra)})
                except ValueError as e:
                    self._json({"error": str(e)}, 400)
            else:
                self._json({"error": "not found — try /health, /search?q=... or /ask?q=..."}, 404)

        def _answer(self, question, top_k):
            if not corpus.cfg["answer"]:
                self._json({"error": "answer generation is off in this server (start it with --answer)"}, 404)
                return
            try:
                self._json(corpus.answer(question, top_k))
            except ValueError as e:
                self._json({"error": str(e)}, 400)
            except RuntimeError as e:
                self._json({"error": str(e)}, 502)

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
                if urlparse(self.path).path == "/ask":
                    self._answer(str(body.get("question") or body.get("query") or ""),
                                 int(top_k) if top_k is not None else None)
                    return
                hits = corpus.search(str(body.get("query", "")),
                                     int(top_k) if top_k is not None else None,
                                     str(body.get("source_prefix") or ""),
                                     int(body["dense_extra"]) if body.get("dense_extra") is not None else None)
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
            "top_k": {"type": "integer", "description": "How many passages (server default, usually 10)."},
            "source_prefix": {"type": "string",
                              "description": "Only passages from files whose name starts with this (optional)."},
            "dense_extra": {"type": "integer",
                            "description": "Also return up to this many best passages by meaning alone (optional)."},
        },
        "required": ["query"],
    },
}, {
    "name": "rag_info",
    "description": "Corpus stats: name, chunk/document counts, which search layers run.",
    "inputSchema": {"type": "object", "properties": {}},
}]
_ANSWER_TOOL = {
    "name": "rag_answer",
    "description": ("Answer a question from this corpus: searches it, then a language model writes the answer "
                    "from the passages found, citing their source files."),
    "inputSchema": {
        "type": "object",
        "properties": {"question": {"type": "string", "description": "The question."},
                       "top_k": {"type": "integer", "description": "How many passages to give the model (optional)."}},
        "required": ["question"],
    },
}


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
            tools = _TOOLS + ([_ANSWER_TOOL] if corpus.cfg["answer"] else [])
            return {"jsonrpc": "2.0", "id": mid, "result": {"tools": tools}}
        if method == "tools/call":
            params = msg.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            try:
                if name == "rag_search":
                    out = corpus.search(str(args.get("query", "")),
                                        int(args["top_k"]) if args.get("top_k") is not None else None,
                                        str(args.get("source_prefix") or ""),
                                        int(args["dense_extra"]) if args.get("dense_extra") is not None else None)
                elif name == "rag_info":
                    out = corpus.info()
                elif name == "rag_answer" and corpus.cfg["answer"]:
                    out = corpus.answer(str(args.get("question", "")),
                                        int(args["top_k"]) if args.get("top_k") is not None else None)
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
    ap.add_argument("--source-prefix", dest="source_prefix", default="",
                    help="with --query: only passages from files whose name starts with this")
    ap.add_argument("--host", help="bind address (default 127.0.0.1)")
    ap.add_argument("--port", type=int, help="HTTP port (default 8899)")
    ap.add_argument("--top-k", dest="top_k", type=int, help="results per search (default 10)")
    ap.add_argument("--device", help="auto (GPU first, default) | cpu | cuda | cuda:N | xpu | mps for bundled models")
    ap.add_argument("--reranker", dest="reranker", action="store_true", default=None,
                    help="enable the bundled reranker")
    ap.add_argument("--no-reranker", dest="reranker", action="store_false",
                    help="disable the reranker")
    ap.add_argument("--semantic", dest="semantic", action="store_true", default=None,
                    help="meaning search over the vectors (default on when the package ships them)")
    ap.add_argument("--no-semantic", dest="semantic", action="store_false", help="turn meaning search off")
    ap.add_argument("--keyword", dest="keyword", action="store_true", default=None,
                    help="word search over the BM25 index (default on when the package ships it)")
    ap.add_argument("--no-keyword", dest="keyword", action="store_false", help="turn word search off")
    ap.add_argument("--answer", dest="answer", action="store_true", default=None,
                    help="answer generation: /ask and the rag_answer tool (needs --llm-base-url and --llm-model)")
    ap.add_argument("--no-answer", dest="answer", action="store_false", help="turn answer generation off")
    ap.add_argument("--llm-base-url", dest="llm_base_url",
                    help="OpenAI-compatible chat endpoint for answers, e.g. http://127.0.0.1:11434/v1")
    ap.add_argument("--llm-model", dest="llm_model", help="model name the answer endpoint expects")
    ap.add_argument("--ask", help="answer one question (answer generation must be on), print JSON, then exit")
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
              "embed_model", "embed_base_url", "semantic", "keyword", "answer", "llm_base_url", "llm_model")


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
        print(json.dumps(corpus.search(args.query, None, args.source_prefix), ensure_ascii=False, indent=2))
        return 0
    if args.ask:
        try:
            print(json.dumps(corpus.answer(args.ask), ensure_ascii=False, indent=2))
        except (ValueError, RuntimeError) as e:
            print(f"cannot answer: {e}", file=sys.stderr)
            return 2
        return 0
    if args.http is not None:
        serve_http(corpus, cfg)
        return 0
    serve_mcp(corpus)
    return 0


if __name__ == "__main__":
    sys.exit(main())
