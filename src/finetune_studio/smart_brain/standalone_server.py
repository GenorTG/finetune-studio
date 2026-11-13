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
    --http [PORT]   GET /  chat page (bring an OpenAI / Claude / compatible API, or the package's own connection)
                    POST /mcp  MCP over HTTP (streamable HTTP; legacy and 2026-07-28 clients)
                    REST API:   GET /health   GET /version (no token needed)   GET /profile   GET /search?q=...&top_k=5[&source_prefix=pl-]
                                GET /files/<document_id>  (an original file, when the package ships them)
                    Binds 127.0.0.1 by default.  A non-loopback --host
                    REQUIRES an auth token (Authorization: Bearer ...).
    --mcp           MCP stdio server (JSON-RPC 2.0 over stdin/stdout).
                    Exposes rag_search(query, top_k), rag_profile(), rag_info(), rag_list_documents() and,
                    with original files inside, rag_get_document(document_id | filename).
    --query TEXT    one search, JSON on stdout.
    --print-config  show the effective merged config (secrets redacted).
    --update ARCHIVE  replace this package's corpus and server with a newer version of the same package
                    (keeps rag.config.json edits and the keyfile; --force for a different package id).
    --save-config   persist the given flags into rag.config.json.

Configuration precedence:  CLI flags > env (RAG_*) > rag.config.json > defaults.
    host/RAG_HOST  port/RAG_PORT  top_k/RAG_TOP_K  device/RAG_DEVICE
    reranker/RAG_RERANKER  auth_token/RAG_AUTH_TOKEN  embed_model/RAG_EMBED_MODEL
    embed_base_url/RAG_EMBED_BASE_URL  embed_api_key/RAG_EMBED_API_KEY
    query_prefix/RAG_QUERY_PREFIX (prepended to the embedded query; Qwen3: "Instruct: ...\\nQuery:")
    dense_extra/RAG_DENSE_EXTRA (also return the N best passages by meaning alone)
    follow_up/RAG_FOLLOW_UP (answers: follow-up searches the answer model may ask for, 0 = off; default 1)
    file_inline_max_mb/RAG_FILE_INLINE_MAX_MB (rag_get_document: larger files come back as a /files/ link; 20)
    hide_file_names/RAG_HIDE_FILE_NAMES (answers: a public bot names no source files: the model never sees them, any
        bracketed file name is removed from the answer and /ask returns sources only when asked: "sources": true)
    retry_missing/RAG_RETRY_MISSING (answers: when the answer says part of the question is not stated, search once
        more for that part and answer again; needs follow_up >= 1; default on)
    max_per_document/RAG_MAX_PER_DOCUMENT (passages of one document in a search's top results before others get a
        turn: -1 = the index setting, default 3; 0 = no cap)
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
import contextlib
import hmac
import io
import ipaddress
import json
import math
import os
import re
import sys
import threading
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
ORIGINALS_INDEX = "originals.json"
ORIGINALS_DIR = "originals"
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[\-'][A-Za-z0-9]+)*|[\u00A0-\uFFFF]+", re.UNICODE)
# MCP: the 2026-07-28 revision is stateless (version in each request's _meta, server/discover); older clients open
# with initialize. This server answers both ("dual-era").
MODERN_VERSIONS = ("2026-07-28",)
LEGACY_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
PROTOCOL_VERSION = LEGACY_VERSIONS[0]
META_VERSION = "io.modelcontextprotocol/protocolVersion"
UNSUPPORTED_VERSION = -32022
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
    "llm_headers": {}, "llm_body": {}, "follow_up": 1, "provider": "openai", "file_inline_max_mb": 20,
    "max_per_document": -1,
    "retry_missing": True, "hide_file_names": False,
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
    "llm_headers": "RAG_LLM_HEADERS", "llm_body": "RAG_LLM_BODY", "follow_up": "RAG_FOLLOW_UP",
    "provider": "RAG_LLM_PROVIDER", "file_inline_max_mb": "RAG_FILE_INLINE_MAX_MB",
    "max_per_document": "RAG_MAX_PER_DOCUMENT",
    "retry_missing": "RAG_RETRY_MISSING", "hide_file_names": "RAG_HIDE_FILE_NAMES",
}
SECRET_KEYS = frozenset({"auth_token", "embed_api_key", "llm_api_key"})
BOOL_KEYS = frozenset({"reranker", "semantic", "keyword", "answer", "retry_missing", "hide_file_names"})
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
MAX_FOLLOW_UPS = 3
# A question that reaches its answer through another item ("who manages the site where asset X is installed?") finds
# the first hop (X's row names the site) but never the second (that site's profile): the site is not in the question.
# The answer model is allowed to ask for one more search instead of answering. The studio's Smart Brain chat calls
# answer_with_follow_ups below, so both answer the same way (tests/test_brain_follow_up.py).
FOLLOW_UP_INSTRUCTION = (
    "If the context lacks a fact the question needs (often a second fact about something the context names but "
    "does not describe: the question reaches it through another item, e.g. the site an asset is installed at and "
    "then that site's manager), do not guess and do not answer yet: reply with only one line, SEARCH: followed by "
    "a short search for the missing fact."
)
_SEARCH_LINE = re.compile(r"(?:^[\s*`_>#-]*|(?<=[.!?)\]])\s+[*`_]*)SEARCH:[*`_]*\s*(?P<q>[^\n]{2,300}?)[\s*`_.]*$",
                          re.IGNORECASE | re.MULTILINE)
_STOP = frozenset(["what", "which", "when", "where", "whom", "whose", "does", "that", "this", "with", "from", "have",
                   "about", "tell", "there", "their", "into"])


#: package.json keys GET /version reports (no document names: it is public, like the chat page).
VERSION_KEYS = ("name", "version", "built", "created", "package_id", "slug", "parent_version", "project",
                "studio_version", "documents", "manifest_sha256", "scope", "audience", "originals", "layers")
_PACKAGE_META: dict | None = None


def hosted_meta(root: Path | None = None) -> dict:
    """The package's own metadata (``package.json`` beside server.py, plaintext even for an encrypted corpus), read
    once: name, version, build date, id, stable install name. ``{}`` for a package exported before it."""
    global _PACKAGE_META
    if root is not None:
        return _read_package_json(root)
    if _PACKAGE_META is None:
        _PACKAGE_META = _read_package_json(HERE)
    return _PACKAGE_META


def version_info(meta: Mapping | None = None) -> dict:
    """``GET /version``: the hosted metadata, the server version, and whether the corpus is encrypted."""
    meta = hosted_meta() if meta is None else meta
    out = {k: meta.get(k) for k in VERSION_KEYS if k in meta}
    out.setdefault("version", None)
    if "built" not in out and meta.get("created"):
        out["built"] = str(meta["created"])[:10]
    out["server"] = f"{SERVER_NAME}/{SERVER_VERSION}"
    out["encrypted"] = (CORPUS_DIR / ENC_NAME).is_file()
    return out


def version_label(meta: Mapping | None = None) -> str:
    """``Acme docs · 1.1.0 · 2026-11-13`` (empty for a package without package.json)."""
    meta = hosted_meta() if meta is None else meta
    if not meta.get("version"):
        return ""
    built = meta.get("built") or str(meta.get("created") or "")[:10]
    return " · ".join(str(x) for x in (meta.get("name") or "Smart Brain package", meta["version"], built) if x)


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


DEFAULT_MAX_PER_DOCUMENT = 3
CANDIDATES_PER_RESULT = 4


def diversify(hits: list, top_k: int, max_per_document: int) -> list:
    """The first ``top_k`` hits with at most ``max_per_document`` per document first, then the overflow in order
    (0 = no cap). A copy of the studio's ``smart_brain.diversity.diversify`` (this file ships on its own)."""
    if max_per_document <= 0:
        return hits[:top_k]
    kept: list = []
    overflow: list = []
    per_doc: dict = {}
    for h in hits:
        if len(kept) >= top_k:
            break
        key = str(h.get("document_id") or h.get("filename") or h.get("source") or "")
        if per_doc.get(key, 0) < max_per_document:
            per_doc[key] = per_doc.get(key, 0) + 1
            kept.append(h)
        else:
            overflow.append(h)
    return kept + overflow[: max(0, top_k - len(kept))]


# ── Image search layer (shared with the studio: smart_brain.image_embeddings imports these) ───────────────────────
# Pictures are CLIP-embedded at build time (image_vectors.npy, one row per image_assets.json record). A text query is
# embedded with the aligned text encoder; an uploaded picture with the image encoder. Pillow and
# sentence-transformers load only when an image search runs.
IMAGE_ASSETS = "image_assets.json"
IMAGE_VECTORS = "image_vectors.npy"
IMAGE_MODEL_DIR = "image_model"
IMAGE_MAX_BYTES = 12 * 1024 * 1024
IMAGE_MAX_TOP_K = 50
# Below these cosine similarities a hit is a weak match (no picture looks like the query). Measured on the
# Tarrowfield image set (tests/corpus/tarrowfield RESULTS.md, 2026-11-13).
IMAGE_WEAK_TEXT = 0.24
IMAGE_WEAK_IMAGE = 0.80
_IMAGE_STOP = _STOP | frozenset(["show", "photo", "photos", "picture", "pictures", "image", "images", "find",
                                  "the", "and", "for", "our", "you", "me", "of", "a", "an", "is", "are", "on",
                                  "in", "to", "it", "look", "like", "looks", "any", "do", "we", "get", "see"])


def decode_image(data: bytes):
    """A PIL RGB image from bytes: EXIF orientation applied (phone photos), transparency flattened on white, the first
    frame of an animation. Raises ValueError for anything Pillow cannot read."""
    try:
        from PIL import Image, ImageOps
    except ImportError as e:   # pragma: no cover - sentence-transformers brings Pillow
        raise ValueError("image search needs Pillow (pip install pillow)") from e
    with contextlib.suppress(Exception):   # HEIC/HEIF photos when the optional pillow-heif is installed
        import pillow_heif  # type: ignore[import-not-found]
        pillow_heif.register_heif_opener()
    if not data:
        raise ValueError("the image is empty")
    if len(data) > IMAGE_MAX_BYTES:
        raise ValueError(f"the image is larger than {IMAGE_MAX_BYTES // (1024 * 1024)} MB")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.seek(0)
            im = ImageOps.exif_transpose(im)
            im.load()
            if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
                rgba = im.convert("RGBA")
                flat = Image.new("RGB", rgba.size, (255, 255, 255))
                flat.paste(rgba, mask=rgba.split()[-1])
                return flat
            return im.convert("RGB")
    except Exception as e:  # PIL raises many types for a bad file
        raise ValueError(f"not a readable image: {e}") from e


class ImageEncoder:
    """A CLIP image encoder and the text encoder aligned with it (the same model when ``text_dir`` is None)."""

    def __init__(self, image_dir: Path, text_dir: Path | None, device: str):
        from sentence_transformers import SentenceTransformer

        self.device = device
        self._lock = threading.RLock()
        self._image = SentenceTransformer(str(image_dir), device=device)
        self._text = SentenceTransformer(str(text_dir), device=device) if text_dir else self._image
        probe = self.encode_texts(["a photo"])
        self.dim = int(probe.shape[1])
        if self._text is not self._image and int(self.encode_images([_blank_image()]).shape[1]) != self.dim:
            raise ValueError("the image and text encoders do not share a vector space (different dimensions)")

    def encode_images(self, images: list) -> np.ndarray:
        if not images:
            return np.zeros((0, getattr(self, "dim", 0)), dtype=np.float32)
        with self._lock:
            v = self._image.encode(list(images), normalize_embeddings=True, show_progress_bar=False, batch_size=16)
        return np.asarray(v, dtype=np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, getattr(self, "dim", 0)), dtype=np.float32)
        with self._lock:
            v = self._text.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)


def _blank_image():
    from PIL import Image
    return Image.new("RGB", (32, 32), (255, 255, 255))


def open_image_encoder(image_dir: Path, text_dir: Path | None, device: str) -> ImageEncoder:
    return ImageEncoder(Path(image_dir), Path(text_dir) if text_dir else None, device)


def image_words(record: Mapping) -> str:
    """The words a picture can be found by: its file name, where it sits, the text around it, what the parse read."""
    name = str(record.get("filename") or "").rsplit("/", 1)[-1]
    stem = re.sub(r"[_\-.]+", " ", name.rsplit(".", 1)[0])
    return " ".join(str(x) for x in (stem, record.get("locator") or "", record.get("context") or "",
                                     record.get("description") or "") if x)


def _image_terms(text: str) -> list[str]:
    return [t for t in tokenize(text) if len(t) > 1 and t not in _IMAGE_STOP]


def image_word_scores(records: list, query: str) -> np.ndarray:
    """BM25 of ``query`` over each picture's words (``image_words``); 0 = no shared term."""
    docs = [_image_terms(image_words(r)) for r in records]
    q = set(_image_terms(query))
    scores = np.zeros(len(records), dtype=np.float32)
    if not q or not docs:
        return scores
    n = len(docs)
    avgdl = (sum(len(d) for d in docs) / n) or 1.0
    df = {t: sum(1 for d in docs if t in d) for t in q}
    for i, d in enumerate(docs):
        if not d:
            continue
        counts: dict[str, int] = {}
        for t in d:
            if t in q:
                counts[t] = counts.get(t, 0) + 1
        for t, tf in counts.items():
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            scores[i] += idf * (tf * 2.5) / (tf + 1.5 * (0.25 + 0.75 * len(d) / avgdl))
    return scores


def rank_images(records: list, vectors: np.ndarray, query_vector: np.ndarray | None, *, query: str = "",
                by_image: bool = False, top_k: int = 8, rrf_k: int = 60,
                keep=None) -> tuple[list[dict], bool]:
    """Rank pictures for a query: ``(hits, weak)``.

    A text query fuses two rankings by Reciprocal Rank Fusion: how much each picture looks like the words
    (``visual_score``: cosine of the picture's vector and the text encoder's query vector) and the words found
    with it (``words_score``: BM25 over its file name, caption/page/slide text and OCR or Vision reading). An image
    query ranks by visual similarity alone. ``weak`` = nothing looks like the query and no words matched (a
    caller should say "no matching picture" rather than show the nearest one as an answer). ``keep(record)`` filters
    (a scope, a source prefix)."""
    top_k = min(max(1, int(top_k)), IMAGE_MAX_TOP_K)
    idx = [i for i, r in enumerate(records) if keep is None or keep(r)]
    if not idx or vectors.ndim != 2 or vectors.shape[0] != len(records):
        return [], True
    allowed = np.asarray(idx, dtype=int)
    visual = np.zeros(len(records), dtype=np.float32)
    if query_vector is not None:
        visual = vectors @ np.asarray(query_vector, dtype=np.float32)
    words = image_word_scores(records, query) if (query and not by_image) else np.zeros(len(records), np.float32)
    rankings = []
    if query_vector is not None:
        rankings.append(list(allowed[np.argsort(-visual[allowed], kind="stable")]))
    hit_words = allowed[words[allowed] > 0]
    if hit_words.size:
        rankings.append(list(hit_words[np.argsort(-words[hit_words], kind="stable")]))
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking, start=1):
            fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (rrf_k + rank)
    threshold = IMAGE_WEAK_IMAGE if by_image else IMAGE_WEAK_TEXT
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], -float(visual[kv[0]])))[:top_k]
    hits = []
    for rank, (i, score) in enumerate(ordered, start=1):
        r = records[i]
        looks = query_vector is not None and float(visual[i]) >= threshold
        said = float(words[i]) > 0
        hits.append({
            "rank": rank, "asset_id": r.get("asset_id"), "document_id": r.get("document_id", ""),
            "filename": r.get("filename", ""), "source": r.get("source", ""), "locator": r.get("locator", ""),
            "kind": r.get("kind", "picture"), "mime_type": r.get("mime_type", ""),
            "width": r.get("width"), "height": r.get("height"),
            "description": str(r.get("description") or "")[:600], "read_by": r.get("read_by", ""),
            "score": round(float(score), 6), "visual_score": round(float(visual[i]), 4),
            "words_score": round(float(words[i]), 4),
            "match": "+".join(m for m, on in (("looks", looks), ("words", said)) if on) or "weak",
        })
    weak = not any(h["match"] != "weak" for h in hits)
    return hits, weak


def image_documents(hits: list[dict]) -> list[dict]:
    """Image hits grouped by their document, best first (image -> document search)."""
    docs: dict[str, dict] = {}
    for h in hits:
        did = str(h.get("document_id") or "")
        row = docs.setdefault(did, {"document_id": did, "filename": h.get("filename", ""), "score": h["score"],
                                    "visual_score": h["visual_score"], "images": []})
        row["images"].append({"asset_id": h["asset_id"], "locator": h.get("locator", ""), "rank": h["rank"]})
    for rank, row in enumerate(docs.values(), start=1):
        row["rank"] = rank
    return list(docs.values())


class ConfigError(Exception):
    """Bad configuration / refusal to start (reported, exit code 2)."""


def _coerce(key: str, value: object) -> object:
    try:
        if key in ("port", "top_k", "dense_extra", "answer_max_tokens", "follow_up", "file_inline_max_mb",
                   "max_per_document"):
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
    if not 0 <= cfg["follow_up"] <= MAX_FOLLOW_UPS:
        raise ConfigError(f"follow_up must be 0..{MAX_FOLLOW_UPS}")
    if not -1 <= cfg["max_per_document"] <= MAX_TOP_K:
        raise ConfigError(f"max_per_document must be -1 (the index setting), 0 (no cap) or 1..{MAX_TOP_K}")
    if not 1 <= cfg["file_inline_max_mb"] <= 512:
        raise ConfigError("file_inline_max_mb must be 1..512")
    if not (cfg["semantic"] or cfg["keyword"]):
        raise ConfigError("meaning search and word search are both off: turn on at least one")
    cfg["llm_base_url"] = str(cfg["llm_base_url"]).rstrip("/")
    cfg["provider"] = str(cfg["provider"] or "openai").lower()
    if cfg["provider"] not in ("openai", "anthropic"):
        raise ConfigError("provider must be openai (any OpenAI-compatible API) or anthropic")
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
        self.identity = dict(self.manifest.get("identity") or {})   # who it speaks for and how it answers (studio)
        # Original files ("Include original files" at export): originals.json lists them, originals/<id> holds each.
        self.originals: dict[str, dict] = {}
        if has(ORIGINALS_INDEX):
            for row in json.loads(read(ORIGINALS_INDEX).decode("utf-8")):
                if isinstance(row, dict) and _SAFE_ID.fullmatch(str(row.get("document_id") or "")):
                    self.originals[str(row["document_id"])] = row
        self._read = read
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
        # One lock for loading AND running the bundled models: the HTTP server answers each request on its own
        # thread, and parallel requests running the models on the GPU at once crashed the server with "CUDA error:
        # an illegal memory access" (4 clients, 165 searches, 2026-10-29). Concurrent first requests also loaded a
        # model twice. The studio's own search serialises each model the same way (model_cache).
        self._model_lock = threading.RLock()
        self.rerank_enabled = bool(self.cfg["reranker"]) and bool(rs.get("rerank_enabled", False))
        self.rerank_top_n = int(rs.get("rerank_top_n", 50))
        # -1 = the index's own setting (the studio's Smart Brain settings; 3 when it has none)
        cap = int(self.cfg.get("max_per_document", -1))
        self.max_per_document = int(rs.get("max_per_document", DEFAULT_MAX_PER_DOCUMENT)) if cap < 0 else cap
        if self.semantic and not self.keyword and not self.embed_base \
                and not self._looks_like_model(self.dir / "embedder"):
            # Meaning search alone with nothing to embed the question: every search would come back empty.
            raise ConfigError("meaning search is the only layer on, but there is no bundled embedder and no "
                              "RAG_EMBED_BASE_URL to embed questions with")

    def original_for(self, document_id: str) -> dict | None:
        """The original-file row of a document (a fact-card row ``<id>~facts`` shares its file's), or None."""
        return self.originals.get(str(document_id or "").split("~", 1)[0])

    def find_original(self, document_id: str = "", filename: str = "") -> dict | None:
        """By id, else by file name (exact, then case-insensitive, then a unique partial match)."""
        if document_id:
            return self.original_for(document_id)
        name = (filename or "").strip()
        if not name:
            return None
        indexed = {d["document_id"]: d["filename"] for d in self.documents(limit=1 << 30)}
        rows = [(r, {r["filename"], indexed.get(r["document_id"], r["filename"])}) for r in self.originals.values()]
        for test in (lambda n: n == name, lambda n: n.lower() == name.lower()):
            found = [r for r, names in rows if any(test(n) for n in names)]
            if found:
                return found[0]
        partial = [r for r, names in rows if any(name.lower() in n.lower() for n in names)]
        return partial[0] if len(partial) == 1 else None

    def read_original(self, document_id: str) -> bytes:
        """The original file's bytes (decrypted in memory for an encrypted package). KeyError when there is none."""
        row = self.original_for(document_id)
        if row is None:
            raise KeyError(document_id)
        return self._read(f"{ORIGINALS_DIR}/{row['document_id']}")

    def documents(self, name_contains: str = "", about: str = "", limit: int = 200) -> list[dict]:
        """The documents in this package: id, file name, type, size, date, whether the original file is inside.

        ``name_contains`` filters by file name; ``about`` keeps the documents whose passages a search for it finds,
        best first ("which files do you have about X")."""
        rows: dict[str, dict] = {}
        for meta in (self.manifest.get("extra") or {}).get("documents_meta") or []:
            did = str(meta.get("document_id") or "").split("~", 1)[0]
            if did and did not in rows:
                rows[did] = {"document_id": did, "filename": str(meta.get("filename") or "")}
        for c in self.chunks:
            did = str(c.get("document_id") or "").split("~", 1)[0]
            if did and did not in rows:
                rows[did] = {"document_id": did, "filename": c.get("filename") or c.get("source") or ""}
        for did, row in rows.items():
            orig = self.originals.get(did)
            if orig and orig.get("filename") and orig["filename"] != row["filename"]:
                row["original_filename"] = orig["filename"]
            row.update({"type": (orig or {}).get("mime_type") or _guess_type(row["filename"]),
                        "size": (orig or {}).get("size"), "date": (orig or {}).get("date"),
                        "original": orig is not None})
        found = list(rows.values())
        if name_contains:
            needle = name_contains.lower()
            found = [r for r in found if needle in r["filename"].lower()]
        if about:
            order: dict[str, int] = {}
            for h in self.search(about, MAX_TOP_K):
                order.setdefault(str(h.get("document_id") or "").split("~", 1)[0], len(order))
            found = sorted((r for r in found if r["document_id"] in order), key=lambda r: order[r["document_id"]])
        return found[: max(1, int(limit))]

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
        with self._model_lock:
            return self._load_embedder()

    def _load_embedder(self):
        if self._local_embedder is not None or self._local_embedder_failed:
            return self._local_embedder   # another request loaded it while this one waited
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
        with self._model_lock:
            return self._load_reranker()

    def _load_reranker(self):
        if self._reranker is not None or self._reranker_failed:
            return self._reranker
        d = self.dir / "reranker"
        if not self._looks_like_model(d):
            return None
        try:
            from sentence_transformers import CrossEncoder
            kwargs: dict = {"max_length": 512}   # the studio's own reranker reads passages the same length
            if self.device.startswith("cuda"):   # fp16 on a GPU: ~4x faster, same ranking (scores are not stored)
                import torch
                kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
            self._reranker = CrossEncoder(str(d), device=self.device, **kwargs)
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
            with self._model_lock:
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
        if self.max_per_document > 0:     # room to pass over a document's extra passages
            n_candidates = max(n_candidates, top_k * CANDIDATES_PER_RESULT)
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
            if self.original_for(hit["document_id"]) is not None:
                hit["original"] = True
            if dense_scores is not None:
                hit["dense_score"] = round(float(dense_scores[i]), 6)
            if bm25 is not None:
                hit["bm25_score"] = round(float(bm25[i]), 6)
            hits.append(hit)

        if reranker is not None and len(hits) > 1:
            pool = hits[: self.rerank_top_n]
            try:
                with self._model_lock:
                    ce = reranker.predict([(query, h["text"]) for h in pool])
                for h, s in zip(pool, ce):
                    h["ce_score"] = round(float(s), 6)
                pool.sort(key=lambda h: -h["ce_score"])
                hits = pool + hits[len(pool):]
            except Exception as e:  # noqa: BLE001
                print(f"[warn] rerank failed ({e}); keeping RRF order", file=sys.stderr)
        hits = diversify(hits, top_k, self.max_per_document)
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
                if self.original_for(extra_hit["document_id"]) is not None:
                    extra_hit["original"] = True
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
            "identity": self.identity.get("name") or ("set" if self.identity else "none"),
            "original_files": len(self.originals),
            "scope": self.manifest.get("scope") or {"label": "all documents"},
            "package": self.manifest.get("package") or {},
            "version": version_info(),
            "hide_file_names": bool(self.cfg.get("hide_file_names")),
            "answer_connection": {
                "configured": bool(self.cfg["answer"] and self.cfg["llm_base_url"] and self.cfg["llm_model"]),
                "provider": self.cfg["provider"], "base_url": self.cfg["llm_base_url"], "model": self.cfg["llm_model"],
                "key_set": bool(self.cfg["llm_api_key"]),
            },
        }

    def profile(self) -> dict:
        """The identity baked in at export: who it speaks for, how it sounds, how to answer, and those as one prompt."""
        return {key: self.identity.get(key, "") for key in ("name", "about", "personality", "answer_rules")} | {
            "instructions": self.identity.get("system_prompt", "")}

    def answer(self, question: str, top_k: int | None = None, provider: Mapping | None = None,
               history: list[dict] | None = None, include_sources: bool | None = None) -> dict:
        """The generation half of RAG: retrieve, then ask the configured chat model to answer from the passages.

        ``hide_file_names`` (a public bot): the model sees passages without file names and is told to name none,
        any it writes anyway is removed, and the reply carries ``sources`` only when ``include_sources`` asks for
        them (the operator's tools); otherwise ``sources`` is ``[]`` and ``sources_hidden`` is true."""
        hide = bool(self.cfg.get("hide_file_names"))
        if not self.cfg["answer"] and not provider:
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
        chat = self._chat if provider is None else lambda messages: self._chat(messages, provider)
        result = answer_with_follow_ups(
            question, hits, lambda q: self.search(q, top_k), chat,
            source_search=lambda q, prefix: self.search(q, top_k, source_prefix=prefix),
            follow_ups=int(self.cfg["follow_up"]),
            missing_part_retry=bool(self.cfg["retry_missing"]),
            system_prompt=self.identity.get("system_prompt", ""), history=history,
            system_turn=_anonymous_turn if hide else None)
        sources = [{"rank": h["rank"], "filename": h["filename"], "chunk_id": h["chunk_id"],
                    "document_id": h.get("document_id", ""), "original": bool(h.get("original")),
                    "preview": str(h.get("text") or "")[:600]} for h in result["hits"]]
        answer, followup = result["answer"], result.get("followup")
        if hide:
            answer = strip_file_names(answer, [str(h.get("filename") or "").rsplit("/", 1)[-1] for h in result["hits"]])
            if followup:
                followup = {"query": followup["query"], "added_passages": followup["added_passages"]}
            if not include_sources:
                sources = []
        out = {"question": question, "answer": answer, "sources": sources, "weak": weak,
               "model": (provider or {}).get("llm_model", self.cfg["llm_model"]),
               "follow_up_searches": result["follow_up_searches"], "followup": followup}
        if hide:
            out["sources_hidden"] = not include_sources
        return out

    def _chat(self, messages: list[dict], provider: Mapping | None = None) -> str:
        """One completion from the configured answer model; the reply with any <think> block removed."""
        settings = {**self.cfg, **(provider or {})}
        api = str(settings.get("provider") or "openai").lower()
        if api not in {"openai", "anthropic"}:
            raise RuntimeError("provider must be openai or anthropic")
        if api == "anthropic":
            system = next((str(m["content"]) for m in messages if m.get("role") == "system"), "")
            turns = [m for m in messages if m.get("role") != "system"]
            request_body = {"model": settings["llm_model"], "messages": turns,
                            "max_tokens": int(settings["answer_max_tokens"]), "system": system}
            endpoint = str(settings["llm_base_url"]).rstrip("/") + "/messages"
        else:
            request_body = {"model": settings["llm_model"], "messages": messages,
                            "max_tokens": int(settings["answer_max_tokens"]), "temperature": 0.2,
                            **settings["llm_body"]}
            endpoint = str(settings["llm_base_url"]).rstrip("/") + "/chat/completions"
        body = json.dumps(request_body).encode()
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT,
                   **{str(k): str(v).replace("{session}", SESSION_ID) for k, v in settings["llm_headers"].items()}}
        req = urllib.request.Request(endpoint, data=body,
                                     headers=headers, method="POST")
        if settings["llm_api_key"]:
            if api == "anthropic":
                req.add_header("x-api-key", str(settings["llm_api_key"]))
                req.add_header("anthropic-version", "2026-10-21")
            else:
                req.add_header("Authorization", f"Bearer {settings['llm_api_key']}")
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode())
            text = ("".join(str(block.get("text", "")) for block in data.get("content", [])
                            if block.get("type") == "text") if api == "anthropic"
                    else data["choices"][0]["message"]["content"]) or ""
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode("utf-8", "replace")
            raise RuntimeError(f"answer model at {endpoint} said {e.code}: {detail}") from e
        except (urllib.error.URLError, KeyError, IndexError, ValueError, TimeoutError) as e:
            raise RuntimeError(f"answer model unavailable at {endpoint}: {e}") from e
        return strip_thinking(text)


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


def build_system_prompt(context: str, system_prompt: str = "", extra_instructions: str = "") -> str:
    """The studio's ``smart_brain.prompt.build_system_prompt``, copied (this file ships on its own)."""
    instructions = system_prompt or DEFAULT_SYSTEM_PROMPT
    if extra_instructions:
        instructions = f"{instructions} {extra_instructions}"
    return f"{instructions}\n\n{CONTEXT_MARKER}{context}"


def _system_turn(hits: list[dict], max_chars: int, system_prompt: str, extra_instructions: str) -> str:
    return build_system_prompt(format_context_blocks(hits, max_chars=max_chars), system_prompt, extra_instructions)


# "Hide file names in answers" (a public bot): customers should not see internal-looking file names in citations.
CITE_RULE = "Quote the source filename in [brackets] when relevant."
NO_NAMES_RULE = ("Never name source files, file paths or document file names in your answer and write no bracketed "
                 "citations: the reader cannot open them. Say what the information is, not which file it came from.")
_FILE_EXT = (r"pdf|docx?|xlsx?|xlsm|pptx?|odt|ods|odp|rtf|md|markdown|txt|csv|tsv|json|jsonl|html?|xml|eml|msg|epub|"
             r"png|jpe?g|tiff?")
_BRACKET_NAME = re.compile(r"\s*\[[^\]\n]{0,200}?\.(?:" + _FILE_EXT + r")(?:\s*[,;]\s*[^\]\n]{0,200}?)?\]",
                           re.IGNORECASE)
_SOURCE_LABEL = re.compile(r"\(source:\s*[^,)]*,\s*", re.IGNORECASE)


def hide_names_prompt(system_prompt: str) -> str:
    """The answer instructions without the cite-the-file rule, with the never-name-files rule instead."""
    base = system_prompt or DEFAULT_SYSTEM_PROMPT
    return base.replace(CITE_RULE, NO_NAMES_RULE) if CITE_RULE in base else f"{base} {NO_NAMES_RULE}"


def _anonymous_turn(hits: list[dict], max_chars: int, system_prompt: str, extra_instructions: str) -> str:
    """``_system_turn`` with passages labelled only by number: the model never sees a file name."""
    blind = [{**h, "filename": "passage", "source": "passage"} for h in hits]
    context = _SOURCE_LABEL.sub("(", format_context_blocks(blind, max_chars=max_chars))
    return build_system_prompt(context, hide_names_prompt(system_prompt), extra_instructions)


def strip_file_names(answer: str, names: list[str] | tuple[str, ...] = ()) -> str:
    """The answer without bracketed file citations (``[terms_v4.docx]``) or the passages' own file names."""
    out = _BRACKET_NAME.sub("", answer or "")
    for name in sorted({n for n in names if n and len(n) > 4}, key=len, reverse=True):
        out = re.sub(r"\[?" + re.escape(name) + r"\]?", "our documents", out)
    return re.sub(r"[ \t]+([.,;:!?])", r"\1", out).strip()


def follow_up_query(reply: str) -> str:
    """The search an answer model asked for (a ``SEARCH: …`` line in its reply; the last one wins), else ``""``.

    Models often explain first ("X is at site Y, but the context does not name its manager.") and put the
    search on the last line, so the line may come anywhere; a reply that asks for a search is not an answer.
    """
    found = _SEARCH_LINE.findall(reply or "")
    return found[-1].strip().strip("\"'") if found else ""


def merge_follow_up_hits(hits: list[dict], extra: list[dict]) -> list[dict]:
    """The passages already used, interleaved with a follow-up search's new ones, ranks renumbered.

    Interleaved, not appended: the context is cut at a character budget, and appended passages of the second hop
    were the ones cut.
    """
    seen = {h.get("chunk_id") for h in hits}
    new = [h for h in extra if h.get("chunk_id") not in seen]
    merged: list[dict] = []
    for i in range(max(len(hits), len(new))):
        merged.extend(x[i] for x in (hits, new) if i < len(x))
    return [{**h, "rank": rank} for rank, h in enumerate(merged, start=1)]


# An answer that says part of the question is not in the passages ("the value is not stated", "the context does not
# mention who signed it"). Tarrowfield in-008: the £211,450 passage was rank 9 of 10 and the model said "not stated";
# a search for the missing part alone ranks it first. One such retry per answer (``missing_part_retry``).
_NOT_STATED = re.compile(
    r"\b(?:(?:is|are|was|were)\s+not\s+(?:stated|mentioned|specified|given|provided|included|listed|shown|available|"
    r"found|named|disclosed|clear)|(?:isn't|aren't|wasn't|weren't)\s+(?:stated|mentioned|specified|given|provided|"
    r"included|listed|shown|available|found|named)|(?:does|do|did)\s*(?:n't|\s+not)\s+(?:say|state|mention|"
    r"specify|include|contain|give|provide|list|name|show)|no\s+(?:information|mention|details?|figure|value)\s+"
    r"(?:is\s+given\s+)?(?:about|on|of|for|regarding)|(?:could|can)\s*(?:n't|not)\s+find|unable\s+to\s+find|"
    r"not\s+in\s+the\s+(?:context|documents?|passages?|sources?))\b", re.IGNORECASE)
MISSING_PART_PROMPT = (
    "A user asked: {question}\n\nThe answer written from the documents found so far says part of it is missing:\n"
    "{answer}\n\nWrite one short search query (a few words, the names and terms the documents would use) for ONLY "
    "the missing part. Reply with the query alone, no quotes, no explanation."
)


def says_not_stated(answer: str) -> bool:
    """True when an answer says (part of) the question is not in the passages."""
    return bool(_NOT_STATED.search(answer or ""))


def uncovered_clause(question: str, answer: str) -> str:
    """The part of the question the answer covers least (a fallback sub-query when the model writes none)."""
    parts = [p.strip(" ,;?.") for p in re.split(r"\?|;|,|\band\b|\balso\b", question or "") if p.strip(" ,;?.")]
    words = {w for w in re.findall(r"[a-z0-9£€$%.]{3,}", (answer or "").lower())}

    def covered(part: str) -> float:
        toks = [w for w in re.findall(r"[a-z0-9£€$%.]{3,}", part.lower()) if w not in _STOP]
        return sum(t in words for t in toks) / len(toks) if toks else 1.0

    return min(parts, key=covered) if len(parts) > 1 else (question or "").strip()


def missing_part_query(question: str, answer: str, chat) -> str:
    """A search for the part an answer says is missing: the answer model writes it, else the uncovered clause."""
    try:
        reply = chat([{"role": "user", "content": MISSING_PART_PROMPT.format(question=question, answer=answer)}])
    except Exception as e:  # noqa: BLE001 - the retry is best effort: fall back to the question's own words
        print(f"[warn] missing-part query failed ({e}); using the question's clause", file=sys.stderr)
        reply = ""
    line = next((ln.strip().strip("\"'`*") for ln in strip_thinking(reply or "").splitlines() if ln.strip()), "")
    line = re.sub(r"^(?:search|query)\s*:\s*", "", line, flags=re.IGNORECASE).strip()
    return line[:200] if 2 <= len(line) <= 200 else uncovered_clause(question, answer)


def answer_with_follow_ups(question: str, hits: list[dict], search, chat, *, follow_ups: int = 1,
                           max_chars: int = ANSWER_CONTEXT_CHARS, system_prompt: str = "",
                           history: list[dict] | None = None, system_turn=None,
                           missing_part_retry: bool = True, source_search=None) -> dict:
    """Answer from ``hits``; the model may instead ask for up to ``follow_ups`` more searches.

    ``search(query) -> hits`` and ``chat(messages) -> reply`` are the caller's (this server's search and answer
    endpoint, or the studio's index and loaded model). ``history`` = earlier conversation turns before the question.
    ``system_turn(hits, max_chars, system_prompt, extra_instructions) -> str`` builds the system message; the studio
    passes ``smart_brain.prompt``'s (the training rows' layout), this server its own copy of it.
    The last round never offers a search, so the model always ends with an answer.
    ``missing_part_retry``: when that answer says part of the question is not stated, one more search runs for the
    missing part (``missing_part_query``) and the answer is written once more with the new passages merged in;
    when ``source_search`` is provided and the answer cited a source, the retry searches only within up to two cited
    sources (falling back to global search if that scoped search finds nothing). ``followup`` then says what was
    searched and which sources it added (None when it did not run).
    Returns ``{"answer", "hits", "follow_up_searches", "followup"}``.
    """
    asked: list[str] = []
    system_turn = system_turn or _system_turn
    offering = follow_ups > 0
    while True:
        system = system_turn(hits, max_chars, system_prompt, FOLLOW_UP_INSTRUCTION if offering else "")
        messages = [{"role": "system", "content": system}, *(history or []), {"role": "user", "content": question}]
        reply = chat(messages)
        wanted = follow_up_query(reply) if offering else ""
        if not wanted:
            break
        if wanted.lower() in {a.lower() for a in [question, *asked]}:   # nothing new to find: answer now
            offering = False
            continue
        asked.append(wanted)
        hits = merge_follow_up_hits(hits, search(wanted))
        offering = len(asked) < follow_ups
    if follow_up_query(reply):   # it asked for a search although none was offered: there is no answer to show
        reply = "I could not find the answer in the documents."
    followup = None
    if missing_part_retry and follow_ups > 0 and says_not_stated(reply):
        wanted = missing_part_query(question, reply, chat)
        cited_prefixes = _cited_source_prefixes(reply, hits)
        if source_search and cited_prefixes:
            scoped = [h for prefix in cited_prefixes for h in source_search(wanted, prefix)]
            candidates = scoped or search(wanted)
        else:
            candidates = search(wanted)
        seen = {h.get("chunk_id") for h in hits}
        extra = [h for h in candidates if h.get("chunk_id") not in seen]
        followup = {"query": wanted, "reason": "the answer said part of the question is not stated",
                    "added_sources": sorted({str(h.get("filename") or h.get("source") or "") for h in extra}),
                    "added_passages": len(extra), "answer_before": reply}
        print(f"[info] missing-part search {wanted!r}: {len(extra)} new passage(s)", file=sys.stderr)
        if extra:
            hits = merge_follow_up_hits(hits, extra)
            system = system_turn(hits, max_chars, system_prompt, "")
            again = chat([{"role": "system", "content": system}, *(history or []), {"role": "user", "content": question}])
            reply = "I could not find the answer in the documents." if follow_up_query(again) else again
        asked.append(wanted)
    return {"answer": reply, "hits": hits, "follow_up_searches": asked, "followup": followup}


def _cited_source_prefixes(answer: str, hits: list[dict]) -> list[str]:
    """Return up to two source prefixes explicitly cited in the answer."""
    answer_lower = (answer or "").lower()
    prefixes: list[str] = []
    for hit in hits:
        filename = str(hit.get("filename") or "")
        source = str(hit.get("source") or "")
        basename = source.rsplit("/", 1)[-1]
        if any(value and value.lower() in answer_lower for value in (filename, basename)):
            prefix = source or filename
            if prefix and prefix not in prefixes:
                prefixes.append(prefix)
        if len(prefixes) == 2:
            break
    return prefixes


def _guess_type(filename: str) -> str:
    import mimetypes
    return mimetypes.guess_type(filename or "")[0] or "application/octet-stream"


def _disposition(filename: str) -> str:
    """``attachment`` with an ASCII fallback name and the real one RFC 5987-encoded."""
    from urllib.parse import quote
    name = (filename or "document").replace("\\", "_").replace("/", "_")
    ascii_name = re.sub(r'[^A-Za-z0-9._ -]', "_", name).strip() or "document"
    return f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{quote(name, safe="")}'


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
        def end_headers(self):
            ver = hosted_meta().get("version")
            if ver:   # every response says which package version answered (proxies, logs, update checks)
                self.send_header("X-Package-Version", str(ver))
            super().end_headers()

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
                if self.command == "GET" and urlparse(self.path).path in ("/", "/app.js", "/version"):
                    # the static client and the package's name/version hold no corpus data; API calls stay authenticated
                    return True
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
            if u.path == "/":
                self._static(_CHAT_PAGE, "text/html; charset=utf-8")
            elif u.path == "/app.js":
                self._static(_CHAT_JS, "text/javascript; charset=utf-8")
            elif u.path == "/mcp":
                self._json({"error": "the MCP endpoint takes POST"}, 405, {"Allow": "POST"})
            elif u.path == "/health":
                self._json({"ok": True, **corpus.info()})
            elif u.path == "/version":
                self._json(version_info())
            elif u.path == "/profile":
                self._json(corpus.profile())
            elif u.path.startswith("/files/"):
                self._file(u.path[len("/files/"):])
            elif u.path == "/ask":
                qs = parse_qs(u.query)
                q = (qs.get("q") or [""])[0]
                self._answer(q, None, {"sources": (qs.get("sources") or [""])[0] in ("1", "true", "yes")})
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
                self._json({"error": "not found — try /, /health, /version, /profile, /search?q=..., /ask?q=..., /files/<id> "
                                     "or POST /mcp"}, 404)

        def _file(self, document_id: str):
            """GET /files/<document_id>: an original file, as a download. Ids come from the package's own list only."""
            row = corpus.original_for(document_id) if _SAFE_ID.fullmatch(document_id) else None
            if row is None:
                self._json({"error": "no original file with that id in this package"}, 404)
                return
            try:
                data = corpus.read_original(row["document_id"])
            except (KeyError, OSError) as e:
                self._json({"error": f"original file unreadable: {e}"}, 500)
                return
            self.send_response(200)
            self.send_header("Content-Type", row.get("mime_type") or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Disposition", _disposition(row.get("filename") or ""))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def _static(self, text: str, kind: str):
            data = text.encode()
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'unsafe-inline'; "
                             "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def _origin_ok(self) -> bool:
            """MCP over HTTP: a browser page from another site must not reach this server (DNS rebinding)."""
            origin = self.headers.get("Origin")
            if not origin:
                return True
            host = urlparse(origin).hostname or ""
            own = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            return is_loopback(host) or (bool(own) and host == own)

        def _mcp(self, body):
            """Streamable HTTP MCP: one JSON-RPC message per POST, one JSON reply (a notification gets 202)."""
            def reply(obj, code):
                self._json(obj, code)
            if not isinstance(body, dict):
                reply({"jsonrpc": "2.0", "error": {"code": -32600, "message": "send one JSON-RPC message per request"}}, 400)
                return
            header = self.headers.get("MCP-Protocol-Version")
            meta = ((body.get("params") or {}).get("_meta") or {}) if isinstance(body.get("params"), dict) else {}
            in_body = meta.get(META_VERSION) if isinstance(meta, dict) else None
            if header and in_body and header != in_body:
                reply({"jsonrpc": "2.0", "id": body.get("id"), "error": {
                    "code": -32600, "message": "HeaderMismatch: MCP-Protocol-Version differs from the request _meta"}}, 400)
                return
            out = mcp_responder(corpus, base_url=f"http://{self.headers.get('Host') or 'localhost'}")(body)
            if out is None:
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            code = (out.get("error") or {}).get("code")
            status = 400 if code == UNSUPPORTED_VERSION else 404 if (code == -32601 and in_body) else 200
            reply(out, status)

        def _answer(self, question, top_k, body=None):
            provider = None
            if body and body.get("provider"):
                provider = {"provider": str(body.get("provider")),
                            "llm_base_url": str(body.get("llm_base_url") or "").strip().rstrip("/"),
                            "llm_model": str(body.get("llm_model") or "").strip(),
                            "llm_api_key": str(body.get("llm_api_key") or ""),
                            "llm_headers": {}, "llm_body": {}}
                if provider["provider"] == "package":
                    provider = None
                elif provider["provider"] not in {"openai", "anthropic"}:
                    self._json({"error": "provider must be openai or anthropic"}, 400)
                    return
                if provider and (not provider["llm_base_url"].startswith(("https://", "http://"))
                                 or not provider["llm_model"]):
                    self._json({"error": "enter a valid provider URL and model"}, 400)
                    return
            elif not corpus.cfg["answer"]:
                self._json({"error": "connect a provider in the chat settings or start with --answer and configure an endpoint"}, 400)
                return
            history = []
            if body and body.get("history") is not None:
                raw_history = body["history"]
                if not isinstance(raw_history, list):
                    self._json({"error": "history must be a list of user and assistant turns"}, 400)
                    return
                for turn in raw_history[-12:]:
                    if (not isinstance(turn, dict) or turn.get("role") not in {"user", "assistant"}
                            or not isinstance(turn.get("content"), str)):
                        self._json({"error": "history turns need a user or assistant role and text content"}, 400)
                        return
                    history.append({"role": turn["role"], "content": turn["content"][:8000]})
            want = bool(body.get("sources")) if body else False
            try:
                self._json(corpus.answer(question, top_k, provider, history, include_sources=want))
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
                if urlparse(self.path).path == "/mcp":
                    if not self._origin_ok():
                        self._json({"jsonrpc": "2.0", "error": {"code": -32600, "message": "forbidden origin"}}, 403)
                        return
                    self._mcp(body)
                    return
                if not isinstance(body, dict):
                    self._json({"error": "send a JSON object"}, 400)
                    return
                top_k = body.get("top_k")
                if urlparse(self.path).path == "/ask":
                    self._answer(str(body.get("question") or body.get("query") or ""),
                                 int(top_k) if top_k is not None else None, body)
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
    print(f"[rag-server] {version_label() or 'package'} {corpus.info()} — listening on http://{host}:{port}/search?q=... "
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
    "annotations": {"title": "Search the knowledge base", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True,
                    "openWorldHint": False},
}, {
    "name": "rag_profile",
    "description": ("Who this knowledge base speaks for: its name, background, personality and answering rules. "
                    "Read it first and follow its instructions when you answer."),
    "inputSchema": {"type": "object", "properties": {}},
    "annotations": {"title": "Who this knowledge base speaks for", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True,
                    "openWorldHint": False},
}, {
    "name": "rag_info",
    "description": "Corpus stats: name, chunk/document counts, which search layers run.",
    "inputSchema": {"type": "object", "properties": {}},
    "annotations": {"title": "Knowledge base stats", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True,
                    "openWorldHint": False},
}]
_LIST_TOOL = {
    "name": "rag_list_documents",
    "description": ("The documents in this knowledge base: file name, id, type, size, date, and whether the original "
                    "file can be downloaded (rag_get_document). Filter by file name, or ask which files are about a "
                    "topic."),
    "inputSchema": {
        "type": "object",
        "properties": {
            "name_contains": {"type": "string", "description": "Only files whose name contains this (optional)."},
            "about": {"type": "string",
                      "description": "Only files with passages about this, best match first (optional)."},
            "limit": {"type": "integer", "description": "At most this many (default 200)."},
        },
    },
    "annotations": {"title": "List the documents", "readOnlyHint": True, "destructiveHint": False,
                    "idempotentHint": True, "openWorldHint": False},
}
_GET_TOOL = {
    "name": "rag_get_document",
    "description": ("Get one original file of this knowledge base (the PDF, spreadsheet, … that was uploaded), by "
                    "document_id (from rag_search or rag_list_documents) or file name. Returned as an embedded "
                    "resource; a file over the size limit comes back as a download link instead."),
    "inputSchema": {
        "type": "object",
        "properties": {
            "document_id": {"type": "string", "description": "The document's id (preferred)."},
            "filename": {"type": "string", "description": "The file name, when the id is not known."},
        },
    },
    "annotations": {"title": "Get an original file", "readOnlyHint": True, "destructiveHint": False,
                    "idempotentHint": True, "openWorldHint": False},
}
_ANSWER_TOOL = {
    "name": "rag_answer",
    "description": ("Answer a question from this corpus: searches it, then a language model writes the answer "
                    "from the passages found, citing their source files."),
    "inputSchema": {
        "type": "object",
        "properties": {"question": {"type": "string", "description": "The question."},
                       "top_k": {"type": "integer", "description": "How many passages to give the model (optional)."},
                       "include_sources": {"type": "boolean", "description": (
                           "Also return the sources when the package hides file names in answers (operator use).")}},
        "required": ["question"],
    },
    # Read-only for the package, but it sends the passages to the configured language model: an outside service.
    "annotations": {"title": "Answer from the knowledge base", "readOnlyHint": True, "destructiveHint": False,
                    "idempotentHint": False, "openWorldHint": True},
}


def mcp_server_info(meta: Mapping | None = None) -> dict:
    """MCP ``serverInfo``: the package's stable install name, its display name and its version (the server's own
    name and version for a package without package.json)."""
    meta = hosted_meta() if meta is None else meta
    if not meta.get("version"):
        return {"name": SERVER_NAME, "version": SERVER_VERSION}
    return {"name": meta.get("slug") or SERVER_NAME, "title": meta.get("name") or "Smart Brain package",
            "version": str(meta["version"])}


def _instructions(corpus: Corpus | None) -> str:
    label = version_label()
    head = (f"Knowledge package {label} (built {hosted_meta().get('built') or str(hosted_meta().get('created'))[:10]})"
            ". " if label else "")
    base = head + "Use rag_search for anything about this knowledge base; its passages are the CONTEXT."
    prompt = corpus.identity.get("system_prompt") if corpus is not None else ""
    if prompt:
        return f"{base}\n\n{prompt}"
    return base if label else ""


class LoadingCorpus:
    """A corpus opening in a background thread, so a stdio MCP client gets its handshake at once.

    Decrypting the corpus and loading the embedder takes seconds; clients such as Codex start their turn without
    waiting for a slow server and then see no tools. A failed open ends the process with the code ``main`` uses."""

    def __init__(self, cfg: dict, opener):
        self.cfg = cfg
        self._done = threading.Event()
        self._corpus: Corpus | None = None
        threading.Thread(target=self._open, args=(opener,), daemon=True, name="corpus-open").start()

    def _open(self, opener) -> None:
        try:
            self._corpus = opener()
        except Exception as e:  # noqa: BLE001 — reported the way main() reports a failed start
            code = 3 if type(e).__name__ in ("ContainerError", "WrongPassphraseOrTampered") else 2
            what = "cannot open encrypted package" if code == 3 else "cannot load corpus"
            print(f"{what}: {e}", file=sys.stderr, flush=True)
            sys.stdout.flush()
            os._exit(code)          # waiters are never released: nothing answers from a corpus that did not open
        self._done.set()
        print(f"[rag-mcp] {self._corpus.info()} — corpus ready", file=sys.stderr, flush=True)

    def get(self, timeout: float | None = None) -> Corpus | None:
        """The corpus, waiting up to ``timeout`` seconds (None: until it is open)."""
        self._done.wait(timeout)
        return self._corpus


def _ready(corpus, wait: float | None = None):
    return corpus.get(wait) if isinstance(corpus, LoadingCorpus) else corpus


#: How long initialize/server/discover wait for the corpus so their instructions carry the identity.
INSTRUCTIONS_WAIT = 2.0


def _tools_for(corpus) -> list[dict]:
    """The tool list: rag_get_document only when the package ships original files (known once the corpus is open)."""
    tools = list(_TOOLS) + [_LIST_TOOL]
    opened = _ready(corpus, INSTRUCTIONS_WAIT)
    if opened is not None and opened.originals:
        tools.append(_GET_TOOL)
    return tools + ([_ANSWER_TOOL] if corpus.cfg["answer"] else [])


def _get_document(corpus: Corpus, args: dict, base_url: str) -> list[dict]:
    """rag_get_document's MCP content: the file's details as text, then the file as an embedded resource (or the
    /files/ link when it is over the size limit)."""
    import base64
    row = corpus.find_original(str(args.get("document_id") or ""), str(args.get("filename") or ""))
    if row is None:
        raise ValueError("no original file matches; rag_list_documents shows what is available")
    did = row["document_id"]
    path = f"/files/{did}"
    meta = {k: row.get(k) for k in ("document_id", "filename", "mime_type", "size", "sha256", "date")}
    meta["url"] = (base_url.rstrip("/") + path) if base_url else path
    cap = int(corpus.cfg.get("file_inline_max_mb") or 20) * 1024 * 1024
    if int(row.get("size") or 0) > cap:
        meta["inline"] = False
        meta["note"] = (f"larger than the {cap // (1024 * 1024)} MB inline limit: download it from the url "
                        "(HTTP server, same access token)")
        return [{"type": "text", "text": json.dumps(meta, ensure_ascii=False)}]
    data = corpus.read_original(did)
    meta["inline"] = True
    return [{"type": "text", "text": json.dumps(meta, ensure_ascii=False)},
            {"type": "resource", "resource": {"uri": f"brain://files/{did}", "mimeType": meta["mime_type"]
                                              or "application/octet-stream",
                                              "blob": base64.b64encode(data).decode("ascii")}}]


def mcp_responder(corpus: Corpus | LoadingCorpus, base_url: str = ""):
    """One JSON-RPC message in, its reply (or None for a notification) out; transport-free, used by stdio and HTTP.

    Modern requests (2026-07-28) carry their version in ``params._meta``; their results get ``resultType`` and the
    server's identity in ``_meta``. A legacy client opens with ``initialize`` and gets its own version back if this
    server speaks it."""
    supported = [*MODERN_VERSIONS, *LEGACY_VERSIONS]
    server_info = mcp_server_info()

    def respond(msg: dict) -> dict | None:
        method = msg.get("method")
        mid = msg.get("id")
        params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
        meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else {}
        version = meta.get(META_VERSION)
        modern = version is not None

        def error(code: int, message: str, data: dict | None = None) -> dict:
            err = {"code": code, "message": message, **({"data": data} if data else {})}
            return {"jsonrpc": "2.0", "id": mid, "error": err}

        def ok(result: dict) -> dict:
            if modern:
                result = {"resultType": "complete", **result,
                          "_meta": {"io.modelcontextprotocol/serverInfo": server_info}}
            return {"jsonrpc": "2.0", "id": mid, "result": result}

        if method and method.startswith("notifications/"):
            return None
        if modern and version not in supported:
            return error(UNSUPPORTED_VERSION, "Unsupported protocol version",
                         {"supported": supported, "requested": version})
        instructions = ""
        if method in ("server/discover", "initialize"):
            opened = _ready(corpus, INSTRUCTIONS_WAIT)
            instructions = _instructions(opened)
        if method == "server/discover":
            return ok({"supportedVersions": supported, "capabilities": {"tools": {}},
                       **({"instructions": instructions} if instructions else {}),
                       "ttlMs": 60000, "cacheScope": "private"})
        if method == "initialize":
            requested = params.get("protocolVersion")
            result = {"protocolVersion": requested if requested in LEGACY_VERSIONS else PROTOCOL_VERSION,
                      "capabilities": {"tools": {}}, "serverInfo": server_info}
            if instructions:   # clients that honour server instructions get the identity up front
                result["instructions"] = instructions
            return {"jsonrpc": "2.0", "id": mid, "result": result}
        if method == "ping":
            return ok({})
        if method == "tools/list":
            tools = _tools_for(corpus)
            return ok({"tools": tools, **({"ttlMs": 60000, "cacheScope": "private"} if modern else {})})
        if method == "tools/call":
            name = params.get("name")
            args = params.get("arguments") or {}
            try:
                corpus_now = _ready(corpus)
                if name == "rag_search":
                    out = corpus_now.search(str(args.get("query", "")),
                                        int(args["top_k"]) if args.get("top_k") is not None else None,
                                        str(args.get("source_prefix") or ""),
                                        int(args["dense_extra"]) if args.get("dense_extra") is not None else None)
                elif name == "rag_info":
                    out = corpus_now.info()
                elif name == "rag_profile":
                    out = corpus_now.profile()
                elif name == "rag_list_documents":
                    out = corpus_now.documents(str(args.get("name_contains") or ""), str(args.get("about") or ""),
                                               int(args["limit"]) if args.get("limit") is not None else 200)
                elif name == "rag_get_document" and corpus_now.originals:
                    return ok({"content": _get_document(corpus_now, args, base_url), "isError": False})
                elif name == "rag_answer" and corpus.cfg["answer"]:
                    out = corpus_now.answer(str(args.get("question", "")),
                                            int(args["top_k"]) if args.get("top_k") is not None else None,
                                            include_sources=bool(args.get("include_sources")))
                else:
                    return error(-32602, f"unknown tool {name!r}")
                return ok({"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}], "isError": False})
            except Exception as e:  # noqa: BLE001 — MCP must not die on one call
                return ok({"content": [{"type": "text", "text": f"error: {e}"}], "isError": True})
        if mid is not None:
            return error(-32601, f"method not found: {method}")
        return None
    return respond


def serve_mcp(corpus: Corpus | LoadingCorpus) -> None:
    respond = mcp_responder(corpus)
    opened = _ready(corpus, 0)
    print(f"[rag-mcp] {opened.info() if opened else 'opening the corpus'} — stdio ready", file=sys.stderr, flush=True)
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


#: Never replaced by an update: local settings, the venv and what setup.sh wrote.
UPDATE_KEEP = frozenset({CONFIG_NAME, ".venv", "MCP-ENTRY.txt", "rag-server.log", "rag-server.pid", "passphrase.txt"})
#: The rag.config.json the package was shipped with: an update keeps only the keys the user changed since.
SHIPPED_CONFIG_NAME = "rag.config.shipped.json"
#: Packages from before SHIPPED_CONFIG_NAME: only these local keys survive an update (where and how it runs).
DEPLOY_KEYS = frozenset({"host", "port", "device", "top_k", "max_per_document", "auth_token", "embed_base_url",
                         "embed_model", "embed_api_key", "llm_api_key"})
#: A keyfile next to server.py that update.sh and setup.sh use when no --keyfile is given.
DEFAULT_KEYFILE = "passphrase.txt"


def _widened(current: dict, new: dict, local_cfg: dict, new_cfg: dict) -> list[str]:
    """What installing ``new`` over ``current`` would open up on this running copy (empty: nothing)."""
    out: list[str] = []
    cur, nxt = current.get("policy") or {}, new.get("policy") or {}
    codes = set(new.get("widens") or [])
    if "scope" in codes or (cur.get("scope") == "part" and nxt.get("scope") == "all"):
        out.append(f"documents: more than version {current.get('version')} carried "
                   f"({cur.get('documents', '?')} -> {nxt.get('documents', '?')} documents"
                   + (", now all of the index" if nxt.get("scope") == "all" else "") + ")")
    if "originals" in codes or (not cur.get("originals") and nxt.get("originals")):
        out.append(f"original files: {cur.get('originals') or 'none'} -> {nxt.get('originals') or 'on'} "
                   "(downloadable by anyone who reaches this server)")
    hid_before = bool(local_cfg.get("hide_file_names", cur.get("hide_file_names")))
    if "file_names" in codes or (hid_before and not bool(new_cfg.get("hide_file_names", nxt.get("hide_file_names")))):
        out.append("file names in answers: hidden -> shown")
    if "audience" in codes or (cur.get("audience") == "public" and nxt.get("audience") not in (None, "", "public")):
        out.append(f"used by: {cur.get('audience') or 'public'} -> {nxt.get('audience')}")
    if "encryption" in codes or (cur.get("encrypted") and nxt.get("encrypted") is False):
        out.append("encryption: on -> off")
    return out


def merge_config(local_cfg: dict, old_shipped: dict | None, new_cfg: dict) -> tuple[dict, list[str], list[str]]:
    """``(merged, kept, taken)``: the new version's settings, with the local values the user changed after install.

    A key counts as the user's when it differs from what the old version shipped (``old_shipped``); without that
    record (a package from before it) only where-and-how-it-runs keys (``DEPLOY_KEYS``) are kept. ``taken`` lists
    the keys whose value now comes from the new version and differs from the local one."""
    if old_shipped is not None:
        mine = {k for k, v in local_cfg.items() if k not in old_shipped or old_shipped[k] != v}
    else:
        mine = {k for k in local_cfg if k in DEPLOY_KEYS}
    merged = {**new_cfg, **{k: local_cfg[k] for k in mine}}
    taken = sorted(k for k in new_cfg if k not in mine and k in local_cfg and local_cfg[k] != new_cfg[k])
    return merged, sorted(mine), taken


def _version_key(version) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", str(version or "0").split("-")[0])[:3])


def _read_package_json(root: Path) -> dict:
    try:
        data = json.loads((root / "package.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _extract_archive(archive: Path, dest: Path) -> Path:
    """Unpack a package archive (no absolute paths, no ``..``, no links); return the folder holding server.py."""
    import tarfile
    import zipfile

    def safe(name: str) -> bool:
        parts = Path(name).parts
        return bool(parts) and not name.startswith(("/", "\\")) and ".." not in parts
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest, [n for n in zf.namelist() if safe(n)])
    else:
        with tarfile.open(archive) as tf:
            members = [m for m in tf.getmembers() if safe(m.name) and (m.isfile() or m.isdir())]
            if hasattr(tarfile, "data_filter"):     # Python 3.12+: also strip odd modes/owners
                tf.extractall(dest, members=members, filter="data")
            else:
                tf.extractall(dest, members=members)
    server = sorted(dest.rglob("server.py"), key=lambda p: len(p.parts))
    if not server:
        raise ConfigError(f"{archive} is not a Smart Brain package (no server.py inside)")
    return server[0].parent


def apply_update(pkg_dir: Path, archive: str | Path, *, force: bool = False, passphrase: str | None = None,
                 allow_widen: bool = False) -> dict:
    """Replace this package with a newer version of it, in place.

    The new archive's corpus, server and scripts replace the old ones; ``rag.config.json`` takes the new version's
    settings and keeps the values the user changed after install (``merge_config``); the venv, the keyfile and
    anything else not in the archive stay. A different ``package_id``, or a version not newer than the installed
    one, is refused unless ``force``. A version that opens up more than the installed one (more documents, original
    files, file names shown, a public audience changed, encryption off: ``_widened``) is refused unless
    ``allow_widen``. An encrypted new version is opened with ``passphrase`` first (required unless ``force``), so a
    wrong passphrase never replaces a working package. Returns ``{"from", "to", "package_id", ...}``."""
    import shutil
    import tempfile

    pkg_dir, archive = Path(pkg_dir).resolve(), Path(archive).resolve()
    if not archive.is_file():
        raise ConfigError(f"no such archive: {archive}")
    current = _read_package_json(pkg_dir)
    stage = Path(tempfile.mkdtemp(prefix=".update-", dir=pkg_dir))
    try:
        new_root = _extract_archive(archive, stage)
        new = _read_package_json(new_root)
        if not new.get("package_id"):
            raise ConfigError("the new archive has no package.json: it was exported before package versions")
        if not force:
            if not current.get("package_id"):
                raise ConfigError("this package has no package id (exported before package versions): "
                                  "add --force to replace it anyway")
            if new["package_id"] != current["package_id"]:
                raise ConfigError(f"the archive is another package ({new['package_id']}, this one is "
                                  f"{current['package_id']}): add --force to replace it anyway")
            if _version_key(new.get("version")) <= _version_key(current.get("version")):
                raise ConfigError(f"the archive is version {new.get('version')}, not newer than the installed "
                                  f"{current.get('version')}: add --force to install it anyway")
        local_cfg = {}
        if (pkg_dir / CONFIG_NAME).is_file():
            local_cfg = json.loads((pkg_dir / CONFIG_NAME).read_text(encoding="utf-8"))
        new_cfg = {}
        if (new_root / CONFIG_NAME).is_file():
            new_cfg = json.loads((new_root / CONFIG_NAME).read_text(encoding="utf-8"))
        old_shipped = None
        if (pkg_dir / SHIPPED_CONFIG_NAME).is_file():
            old_shipped = json.loads((pkg_dir / SHIPPED_CONFIG_NAME).read_text(encoding="utf-8"))
        widened = _widened(current, new, local_cfg, new_cfg)
        if widened and not allow_widen:
            raise ConfigError(f"version {new.get('version')} opens up more than the installed "
                              f"{current.get('version')}: " + "; ".join(widened)
                              + ". Nothing was changed. If this copy may serve that, add --allow-widen.")
        enc = new_root / "corpus" / ENC_NAME
        passphrase_note = ""
        if enc.is_file() and not passphrase and not force:
            raise ConfigError("the new version is encrypted: give --keyfile PATH (the passphrase file this server "
                              f"uses, or put it in {DEFAULT_KEYFILE} here) so it is opened before anything changes"
                              + (". Its passphrase was CHANGED at export: use the new one, and give it to this "
                                 "server's keyfile before restarting" if new.get("passphrase_changed") else ""))
        if passphrase and enc.is_file():
            try:
                _load_container_module().ContainerReader(enc, passphrase).close()
            except Exception as e:  # wrong passphrase or a damaged file: keep the working package
                hint = (" (this version was exported with a NEW passphrase: ask whoever exported it)"
                        if new.get("passphrase_changed") else "")
                raise ConfigError(f"the new corpus does not open with this passphrase ({e}){hint}; nothing was "
                                  "changed") from e
            old_enc = pkg_dir / "corpus" / ENC_NAME
            if old_enc.is_file():
                try:
                    _load_container_module().ContainerReader(old_enc, passphrase).close()
                except Exception:  # noqa: BLE001 - the installed corpus has another passphrase
                    passphrase_note = ("the new version has a NEW passphrase: put it in the keyfile this server "
                                       "starts with (setup.sh service, MCP client) before restarting")
        old_req = (pkg_dir / "requirements.txt").read_text(encoding="utf-8") \
            if (pkg_dir / "requirements.txt").is_file() else ""
        replaced = []
        for item in sorted(new_root.iterdir()):
            if item.name in UPDATE_KEEP:
                continue
            target = pkg_dir / item.name
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            elif target.exists() or target.is_symlink():
                target.unlink()
            os.replace(item, target)     # a running update.sh keeps reading its old copy (new inode)
            replaced.append(item.name)
        merged, kept, taken = merge_config(local_cfg, old_shipped, new_cfg)
        (pkg_dir / CONFIG_NAME).write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        new_req = (pkg_dir / "requirements.txt").read_text(encoding="utf-8") \
            if (pkg_dir / "requirements.txt").is_file() else ""
        return {"package_id": new["package_id"], "from": current.get("version"), "to": new.get("version"),
                "name": new.get("name") or "", "slug": new.get("slug") or "", "built": new.get("built") or "",
                "replaced": replaced, "kept_config": kept, "new_config": taken,
                "added_config": sorted(set(new_cfg) - set(local_cfg)), "widened": widened,
                "passphrase_note": passphrase_note, "requirements_changed": old_req != new_req}
    finally:
        shutil.rmtree(stage, ignore_errors=True)


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
    ap.add_argument("--max-per-document", dest="max_per_document", type=int,
                    help="passages of one document in the top results before others get a turn (0 = no cap)")
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
    ap.add_argument("--llm-provider", dest="provider", choices=["openai", "anthropic"],
                    help="API kind of the answer endpoint: openai (any OpenAI-compatible) or anthropic (Claude)")
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
    ap.add_argument("--update", metavar="ARCHIVE",
                    help="replace this package with a newer version of it (keeps rag.config.json and the keyfile)")
    ap.add_argument("--force", action="store_true", help="with --update: accept another package id")
    ap.add_argument("--allow-widen", dest="allow_widen", action="store_true",
                    help="with --update: accept a version that opens up more than the installed one (more "
                         "documents, original files, file names shown, encryption off)")
    return ap


_FLAG_KEYS = ("host", "port", "top_k", "max_per_document", "device", "reranker", "auth_token",
              "embed_model", "embed_base_url", "semantic", "keyword", "answer", "llm_base_url", "llm_model")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    flags = {k: getattr(args, k) for k in _FLAG_KEYS}
    if args.http:  # --http PORT
        flags["port"] = args.http
    env = os.environ
    if args.update:
        try:
            pw = None
            keyfile = args.keyfile or (str(HERE / DEFAULT_KEYFILE) if (HERE / DEFAULT_KEYFILE).is_file()
                                       and not (env.get("RAG_PASSPHRASE") or env.get("RAG_KEYFILE")) else None)
            if keyfile or env.get("RAG_PASSPHRASE") or env.get("RAG_KEYFILE"):
                pw = read_passphrase(keyfile, env)
            done = apply_update(HERE, args.update, force=args.force, passphrase=pw, allow_widen=args.allow_widen)
        except ConfigError as e:
            print(f"update refused: {e}", file=sys.stderr)
            return 2
        print(json.dumps(done, indent=2))
        print(f"updated package {done['name'] or done['package_id']}: version {done['from'] or '(none)'} -> "
              f"{done['to']}" + (f" (built {done['built']})" if done["built"] else ""), file=sys.stderr)
        if done["kept_config"]:
            print("kept your settings: " + ", ".join(done["kept_config"]), file=sys.stderr)
        if done["new_config"]:
            print("new version's settings: " + ", ".join(done["new_config"]), file=sys.stderr)
        for line in done["widened"]:
            print(f"WIDER (allowed with --allow-widen): {line}", file=sys.stderr)
        if done["passphrase_note"]:
            print("NOTE: " + done["passphrase_note"], file=sys.stderr)
        return 0
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
        keyfile = args.keyfile
        if encrypted and not keyfile and not (env.get("RAG_PASSPHRASE") or env.get("RAG_KEYFILE")) \
                and (HERE / DEFAULT_KEYFILE).is_file():
            keyfile = str(HERE / DEFAULT_KEYFILE)      # setup.sh writes it: MCP clients need no flag
        passphrase = read_passphrase(keyfile, env) if encrypted else None
        if args.http is None and not args.query and not args.ask:   # stdio MCP: answer the handshake while it opens
            corpus = LoadingCorpus(cfg, lambda: Corpus(corpus_path, cfg, passphrase))
        else:
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
        try:
            serve_http(corpus, cfg)
        except OSError as e:
            if e.errno in (98, 48, 10048):     # EADDRINUSE on Linux / macOS / Windows
                print(f"port {cfg['port']} is in use: start it on another one, e.g. bash run-http.sh --port "
                      f"{int(cfg['port']) + 1}", file=sys.stderr)
                return 2
            raise
        return 0
    serve_mcp(corpus)
    return 0



#: The package's chat page (GET /) and its script (GET /app.js). No corpus data in either; every call is authenticated.
_CHAT_PAGE = r"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Smart Brain</title><style>
:root{color-scheme:light dark;font:16px system-ui,sans-serif;--bg:#10151b;--panel:#19212b;--line:#303d4b;--accent:#8bd5ca;--text:#edf3f7;--muted:#aab8c4;--err:#f08087}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at top,#20333a,var(--bg) 55%);color:var(--text);min-height:100vh}
main{width:min(880px,100%);margin:auto;padding:clamp(16px,4vw,40px)}h1{margin:0 0 4px;font-size:clamp(26px,5vw,38px)}
.muted{color:var(--muted)}.small{font-size:13px}.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px;margin:14px 0}
summary{cursor:pointer;font-weight:600}label{display:block;font-size:13px;color:var(--muted);margin-top:10px}
input,select,textarea,button{font:inherit}input,select,textarea{width:100%;background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:9px;padding:10px;margin-top:4px}
textarea{min-height:76px;resize:vertical}.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
button{background:var(--accent);color:#10201e;font-weight:700;border:0;border-radius:9px;padding:10px 18px;cursor:pointer}button.ghost{background:transparent;color:var(--muted);border:1px solid var(--line)}
button:disabled{opacity:.6;cursor:wait}#chat{display:grid;gap:12px;margin:14px 0}.turn{border:1px solid var(--line);border-radius:14px;padding:14px;white-space:pre-wrap;line-height:1.55;overflow-wrap:anywhere}
.user{background:#1d2f31;border-color:#537d77;justify-self:end;max-width:90%}.bot{background:var(--panel)}.err{border-color:var(--err);color:var(--err)}
.src{margin-top:10px;font-size:13px;color:var(--muted)}.src a.dl{display:inline-block;margin:2px 0 4px;color:var(--accent)}.src details{margin-top:4px}.src pre{white-space:pre-wrap;font:12px/1.5 ui-monospace,monospace;margin:6px 0 0;padding:8px;border-left:2px solid var(--line)}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}@media(max-width:600px){.row{grid-template-columns:1fr}}
</style><main>
<h1 id="title">Smart Brain</h1><p class="muted" id="about">Ask the documents in this package. Every answer lists the files it came from.</p>
<p class="small muted" id="stats"></p>
<details class="card" id="conn"><summary id="conn-summary">Answer with…</summary>
<label for="kind">Provider</label><select id="kind"></select>
<div id="custom"><div class="row"><div><label for="base">API base URL</label><input id="base" autocomplete="url" placeholder="https://api.openai.com/v1"></div>
<div><label for="model">Model</label><input id="model" autocomplete="off" placeholder="model name"></div></div>
<label for="key">API key</label><input id="key" type="password" autocomplete="off" placeholder="Your provider's key (not needed for a local server)">
<label class="bar"><input id="keep" type="checkbox" style="width:auto"> Keep the key in this browser tab until it is closed</label></div>
<label for="token">Server access token</label><input id="token" type="password" autocomplete="off" placeholder="Only if this server was started with RAG_AUTH_TOKEN">
<p class="small muted">Your question and the passages found are sent to the provider you pick. The key goes to this server with each question and is never stored here. Over a network, use HTTPS.</p>
</details>
<section id="chat" aria-live="polite"></section>
<form id="ask" class="card"><label for="question">Your question</label><textarea id="question" required placeholder="Ask about the documents… (Enter asks, Shift+Enter: new line)"></textarea>
<div class="bar" style="margin-top:10px"><button id="send">Ask</button><button type="button" class="ghost" id="reset">New conversation</button><span class="small muted" id="status"></span></div></form>
<p class="small muted" id="pkg-version"></p>
<script src="/app.js"></script></main></html>"""

_CHAT_JS = r"""'use strict';
const $ = (id) => document.getElementById(id);
const PRESETS = {
  openai: { label: 'OpenAI', api: 'openai', base: 'https://api.openai.com/v1' },
  anthropic: { label: 'Anthropic Claude', api: 'anthropic', base: 'https://api.anthropic.com/v1' },
  openrouter: { label: 'OpenRouter', api: 'openai', base: 'https://openrouter.ai/api/v1' },
  local: { label: 'Local server (Ollama, LM Studio, llama.cpp…)', api: 'openai', base: 'http://127.0.0.1:11434/v1' },
  other: { label: 'Other OpenAI-compatible API', api: 'openai', base: '' },
  search: { label: 'No answer model: show the matching passages', api: '', base: '' },
};
let hideNames = false;
let history = [];
let pkg = null;   // the package's own connection, when it has one
const saved = JSON.parse(localStorage.getItem('sb-conn') || '{}');

function headers() {
  const h = { 'Content-Type': 'application/json' };
  if ($('token').value) h.Authorization = 'Bearer ' + $('token').value;
  return h;
}
function fillKinds() {
  const sel = $('kind');
  sel.innerHTML = '';
  if (pkg) sel.add(new Option(`This package's connection (${pkg.model})`, 'package'));
  for (const [id, p] of Object.entries(PRESETS)) sel.add(new Option(p.label, id));
  sel.value = saved.kind && [...sel.options].some((o) => o.value === saved.kind) ? saved.kind : (pkg ? 'package' : 'openai');
}
function syncKind(fromUser) {
  const k = $('kind').value, custom = k !== 'package' && k !== 'search';
  $('custom').hidden = !custom;
  if (custom && (fromUser || !$('base').value)) $('base').value = (saved.kind === k && saved.base) || PRESETS[k].base;
  if (custom && !fromUser && saved.kind === k && saved.model) $('model').value = saved.model;
  $('key').placeholder = k === 'local' ? 'Usually not needed for a local server' : "Your provider's key";
  $('conn-summary').textContent = 'Answer with: ' + (k === 'package' ? `this package's connection (${pkg.model})`
    : k === 'search' ? 'no model (passages only)' : PRESETS[k].label + ($('model').value ? ' · ' + $('model').value : ''));
}
function remember() {
  const k = $('kind').value;
  localStorage.setItem('sb-conn', JSON.stringify({ kind: k, base: $('base').value.trim(), model: $('model').value.trim() }));
  if ($('keep').checked) sessionStorage.setItem('sb-key', $('key').value); else sessionStorage.removeItem('sb-key');
}
async function showVersion() {   // public: name · version · build date, also before the token is entered
  try {
    const v = await fetch('/version').then((x) => x.ok ? x.json() : {});
    if (v.version) {
      const label = [v.name || 'Smart Brain package', 'version ' + v.version, v.built].filter(Boolean).join(' · ');
      $('pkg-version').textContent = label;
      $('about').title = label;
    }
  } catch (e) { /* the version line is optional */ }
}
async function load() {
  showVersion();
  $('key').value = sessionStorage.getItem('sb-key') || '';
  $('keep').checked = !!$('key').value;
  try {
    const r = await fetch('/health', { headers: headers() });
    if (r.status === 401) { $('conn').open = true; $('status').textContent = 'This server needs its access token: enter it under "Answer with".'; fillKinds(); syncKind(false); return; }
    $('status').textContent = '';
    const h = await r.json();
    const c = h.answer_connection || {};
    pkg = c.configured ? c : null;
    hideNames = !!h.hide_file_names;
    $('stats').textContent = `${h.documents} documents · ${h.chunks} passages · search: ${h.mode}`;
    const p = await fetch('/profile', { headers: headers() }).then((x) => x.ok ? x.json() : {});
    if (p.name) { $('title').textContent = p.name; document.title = p.name; }
  } catch (e) { $('status').textContent = 'Could not reach the server: ' + e.message; }
  fillKinds();
  if (!pkg && !saved.kind) $('kind').value = 'search';   // no connection anywhere: questions still find the passages
  syncKind(false);
  if (!pkg && !saved.model) $('conn').open = true;
}
const esc = (s) => String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
// Answers are Markdown-ish (**bold**, `code`, ### headings, - lists): escape first, then add <strong>/<code> only.
const mdInline = (html) => String(html)
  .replace(/^#{1,4} +(.+)$/gm, '<strong>$1</strong>')
  .replace(/^([ \t]*)[-*] +/gm, '$1\u2022 ')
  .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
  .replace(/`([^`\n]+)`/g, '<code>$1</code>');
function bubble(cls, text) {
  const d = document.createElement('div');
  d.className = 'turn ' + cls;
  if (cls === 'bot') d.innerHTML = mdInline(esc(text)); else d.textContent = text;
  $('chat').append(d); d.scrollIntoView({ block: 'nearest' });
  return d;
}
function sources(box, list, answer) {
  if (!list || !list.length) return;
  // The files the answer names (it cites them as [file]); all retrieved files only when it names none.
  const cited = answer ? list.filter((s) => s.filename && answer.includes(s.filename)) : [];
  if (cited.length) list = cited;
  const wrap = document.createElement('div'); wrap.className = 'src'; wrap.textContent = 'Sources';
  const seen = new Map();
  for (const s of list) {
    if (!seen.has(s.filename)) seen.set(s.filename, { parts: [], doc: '' });
    const e = seen.get(s.filename); e.parts.push(s.preview || '');
    if (s.original && s.document_id) e.doc = s.document_id.split('~')[0];
  }
  for (const [name, { parts, doc }] of seen) {
    const d = document.createElement('details'), sum = document.createElement('summary');
    sum.textContent = `${name} · ${parts.length} passage${parts.length === 1 ? '' : 's'}`; d.append(sum);
    for (const t of parts) { const pre = document.createElement('pre'); pre.textContent = t; d.append(pre); }
    wrap.append(d);
    if (doc) {   // the package ships the original file: download it with the server's token
      const a = document.createElement('a'); a.href = '#'; a.className = 'dl'; a.textContent = '⬇ Download ' + name;
      a.addEventListener('click', (ev) => { ev.preventDefault(); download(doc, name); }); wrap.append(a);
    }
  }
  box.append(wrap);
}
async function download(id, name) {
  try {
    const r = await fetch('/files/' + encodeURIComponent(id), { headers: headers() });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `download failed (${r.status})`);
    const url = URL.createObjectURL(await r.blob()), a = document.createElement('a');
    a.href = url; a.download = name; document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  } catch (e) { $('status').textContent = 'Download failed: ' + e.message; }
}
$('kind').addEventListener('change', () => { syncKind(true); remember(); });
for (const id of ['base', 'model', 'key', 'keep']) $(id).addEventListener('change', () => { remember(); syncKind(false); });
$('token').addEventListener('change', load);
$('reset').addEventListener('click', () => { history = []; $('chat').innerHTML = ''; $('status').textContent = ''; $('question').focus(); });
$('ask').addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = $('question').value.trim(); if (!q) return;
  const k = $('kind').value;
  if (k === 'search') {   // no answer model: the passages that match, with their files unless names are hidden
    bubble('user', q); $('question').value = ''; $('send').disabled = true; $('status').textContent = 'Searching…';
    try {
      const r = await fetch('/search', { method: 'POST', headers: headers(), body: JSON.stringify({ query: q, top_k: 5 }) });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || `request failed (${r.status})`);
      const hits = Array.isArray(d) ? d : (d.results || d.hits || []);
      const text = hits.length ? hits.map((h, i) => (hideNames ? `Passage ${i + 1}` : `${i + 1}. ${h.filename || 'document'}`)
        + '\n' + String(h.text || h.preview || '').trim()).join('\n\n') : 'Nothing in the documents matches that.';
      bubble('bot', text);
      $('status').textContent = 'No answer model is set: these are the passages that match. Pick a provider under "Answer with" for written answers.';
    } catch (err) { bubble('err', err.message); $('question').value = q; $('status').textContent = ''; }
    finally { $('send').disabled = false; $('question').focus(); }
    return;
  }
  if (k !== 'package' && (!$('base').value.trim() || !$('model').value.trim())) { $('conn').open = true; $('status').textContent = 'Pick a provider and enter its model first (or "No answer model" for the passages only).'; return; }
  bubble('user', q); $('question').value = ''; $('send').disabled = true; $('status').textContent = 'Searching and asking…';
  const body = { question: q, history };
  if (k === 'package') body.provider = 'package';
  else Object.assign(body, { provider: PRESETS[k].api, llm_base_url: $('base').value.trim(), llm_model: $('model').value.trim(), llm_api_key: $('key').value });
  try {
    const r = await fetch('/ask', { method: 'POST', headers: headers(), body: JSON.stringify(body) });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || `request failed (${r.status})`);
    const a = bubble('bot', d.answer); sources(a, d.sources, d.answer);
    history.push({ role: 'user', content: q }, { role: 'assistant', content: d.answer }); history = history.slice(-12);
    $('status').textContent = d.weak ? 'The passages found may not cover the whole question.' : '';
  } catch (err) { bubble('err', err.message); $('question').value = q; $('status').textContent = ''; }
  finally { $('send').disabled = false; $('question').focus(); }
});
// Enter asks, Shift+Enter is a new line.
$('question').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); $('ask').requestSubmit(); }
});
load();
"""


if __name__ == "__main__":
    sys.exit(main())
