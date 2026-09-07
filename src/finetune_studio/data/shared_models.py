"""Shared model store — one copy of each embedding model, shared across corpora.

When you build a corpus, the embedder + reranker are registered in a shared
store. The corpus just keeps a *reference* (a content-hash key). Only when you
explicitly export-with-model does the model get copied INTO the export bundle
for portability to a different machine.

Layout:
  ~/.finetune-studio/shared_models/
    embedders/<name>@<content-hash>/
      model.safetensors, config.json, tokenizer.json, ...
      META.json (name, hash, size, installed_at, use_count)
    rerankers/<name>@<content-hash>/
      ...

Why content-hash? Two model versions of "all-MiniLM-L6-v2" might have different
configs/weights even at the same HF repo name. Hash the model files so the cache
is exact.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np  # just to keep numpy import path consistent

_ROOT = Path.home() / ".finetune-studio"
SHARED = _ROOT / "shared_models"
EMBEDDERS = SHARED / "embedders"
RERANKERS = SHARED / "rerankers"


def _safe_repo_name(name: str) -> str:
    """HuggingFace 'org/name' becomes 'org__name' for filesystem safety."""
    return name.replace("/", "__")


@dataclass
class ModelRef:
    """A reference to a model in the shared store."""
    name: str            # original HF name, e.g. "intfloat/multilingual-e5-large"
    kind: str            # "embedder" | "reranker"
    short_id: str        # e.g. "intfloat__multilingual-e5-large@abc12345"
    path: str            # absolute local path

    def to_compact(self) -> str:
        return f"shared:{self.kind}:{self.short_id}"


@dataclass
class ModelMeta:
    name: str
    kind: str
    short_id: str
    size_bytes: int = 0
    installed_at: float = 0.0
    use_count: int = 0
    source: str = "hf"   # "hf" or "local"

    def to_json(self) -> dict:
        return asdict(self)


def _dir_for(kind: str) -> Path:
    if kind == "embedder":
        return EMBEDDERS
    if kind == "reranker":
        return RERANKERS
    raise ValueError(f"unknown kind: {kind}")


def _ensure_dirs() -> None:
    SHARED.mkdir(parents=True, exist_ok=True)
    EMBEDDERS.mkdir(parents=True, exist_ok=True)
    RERANKERS.mkdir(parents=True, exist_ok=True)


def content_hash(path: Path) -> str:
    """SHA-256 of the model.safetensors OR the whole directory tree.

    For embedders we use config.json + model.safetensors (sizes are large — fine,
    we want exact match). Falls back to a tree hash if safetensors missing.
    """
    h = hashlib.sha256()
    files_to_hash = []
    sf = path / "model.safetensors"
    if sf.exists():
        files_to_hash.append(sf)
    else:
        # Hash every file in deterministic order
        for p in sorted(path.rglob("*")):
            if p.is_file():
                files_to_hash.append(p)
    for f in files_to_hash:
        # Read in 8MB chunks; sha256 is fast even on big files
        with open(f, "rb") as fh:
            while True:
                chunk = fh.read(8 * 1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
    return h.hexdigest()[:16]  # 16 hex chars is plenty for dedup


def _dir_path(kind: str, name: str, hash16: str) -> Path:
    safe = _safe_repo_name(name)
    return _dir_for(kind) / f"{safe}@{hash16}"


def _model_files(src: Path) -> list[Path]:
    """All files that constitute a sentence-transformers saved model."""
    out = []
    if not src.exists():
        return out
    for p in sorted(src.iterdir()):
        if p.is_file():
            out.append(p)
        elif p.is_dir() and p.name in ("1_Pooling", "2_Normalize"):
            for q in sorted(p.iterdir()):
                if q.is_file():
                    out.append(q)
    return out


def register(model_name: str, kind: str, src_dir: Optional[Path] = None,
             prefer_cached: bool = True) -> ModelRef:
    """Register a model in the shared store.

    If `src_dir` is None, downloads from HuggingFace.
    If `src_dir` is given, copies from there (used by `bundle_models()` style flows).
    Returns a `ModelRef` pointing at the shared dir.
    """
    _ensure_dirs()
    # Compute content hash + figure out the target dir
    if src_dir is not None and src_dir.exists():
        h = content_hash(src_dir)
        target = _dir_path(kind, model_name, h)
        if prefer_cached and target.exists():
            _touch_use(target)
            return ModelRef(name=model_name, kind=kind,
                           short_id=target.name, path=str(target))
        # Need to copy
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir(exist_ok=True)
        for f in _model_files(src_dir):
            shutil.copy2(f, target / f.name)
        size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    else:
        # Download fresh into target dir.
        # Use a temp staging path so the sentence-transformers save() doesn't get
        # fooled by our `pending` placeholder in the path. We then move the dir
        # to its final hashed name.
        from tempfile import mkdtemp
        stage = Path(mkdtemp(prefix=f"fts_{kind}_", dir="/tmp"))
        try:
            if kind == "embedder":
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(model_name, cache_folder="/tmp/hf_cache")
                model.save(str(stage))
            else:  # reranker
                from sentence_transformers import CrossEncoder
                model = CrossEncoder(model_name, max_length=512)
                model.save(str(stage))
            h = content_hash(stage)
            target = _dir_path(kind, model_name, h)
            if target.exists():
                shutil.rmtree(target)
            # Move staged files into target
            target.parent.mkdir(parents=True, exist_ok=True)
            target.mkdir()
            for f in stage.iterdir():
                if f.is_file():
                    shutil.move(str(f), str(target / f.name))
            size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    # Write META.json
    meta = ModelMeta(name=model_name, kind=kind, short_id=target.name,
                     size_bytes=size, installed_at=time.time(), use_count=1)
    (target / "META.json").write_text(json.dumps(meta.to_json(), indent=2),
                                     encoding="utf-8")
    return ModelRef(name=model_name, kind=kind,
                   short_id=target.name, path=str(target))


def _touch_use(path: Path) -> None:
    meta_path = path / "META.json"
    if not meta_path.exists():
        return
    try:
        m = json.loads(meta_path.read_text())
        m["use_count"] = m.get("use_count", 0) + 1
        meta_path.write_text(json.dumps(m, indent=2))
    except Exception:
        pass


def resolve(short_id: str, kind: str) -> Path:
    """Resolve a short_id like 'intfloat__multilingual-e5-large@abc12345' to its dir."""
    p = _dir_for(kind) / short_id
    if not p.exists():
        raise FileNotFoundError(f"shared {kind} '{short_id}' not found at {p}")
    _touch_use(p)
    return p


def stats() -> dict:
    """Global stats for the model store UI."""
    _ensure_dirs()
    out = {"embedders": [], "rerankers": [], "total_size_bytes": 0}
    for kind, base in [("embedder", EMBEDDERS), ("reranker", RERANKERS)]:
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            meta_path = d / "META.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
            out["total_size_bytes"] += size
            out[f"{kind}s"].append({
                "short_id": d.name, "name": meta.get("name", "?"),
                "size_bytes": size, "use_count": meta.get("use_count", 0),
            })
    return out
