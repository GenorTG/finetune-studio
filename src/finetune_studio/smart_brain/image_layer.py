"""The Smart Brain's image layer on disk: pictures, thumbnails, vectors, and the incremental rebuild.

Files in a corpus directory (next to the passage index, never mixed with it)::

    image_assets.json        one record per picture (``ImageAsset.metadata``: document, where, words, hashes)
    image_vectors.npy        float32 [n, dim], row i = record i, L2-normalised (``image_embeddings``)
    image_assets/<id>.<ext>  the picture as extracted (a figure of a document, or the image file itself)
    image_assets/thumbs/<id>.jpg   a small JPEG for result lists (what a package ships without originals)

and ``manifest.extra.image_search`` (model, revisions, dimension, counts, failures).

A build reuses the previous layer: a source whose parsed text did not change keeps its records and files without
being opened again; any picture whose bytes (sha256) were embedded before with the same model keeps its vector. Only
new or changed pictures are embedded.
"""
from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from finetune_studio.smart_brain.image_embeddings import (
    EMBED_BATCH,
    ImageAsset,
    thumbnail,
)

log = logging.getLogger(__name__)

ASSETS_JSON = "image_assets.json"
VECTORS = "image_vectors.npy"
ASSETS_DIR = "image_assets"


@dataclass
class LayerPlan:
    """What a build indexes: pictures extracted now, records kept from the previous layer, extraction failures."""

    fresh: list[ImageAsset] = field(default_factory=list)
    kept: list[dict] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    previous_dir: Path | None = None
    model_id: str = ""


@dataclass
class LayerResult:
    records: list[dict]
    vectors: np.ndarray
    meta: dict[str, Any]
    files: dict[str, bytes | Path]       # relative path -> bytes to write or a file to copy


def read_layer(corpus: Path) -> tuple[list[dict], np.ndarray | None, dict]:
    """``(records, vectors, meta)`` of a corpus' image layer (``[], None, {}`` without one)."""
    corpus = Path(corpus)
    meta: dict = {}
    try:
        meta = (json.loads((corpus / "manifest.json").read_text(encoding="utf-8")).get("extra") or {}) \
            .get("image_search") or {}
    except (OSError, ValueError):
        meta = {}
    if not (corpus / ASSETS_JSON).is_file() or not (corpus / VECTORS).is_file():
        return [], None, meta
    try:
        records = json.loads((corpus / ASSETS_JSON).read_text(encoding="utf-8"))
        vectors = np.load(corpus / VECTORS, allow_pickle=False).astype(np.float32, copy=False)
    except (OSError, ValueError):
        log.warning("image layer of %s unreadable", corpus, exc_info=True)
        return [], None, meta
    if vectors.ndim != 2 or vectors.shape[0] != len(records):
        log.warning("image layer of %s is inconsistent (%s vectors, %d records)", corpus, vectors.shape, len(records))
        return [], None, meta
    return records, vectors, meta


def previous_by_source(corpus: Path, model_id: str) -> dict[str, tuple[str, list[dict]]]:
    """``source_id -> (source_sha, records)`` of the previous layer when it was built with ``model_id`` and its
    files are all still there (a source with a missing file is extracted again)."""
    records, vectors, meta = read_layer(corpus)
    if vectors is None or meta.get("model") != model_id:
        return {}
    out: dict[str, tuple[str, list[dict]]] = {}
    broken: set[str] = set()
    for r in records:
        sid = str(r.get("document_id") or "")
        if not (Path(corpus) / str(r.get("path") or "")).is_file() \
                or not (Path(corpus) / str(r.get("thumbnail") or "")).is_file():
            broken.add(sid)
        out.setdefault(sid, (str(r.get("source_sha") or ""), []))[1].append(r)
    failed = {str(f.get("document_id") or "") for f in meta.get("failed") or []}
    return {k: v for k, v in out.items() if k not in broken and k not in failed}


def compute(plan: LayerPlan, *, device: str = "auto",
            on_progress: Callable[[int, int], None] | None = None) -> LayerResult:
    """Embed what needs it and assemble the new layer (nothing on disk changes here)."""
    from finetune_studio.smart_brain.image_embeddings import (
        cached_image_embedder,
        image_from_bytes,
    )

    old_records, old_vectors, old_meta = read_layer(plan.previous_dir) if plan.previous_dir else ([], None, {})
    same_model = old_vectors is not None and old_meta.get("model") == plan.model_id
    by_sha: dict[str, np.ndarray] = {}
    old_row = {str(r.get("asset_id")): i for i, r in enumerate(old_records)}
    if same_model:
        for i, r in enumerate(old_records):
            by_sha.setdefault(str(r.get("sha256") or ""), old_vectors[i])

    records: list[dict] = []
    rows: list[np.ndarray | None] = []
    files: dict[str, bytes | Path] = {}
    failed = list(plan.failed)
    for r in plan.kept:
        i = old_row.get(str(r.get("asset_id")))
        if i is None or old_vectors is None:
            continue
        records.append(dict(r))
        rows.append(old_vectors[i])
        files[str(r["path"])] = Path(plan.previous_dir) / str(r["path"])
        files[str(r["thumbnail"])] = Path(plan.previous_dir) / str(r["thumbnail"])

    todo: list[tuple[int, ImageAsset]] = []
    for asset in plan.fresh:
        meta = asset.metadata
        try:
            thumb = thumbnail(asset.data)
        except ValueError as e:
            failed.append({"document_id": asset.document_id, "filename": asset.filename, "locator": asset.locator,
                           "error": str(e)})
            continue
        records.append(meta)
        files[meta["path"]] = asset.data
        files[meta["thumbnail"]] = thumb
        vec = by_sha.get(meta["sha256"])
        rows.append(vec)
        if vec is None:
            todo.append((len(rows) - 1, asset))

    embedder = None
    if todo:
        embedder = cached_image_embedder(plan.model_id, device=device)
    total = len(todo)
    if on_progress is not None:
        on_progress(0, total)
    computed: dict[str, np.ndarray] = {}
    pending = [(i, a) for i, a in todo]
    done = 0
    while pending:
        batch, pending = pending[:EMBED_BATCH], pending[EMBED_BATCH:]
        need = [(i, a) for i, a in batch if a.sha256 not in computed]
        images, ok = [], []
        for i, a in need:
            try:
                images.append(image_from_bytes(a.data))
                ok.append((i, a))
            except ValueError as e:
                failed.append({"document_id": a.document_id, "filename": a.filename, "locator": a.locator,
                               "error": str(e)})
        if images:
            vecs = embedder.encode_images(images)
            for (_i, a), v in zip(ok, vecs, strict=True):
                computed[a.sha256] = v
        for i, a in batch:
            rows[i] = computed.get(a.sha256)
        done += len(batch)
        if on_progress is not None:
            on_progress(done, total)

    keep = [k for k, v in enumerate(rows) if v is not None]
    dropped = {records[k]["asset_id"] for k in range(len(records)) if rows[k] is None}
    for k in range(len(records)):
        if rows[k] is None:
            files.pop(records[k]["path"], None)
            files.pop(records[k]["thumbnail"], None)
    records = [records[k] for k in keep]
    if dropped:
        log.warning("%d picture(s) could not be embedded", len(dropped))
    dim = int(rows[keep[0]].shape[0]) if keep else (embedder.dim if embedder is not None else 0)
    vectors = np.stack([rows[k] for k in keep]).astype(np.float32) if keep else np.zeros((0, dim), np.float32)
    if embedder is not None:
        model_meta = embedder.metadata()
    else:   # everything reused: the previous layer's model description still holds
        model_meta = {k: v for k, v in old_meta.items() if k not in ("assets", "embedded", "reused", "failed",
                                                                      "documents")}
    meta = {**model_meta, "assets": len(records), "embedded": len(computed),
            "reused": len(records) - sum(1 for r in records if r["sha256"] in computed),
            "documents": len({r["document_id"] for r in records}), "failed": failed[:500],
            "failed_count": len(failed)}
    return LayerResult(records, vectors, meta, files)


def write(corpus: Path, result: LayerResult | None) -> None:
    """Replace the corpus' image layer with ``result`` (None = remove it). Files are assembled in a side folder and
    swapped in, so the previous layer stays readable until the new one is complete."""
    corpus = Path(corpus)
    target = corpus / ASSETS_DIR
    if result is None or not result.records:
        shutil.rmtree(target, ignore_errors=True)
        for name in (ASSETS_JSON, VECTORS):
            (corpus / name).unlink(missing_ok=True)
        return
    side = corpus / f"{ASSETS_DIR}.new"
    shutil.rmtree(side, ignore_errors=True)
    (side / "thumbs").mkdir(parents=True)
    for rel, src in result.files.items():
        dest = side / Path(rel).relative_to(ASSETS_DIR)
        if isinstance(src, bytes):
            dest.write_bytes(src)
        else:
            shutil.copyfile(src, dest)
    old = corpus / f"{ASSETS_DIR}.old"
    shutil.rmtree(old, ignore_errors=True)
    if target.exists():
        target.rename(old)
    side.rename(target)
    shutil.rmtree(old, ignore_errors=True)
    np.save(corpus / VECTORS, result.vectors)
    (corpus / ASSETS_JSON).write_text(json.dumps(result.records, ensure_ascii=False), encoding="utf-8")


def remove_document(corpus: Path, source_id: str) -> bool:
    """Drop one document's pictures from the layer (a file removed from the project). True if any were there."""
    corpus = Path(corpus)
    records, vectors, _meta = read_layer(corpus)
    if vectors is None:
        return False
    keep = [i for i, r in enumerate(records) if str(r.get("document_id") or "").split("~", 1)[0] != source_id]
    if len(keep) == len(records):
        return False
    kept_set = set(keep)
    for i, r in enumerate(records):
        if i not in kept_set:
            for rel in (r.get("path"), r.get("thumbnail")):
                p = corpus / str(rel or "")
                if rel and p.resolve().is_relative_to((corpus / ASSETS_DIR).resolve()):
                    p.unlink(missing_ok=True)
    kept = [records[i] for i in keep]
    if kept:
        np.save(corpus / VECTORS, vectors[keep])
        (corpus / ASSETS_JSON).write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    else:
        write(corpus, None)
    manifest_path = corpus / "manifest.json"
    if manifest_path.is_file():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        meta = (raw.get("extra") or {}).get("image_search")
        if isinstance(meta, dict):
            if kept:
                meta["assets"] = len(kept)
                meta["documents"] = len({r["document_id"] for r in kept})
            else:
                meta["assets"] = 0
            manifest_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def subset(records: list[dict], vectors: np.ndarray, keep_doc: Callable[[str], bool]) -> tuple[list[dict], np.ndarray]:
    """The records (and vector rows) of the documents ``keep_doc`` accepts (a package's scope)."""
    idx = [i for i, r in enumerate(records) if keep_doc(str(r.get("document_id") or "").split("~", 1)[0])]
    return [records[i] for i in idx], (vectors[idx] if idx else np.zeros((0, vectors.shape[1]), np.float32))


def search(corpus: Path, query: str = "", *, image_data: bytes | None = None, top_k: int = 10,
           group_documents: bool = False, rrf_k: int = 60,
           layer: tuple[list[dict], np.ndarray | None, dict] | None = None) -> dict:
    """Pictures for a text description (50+ languages) or a picture (visual similarity); no passage index needed.

    Returns ``hits`` (each with ``score`` = fused rank score, ``visual_score``, ``words_score``, ``match``),
    ``weak`` (no picture matches: say so instead of showing the nearest), ``mode`` (text / image), ``layer`` (the
    index has one) and, with ``group_documents``, ``documents`` (image -> document). Raises ValueError for a bad
    query or a layer that needs a rebuild, FileNotFoundError when the model is not on this machine."""
    from finetune_studio.smart_brain.standalone_server import (
        image_documents,
        rank_images,
    )

    query = (query or "").strip()
    if not query and not image_data:
        raise ValueError("enter a text query or upload an image")
    if not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50")
    mode = "image" if image_data else "text"
    records, vectors, meta = layer if layer is not None else read_layer(corpus)
    if vectors is None or not records:
        return {"hits": [], "weak": True, "mode": mode, "documents": [], "layer": False}
    from finetune_studio.smart_brain.image_embeddings import (
        cached_image_embedder,
        image_from_bytes,
    )

    embedder = cached_image_embedder(meta.get("model") or None)
    if vectors.shape[1] != embedder.dim:
        raise ValueError("the image embedding model does not match this index; rebuild the Smart Brain")
    qvec = embedder.encode_images([image_from_bytes(image_data)])[0] if image_data else embedder.encode_texts(query)
    hits, weak = rank_images(records, vectors, qvec, query=query, by_image=bool(image_data), top_k=top_k, rrf_k=rrf_k)
    by_id = {r.get("asset_id"): r for r in records}
    for h in hits:   # the studio links the library file (open / download the original)
        if by_id.get(h["asset_id"], {}).get("file_id"):
            h["file_id"] = by_id[h["asset_id"]]["file_id"]
    out = {"hits": hits, "weak": weak, "mode": mode, "layer": True}
    if group_documents:
        out["documents"] = image_documents(hits)
    return out
