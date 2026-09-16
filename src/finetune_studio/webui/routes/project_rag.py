"""Project RAG docs inventory — indexed documents + chunks + rebuild.

Indexed documents live in the PortableRAG corpus on disk
(``~/.finetune-studio/rag_corpora/<pid>/``), not in SQLite. There is no
``rag_documents`` / ``rag_chunks`` table; chunk text is in ``chunks.parquet``
and per-doc originals in ``sources/<doc_id>.txt``.
"""
from __future__ import annotations

import logging
import mimetypes
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from finetune_studio.data.rag_portable.source_labels import prettify_source_label

log = logging.getLogger(__name__)
router = APIRouter(tags=["project-rag"])

_CORPORA_ROOT = Path.home() / ".finetune-studio" / "rag_corpora"


def corpus_dir(pid: str) -> Path:
    """Return the on-disk corpus directory for ``pid``."""
    return _CORPORA_ROOT / pid


def project_files_dir(pid: str) -> Path:
    """Return the project's parsed-files directory used as RAG build input."""
    return Path.home() / ".finetune-studio" / "projects" / pid / "files"


def _guess_mime(filename: str) -> str:
    """Guess a MIME type from ``filename``; default to text/plain."""
    mime, _ = mimetypes.guess_type(filename or "")
    return mime or "text/plain"


def _format_ts(ts: float) -> str:
    """Format a unix timestamp for display; empty string if unset."""
    if not ts:
        return ""
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(ts)))
    except (OSError, ValueError, OverflowError, TypeError):
        return ""


def list_indexed_docs(pid: str) -> list[dict[str, Any]]:
    """Inventory of documents indexed into the project's RAG corpus.

    Each item::

        {
          "id": str,           # document_id / sources stem
          "name": str,         # display filename
          "mime": str,         # guessed from name
          "chunks": int,
          "status": str,       # indexed | pending | error
          "last_indexed": str, # display timestamp
          "last_indexed_ts": float,
          "size_bytes": int,
        }

    Does **not** load the embedder — only reads manifest / parquet / sources.
    """
    root = corpus_dir(pid)
    manifest_path = root / "manifest.json"
    sources_dir = root / "sources"
    chunks_path = root / "chunks.parquet"

    has_sources = sources_dir.exists() and any(sources_dir.glob("*.txt"))
    if not manifest_path.exists() and not has_sources:
        return []

    updated_at = 0.0
    if manifest_path.exists():
        try:
            from finetune_studio.data.rag_portable.io import read_json

            m = read_json(manifest_path)
            updated_at = float(m.get("updated_at") or m.get("created_at") or 0.0)
        except Exception:  # noqa: BLE001
            updated_at = 0.0

    # document_id -> {filename, chunk_count}
    by_doc: dict[str, dict[str, Any]] = {}
    if chunks_path.exists():
        try:
            from finetune_studio.data.rag_portable.io import try_import_pandas

            pd = try_import_pandas()
            df = pd.read_parquet(chunks_path)
            if len(df) > 0 and "document_id" in df.columns:
                for doc_id, group in df.groupby("document_id"):
                    did = str(doc_id)
                    fname = ""
                    source_path = ""
                    if "filename" in group.columns and len(group):
                        fname = str(group["filename"].iloc[0] or "")
                    if "source" in group.columns and len(group):
                        source_path = str(group["source"].iloc[0] or "")
                    by_doc[did] = {
                        "filename": prettify_source_label(fname, source_path or None),
                        "chunks": len(group),
                    }
        except Exception as e:  # noqa: BLE001
            log.warning("list_indexed_docs: parquet read failed for %s: %s", pid, e)

    # Merge sources/ so orphaned source files still appear
    if sources_dir.exists():
        for f in sorted(sources_dir.glob("*.txt")):
            did = f.stem
            entry = by_doc.setdefault(
                did, {"filename": did, "chunks": 0}
            )
            entry["size_bytes"] = f.stat().st_size
            entry["mtime"] = f.stat().st_mtime
            if entry["filename"] == did:
                # Prefer a nicer name if we never saw parquet
                entry["filename"] = did

    docs: list[dict[str, Any]] = []

    def _sort_key(item: tuple[str, dict[str, Any]]) -> str:
        return str(item[1].get("filename") or item[0])

    for did, meta in sorted(by_doc.items(), key=_sort_key):
        chunks = int(meta.get("chunks") or 0)
        name = str(meta.get("filename") or did)
        mtime = float(meta.get("mtime") or updated_at or 0.0)
        if chunks > 0:
            status = "indexed"
        elif (sources_dir / f"{did}.txt").exists():
            status = "pending"
        else:
            status = "error"
        docs.append({
            "id": did,
            "name": name,
            "mime": _guess_mime(name),
            "chunks": chunks,
            "status": status,
            "last_indexed": _format_ts(mtime if mtime else updated_at),
            "last_indexed_ts": mtime if mtime else updated_at,
            "size_bytes": int(meta.get("size_bytes") or 0),
        })
    return docs


def list_doc_chunks(pid: str, doc_id: str) -> list[dict[str, Any]]:
    """Return chunk previews for one document (id + first 120 chars of text)."""
    chunks_path = corpus_dir(pid) / "chunks.parquet"
    if not chunks_path.exists():
        return []
    try:
        from finetune_studio.data.rag_portable.io import try_import_pandas

        pd = try_import_pandas()
        df = pd.read_parquet(chunks_path)
    except Exception as e:  # noqa: BLE001
        log.warning("list_doc_chunks failed: %s", e)
        return []
    if len(df) == 0 or "document_id" not in df.columns:
        return []
    rows = df[df["document_id"].astype(str) == str(doc_id)]
    out: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        text = str(row.get("text") or "")
        out.append({
            "chunk_id": str(row.get("id") or ""),
            "chunk_index": int(row.get("chunk_index") or 0),
            "preview": text[:120],
            "chars": len(text),
        })
    out.sort(key=lambda c: c["chunk_index"])
    return out


def total_chunk_count(docs: list[dict[str, Any]]) -> int:
    """Sum of ``chunks`` across indexed-doc dicts."""
    return sum(int(d.get("chunks") or 0) for d in docs)


class RebuildRequest(BaseModel):
    """Optional per-doc id — currently rebuilds the whole project corpus.

    PortableRAG stores one vectors.npy for all chunks, so a true single-doc
    re-embed isn't supported; ``doc_id`` is accepted for API forward-compat
    and logged, then the full project files dir is rebuilt.
    """

    doc_id: str | None = Field(default=None, description="Optional source doc id")
    chunk_size: int = 400
    overlap: int = 80
    embedder: str | None = None
    reset: bool = True


@router.get("/projects/{pid}/rag/docs")
async def rag_docs_inventory(pid: str) -> dict[str, Any]:
    """List indexed documents with per-doc chunk counts and status."""
    from finetune_studio import db

    if not db.get_project(pid):
        raise HTTPException(status_code=404, detail="project not found")
    docs = list_indexed_docs(pid)
    return {
        "docs": docs,
        "document_count": len(docs),
        "chunk_count": total_chunk_count(docs),
    }


@router.get("/projects/{pid}/rag/docs/{doc_id}/chunks")
async def rag_doc_chunks(pid: str, doc_id: str) -> dict[str, Any]:
    """List chunk id + text preview for one indexed document."""
    from finetune_studio import db

    if not db.get_project(pid):
        raise HTTPException(status_code=404, detail="project not found")
    chunks = list_doc_chunks(pid, doc_id)
    return {"doc_id": doc_id, "chunks": chunks, "count": len(chunks)}


@router.post("/projects/{pid}/rag/rebuild")
async def rag_rebuild(pid: str, req: RebuildRequest | None = None) -> dict[str, Any]:
    """Rebuild the project RAG corpus (whole-project scope).

    Optional ``doc_id`` is accepted but the corpus is rebuilt in full —
    PortableRAG has no per-document vector splice path.
    """
    from finetune_studio import db
    from finetune_studio.data.rag_portable import PortableRAG

    if not db.get_project(pid):
        raise HTTPException(status_code=404, detail="project not found")
    body = req or RebuildRequest()
    if body.doc_id:
        log.info(
            "rag rebuild requested for doc_id=%s (full corpus rebuild)",
            body.doc_id,
        )

    files_dir = project_files_dir(pid)
    if not files_dir.exists():
        raise HTTPException(
            status_code=400,
            detail="no parsed files in this project yet — upload & parse first",
        )

    proj = db.get_project(pid)
    name = proj["name"] if proj else pid
    corpus = corpus_dir(pid)
    if body.reset and corpus.exists():
        import shutil

        shutil.rmtree(corpus)

    rag = PortableRAG(corpus)
    try:
        result = rag.build_from_directory(
            source_dir=str(files_dir),
            name=name,
            embedder=body.embedder or "intfloat/multilingual-e5-large",
            chunk_size=body.chunk_size,
            overlap=body.overlap,
            extensions=[".txt"],
        )
    except Exception as e:
        log.exception("RAG rebuild failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Keep chat attachments + RAG page on the same corpus directory.
    try:
        db.ensure_portable_rag(
            pid,
            str(corpus),
            name=name,
            doc_count=int(result.get("documents") or 0),
            chunk_count=int(result.get("chunks") or 0),
        )
    except Exception:
        log.exception("Failed to register project_rags for PortableRAG corpus")

    docs = list_indexed_docs(pid)
    return {
        "ok": True,
        "doc_id": body.doc_id,
        "result": result,
        "docs": docs,
        "document_count": len(docs),
        "chunk_count": total_chunk_count(docs),
    }
