"""Project RAG docs inventory — indexed documents + chunks + rebuild.

Indexed documents live in the PortableRAG corpus on disk
(``~/.finetune-studio/rag_corpora/<pid>/``), not in SQLite. There is no
``rag_documents`` / ``rag_chunks`` table; chunk text is in ``chunks.parquet``
and per-doc originals in ``sources/<doc_id>.txt``.
"""
from __future__ import annotations

import asyncio
import logging
import mimetypes
import re
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from finetune_studio.data.fs.paths import project_files_root, rag_corpus_dir
from finetune_studio.data.rag_portable.source_labels import prettify_source_label

log = logging.getLogger(__name__)
router = APIRouter(tags=["project-rag"])

def corpus_dir(pid: str) -> Path:
    """Return the on-disk corpus directory for ``pid``."""
    return rag_corpus_dir(pid)


def project_files_dir(pid: str) -> Path:
    """Return the project's parsed-files directory used as RAG build input."""
    return project_files_root(pid, create=False)


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

    has_corpus = manifest_path.exists() or chunks_path.exists()
    if not has_corpus:
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

    # Prefer manifest documents_meta when parquet is empty/missing but
    # the current corpus records documents there (still no orphan scan).
    if not by_doc and manifest_path.exists():
        try:
            from finetune_studio.data.rag_portable.io import read_json

            m = read_json(manifest_path)
            for d in (m.get("extra") or {}).get("documents_meta") or []:
                did = str(d.get("document_id") or d.get("id") or "")
                if not did:
                    continue
                by_doc[did] = {
                    "filename": prettify_source_label(
                        str(d.get("filename") or ""),
                        d.get("source"),
                    ),
                    "chunks": 0,
                }
        except Exception as e:  # noqa: BLE001
            log.warning("list_indexed_docs: documents_meta read failed for %s: %s", pid, e)

    # Attach size/mtime from sources/ only for docs already in the current
    # corpus — never invent inventory rows from stale orphan .txt files.
    if sources_dir.exists():
        for did, entry in by_doc.items():
            f = sources_dir / f"{did}.txt"
            if not f.is_file():
                continue
            entry["size_bytes"] = f.stat().st_size
            entry["mtime"] = f.stat().st_mtime

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
    docs = await asyncio.to_thread(list_indexed_docs, pid)
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
    chunks = await asyncio.to_thread(list_doc_chunks, pid, doc_id)
    return {"doc_id": doc_id, "chunks": chunks, "count": len(chunks)}


class ExportConfigPatch(BaseModel):
    """Partial export settings (all optional; see ``RagExportConfig``)."""

    archive_format: str | None = None
    include_models: bool | None = None
    include_reranker: bool | None = None
    reranker_enabled: bool | None = None
    device: str | None = None
    host: str | None = None
    port: int | None = None
    top_k: int | None = None
    encrypt: bool | None = None
    kdf_log_n: int | None = None


class ExportConfigSave(BaseModel):
    scope: str = "studio"            # "studio" defaults | "project" override
    values: ExportConfigPatch = Field(default_factory=ExportConfigPatch)
    clear: bool = False              # project scope only: drop the override


class McpPackageRequest(ExportConfigPatch):
    """Export request: optional overrides + the (never stored) passphrase."""

    name: str | None = None
    passphrase: str | None = Field(default=None, repr=False)


def _patch_values(p: ExportConfigPatch) -> dict[str, Any]:
    return {k: v for k, v in p.model_dump().items() if v is not None}


def _require_project(pid: str) -> dict:
    from finetune_studio import db
    proj = db.get_project(pid)
    if not proj:
        raise HTTPException(status_code=404, detail="project not found")
    return proj


def _export_config_payload(pid: str) -> dict[str, Any]:
    from finetune_studio.data.rag_portable import export_config as ec
    return {
        "defaults": ec.load_defaults().model_dump(),
        "project_override": ec.load_project_override(pid),
        "effective": ec.effective_config(pid).model_dump(),
    }


@router.get("/projects/{pid}/rag/export-config")
async def rag_export_config(pid: str) -> dict[str, Any]:
    """Studio-wide export defaults, this project's override, and the merge."""
    _require_project(pid)
    return _export_config_payload(pid)


@router.put("/projects/{pid}/rag/export-config")
async def rag_export_config_save(pid: str, body: ExportConfigSave) -> dict[str, Any]:
    """Save studio defaults (``scope=studio``) or a project override
    (``scope=project``; ``clear=true`` removes it). Never stores a passphrase."""
    from finetune_studio.data.rag_portable import export_config as ec
    _require_project(pid)
    try:
        if body.scope == "studio":
            ec.save_defaults(_patch_values(body.values))
        elif body.scope == "project":
            ec.save_project_override(pid, None if body.clear else _patch_values(body.values))
        else:
            raise HTTPException(status_code=400, detail="scope must be 'studio' or 'project'")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return _export_config_payload(pid)


def _package_dir(pid: str) -> Path:
    return Path("output") / "projects" / pid / "rag-packages"


_SAFE_PKG_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.(tar\.gz|tar|zip)$")


@router.post("/projects/{pid}/rag/mcp-package")
async def rag_mcp_package(pid: str, body: McpPackageRequest | None = None):
    """Build a hostable, self-installing RAG package and return its metadata.

    Options come from studio defaults < project override < this request.
    Encryption (AES-256-GCM, key from a passphrase) is ON unless the request
    or saved settings turn it off. A blank ``passphrase`` generates one; it is
    returned **once** in this response and never stored. Download the file via
    ``GET .../rag/mcp-package/download?file=<filename>``.
    """
    from fastapi.responses import JSONResponse

    from finetune_studio.data.rag_portable import export_config as ec
    from finetune_studio.data.rag_portable.mcp_package import build_package

    proj = _require_project(pid)
    req = body or McpPackageRequest()
    corpus = corpus_dir(pid)
    if not (corpus / "manifest.json").is_file():
        raise HTTPException(
            status_code=404,
            detail="no corpus yet — build one first (section 2 on this page)",
        )
    try:
        cfg = ec.effective_config(pid, _patch_values(req))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    title = req.name or proj["name"] or pid
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", title).strip(".-")[:48] or "project"
    out_dir = _package_dir(pid)
    suffix = "-with-models" if cfg.include_models else ""
    plain = "" if cfg.encrypt else "-PLAINTEXT"
    ext = "zip" if cfg.archive_format == "zip" else cfg.archive_format
    out_path = out_dir / f"{safe}-rag-package{suffix}{plain}.{ext}"
    try:
        result = await asyncio.to_thread(
            build_package, corpus, out_path, name=title, config=cfg,
            passphrase=(req.passphrase or None))
    except FileNotFoundError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        log.exception("rag package build failed")  # never includes the passphrase
        raise HTTPException(status_code=500, detail=f"package build failed: {type(e).__name__}") from e
    return JSONResponse(
        {
            "filename": result.path.name,
            "download_url": f"/api/projects/{pid}/rag/mcp-package/download?file={result.path.name}",
            "size": result.size,
            "encrypted": result.encrypted,
            # Shown once; only set when generated here.
            "passphrase": result.passphrase,
            "config": cfg.model_dump(),
        },
        headers={"Cache-Control": "no-store"},
    )


@router.get("/projects/{pid}/rag/mcp-package/download")
async def rag_mcp_package_download(pid: str, file: str):
    """Download a previously built package (filename validated; no paths)."""
    from fastapi.responses import FileResponse

    _require_project(pid)
    if not _SAFE_PKG_NAME.match(file):
        raise HTTPException(status_code=400, detail="invalid package filename")
    path = _package_dir(pid) / file
    if not path.is_file():
        raise HTTPException(status_code=404, detail="package not found — export again")
    media = "application/zip" if file.endswith(".zip") else "application/gzip"
    return FileResponse(path=str(path), media_type=media, filename=file,
                        headers={"Cache-Control": "no-store"})


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
        await asyncio.to_thread(shutil.rmtree, corpus)

    # Ensure a project_rags row exists so we have a rag_id for build tracking.
    existing_rags = db.list_rags(pid)
    rag_row = existing_rags[0] if existing_rags else db.create_rag(pid, name)
    rag_id = rag_row["id"]

    build = db.create_rag_build(pid, rag_id)
    db.mark_rag_build_running(build["id"])

    try:
        rag = PortableRAG(corpus)
        result = await asyncio.to_thread(
            rag.build_from_directory,
            source_dir=str(files_dir),
            name=name,
            embedder=body.embedder or "intfloat/multilingual-e5-large",
            chunk_size=body.chunk_size,
            overlap=body.overlap,
            extensions=[".txt"],
        )
    except Exception as e:
        log.exception("RAG rebuild failed")
        db.mark_rag_build_failed(build["id"], error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e

    doc_count = int(result.get("documents") or 0)
    chunk_count = int(result.get("chunks") or 0)
    db.mark_rag_build_done(build["id"], doc_count=doc_count, chunk_count=chunk_count)

    # Keep chat attachments + RAG page on the same corpus directory.
    try:
        db.ensure_portable_rag(
            pid,
            str(corpus),
            name=name,
            doc_count=doc_count,
            chunk_count=chunk_count,
        )
    except Exception:
        log.exception("Failed to register project_rags for PortableRAG corpus")

    docs = await asyncio.to_thread(list_indexed_docs, pid)
    return {
        "ok": True,
        "doc_id": body.doc_id,
        "result": result,
        "docs": docs,
        "document_count": len(docs),
        "chunk_count": total_chunk_count(docs),
    }
