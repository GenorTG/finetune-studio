"""Routes for the per-project RAG.

Mounted under ``/api/projects``. Endpoints (all keyed by ``{pid}``):

  GET    /{pid}/rag                  -- status + manifest summary
  GET    /{pid}/rag/settings         -- settings (embedder, reranker, hybrid, etc.)
  POST   /{pid}/rag/settings         -- update settings (no rebuild)
  POST   /{pid}/rag/build            -- (re)build corpus from project's parsed files
  POST   /{pid}/rag/quick            -- promote unparsed files, then build
  GET    /{pid}/rag/build/status     -- one-shot build progress
  GET    /{pid}/rag/build/progress   -- SSE build progress
  POST   /{pid}/rag/rebuild-vectors  -- re-embed with current settings
  GET    /{pid}/rag/sources          -- list corpus sources
  DELETE /{pid}/rag/sources[/{id}]   -- remove one / all sources
  POST   /{pid}/rag/search           -- {query, top_k, hybrid?, rerank?, rerank_top_n?}
  POST   /{pid}/rag/bundle           -- export encrypted .ftsrag corpus bundle
  GET    /{pid}/rag/bundle/download  -- download it
  POST   /{pid}/rag/import           -- upload a bundle (+passphrase if .ftsrag)
  POST   /{pid}/rag/chat             -- {messages, top_k?} -> {reply, sources, hits}
  GET    /shared-models/stats        -- shared model pool stats
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import time
from pathlib import Path

from fastapi import APIRouter, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from finetune_studio.data.fs.paths import (
    project_dir,
    project_files_root,
    rag_corpus_dir,
)
from finetune_studio.data.rag_portable.constants import DEFAULT_TOP_K
from finetune_studio.webui.engine_guard import ENGINE_LOCK
from finetune_studio.webui.live_sse import sse_data, sse_response

log = logging.getLogger(__name__)
router = APIRouter()

def _corpus_dir(pid: str) -> Path:
    return rag_corpus_dir(pid)


def _project_404(pid: str) -> JSONResponse | None:
    """Return a 404 response when the project does not exist, else None.

    This module answers errors with ``JSONResponse`` (see the corpus 404s
    below), so the guard matches that style. Called before any corpus
    directory is touched — otherwise a bad pid gets a real ``corpus_dir``
    path back with a 200.
    """
    from finetune_studio import db
    if not db.get_project(pid):
        return JSONResponse({"error": "project not found"}, status_code=404)
    return None


# ── DTOs ────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    top_k: int = DEFAULT_TOP_K
    hybrid: bool | None = None
    rerank: bool | None = None
    rerank_top_n: int | None = None


class ChatRequest(BaseModel):
    messages: list[dict]
    system_prompt: str = ""
    top_k: int = DEFAULT_TOP_K
    temperature: float = 0.0
    max_tokens: int = 400


class BuildRequest(BaseModel):
    chunk_size: int = 400
    overlap: int = 80
    embedder: str | None = None
    reset: bool = False


class RebuildVectorsRequest(BaseModel):
    embedder: str | None = None


class SettingsPatch(BaseModel):
    embedder: str | None = None
    reranker: str | None = None
    rerank_enabled: bool | None = None
    rerank_top_n: int | None = None
    hybrid_enabled: bool | None = None
    rrf_k: int | None = None


# ── Routes ──────────────────────────────────────────────────────────────

@router.get("/{pid}/rag")
async def rag_status(pid: str):
    missing = _project_404(pid)
    if missing is not None:
        return missing
    from finetune_studio.data.rag_portable import PortableRAG
    rag = PortableRAG(_corpus_dir(pid))
    if not rag.exists():
        return {"exists": False, "corpus_dir": str(_corpus_dir(pid))}
    m = json.loads((_corpus_dir(pid) / "manifest.json").read_text())
    return {
        "exists": True,
        "corpus_dir": str(_corpus_dir(pid)),
        "name": m["name"],
        "documents": m["documents"],
        "chunks": m["chunks"],
        "embedder": m["embedding_model"]["name"],
        "embedder_dim": m["embedding_model"]["dim"],
        "reranker": m["rag_settings"]["reranker"],
        "rerank_enabled": m["rag_settings"]["rerank_enabled"],
        "rerank_top_n": m["rag_settings"]["rerank_top_n"],
        "hybrid_enabled": m["rag_settings"]["hybrid_enabled"],
        "rrf_k": m["rag_settings"]["rrf_k"],
        "created_at": m["created_at"],
        "updated_at": m["updated_at"],
    }


@router.get("/{pid}/rag/settings")
async def rag_get_settings(pid: str):
    from finetune_studio.data.rag_portable import PortableRAG
    rag = PortableRAG(_corpus_dir(pid))
    if not rag.exists():
        # Not an error: the page asks before any index exists.
        return {"exists": False}
    m = rag.load().manifest
    return {
        "exists": True,
        "embedder": m.embedding_model.name,
        "reranker": m.rag_settings.reranker,
        "rerank_enabled": m.rag_settings.rerank_enabled,
        "rerank_top_n": m.rag_settings.rerank_top_n,
        "hybrid_enabled": m.rag_settings.hybrid_enabled,
        "rrf_k": m.rag_settings.rrf_k,
        "chunk_settings": {
            "size": m.chunk_settings.size,
            "overlap": m.chunk_settings.overlap,
            "splitter": m.chunk_settings.splitter,
        },
    }


@router.post("/{pid}/rag/settings")
async def rag_patch_settings(pid: str, req: SettingsPatch):
    """Update settings. Changes to embedder/rerank_top_n don't break anything,
    but changing embedder will make existing vectors invalid → caller must
    rebuild vectors afterwards."""
    from finetune_studio.data.rag_portable import (
        Manifest,
        PortableRAG,
        write_json,
    )
    rag = PortableRAG(_corpus_dir(pid))
    if not rag.exists():
        return JSONResponse({"error": "no corpus"}, status_code=404)
    m = Manifest.from_json(json.loads((_corpus_dir(pid) / "manifest.json").read_text()))
    if req.embedder is not None:
        m.rag_settings.embedder = req.embedder
    if req.reranker is not None:
        m.rag_settings.reranker = req.reranker
    if req.rerank_enabled is not None:
        m.rag_settings.rerank_enabled = req.rerank_enabled
    if req.rerank_top_n is not None:
        m.rag_settings.rerank_top_n = req.rerank_top_n
    if req.hybrid_enabled is not None:
        m.rag_settings.hybrid_enabled = req.hybrid_enabled
    if req.rrf_k is not None:
        m.rag_settings.rrf_k = req.rrf_k
    m.updated_at = time.time()
    write_json(_corpus_dir(pid) / "manifest.json", m.to_json())
    return {"ok": True, "updated": req.model_dump(exclude_none=True)}


@router.post("/{pid}/rag/build")
async def rag_build(pid: str, req: BuildRequest):
    """(Re)build the project's RAG from the project's files dir.

    The corpus lives at `$FTS_ROOT/rag_corpora/<pid>/`. We source
    chunks from the project's structured filesystem (parsed.txt files), NOT
    from raw uploads — these are the OCR/parsed outputs that the data-prep
    pipeline already produced.
    """
    from finetune_studio.data.rag_portable import PortableRAG

    missing = _project_404(pid)
    if missing is not None:
        return missing
    # Find source dir: project's files/<sha>/*.txt
    project_files_dir = project_files_root(pid, create=False)
    if not project_files_dir.exists():
        return JSONResponse({"error": "no parsed files in this project yet — upload & parse first"},
                            status_code=400)
    # Use the project name as corpus name if available
    from finetune_studio import db
    name = db.get_project(pid)["name"]

    corpus = _corpus_dir(pid)
    if req.reset and corpus.exists():
        await asyncio.to_thread(shutil.rmtree, corpus)
    rag = PortableRAG(corpus)

    # Count the .txt files we are about to feed — gives the UI a real
    # "queued N files" message instead of the '?' fallback while the
    # background embedder spins up (which can take minutes on CPU).
    def _count_txt() -> tuple[int, int]:
        txt_files = sorted(project_files_dir.rglob("*.txt"))
        return (sum(1 for f in txt_files if f.is_file()),
                sum(f.stat().st_size for f in txt_files))

    queued_files, queued_chars = await asyncio.to_thread(_count_txt)

    # The request stays open until the build finishes (embedding 45 small
    # files takes ~30s), but runs in a worker thread so the event loop can
    # still serve /build/progress. A thread inherits HF_HOME from the systemd
    # unit; detached background tasks did not.
    try:
        result = await asyncio.to_thread(
            rag.build_from_directory,
            source_dir=str(project_files_dir),
            name=name,
            embedder=req.embedder or "intfloat/multilingual-e5-large",
            chunk_size=req.chunk_size,
            overlap=req.overlap,
            extensions=[".txt"],
        )
        log.info("RAG build complete: %s", result)
        # Register/update project_rags so chat attachments point at this corpus.
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
    except Exception as e:
        log.exception("RAG build failed")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    # Build is synchronous: the corpus is already written when we return.
    # ``building: false`` keeps the UI from claiming an in-flight embed.
    return {
        "ok": True,
        "building": False,
        "corpus_dir": str(corpus),
        "queued_files": queued_files,
        "queued_chars": queued_chars,
        "documents": int(result.get("documents") or 0),
        "chunks": int(result.get("chunks") or 0),
        "vector_dim": result.get("vector_dim"),
        "skipped": int(result.get("skipped") or 0),
        "reset": bool(req.reset),
    }


class QuickRequest(BaseModel):
    file_ids: list[str] | None = None
    promote_missing: bool = True


@router.post("/{pid}/rag/quick")
async def rag_quick(pid: str, req: QuickRequest):
    """⚡ Quick index: promote every not-yet-parsed library file into QA
    sources, then build the corpus from all parsed text — one click, no
    leaving the RAG page. ``file_ids`` restricts the promote step.
    """
    from finetune_studio import db
    from finetune_studio.data import project_filesystem as pfs
    from finetune_studio.data.fs import file_library as fl
    from finetune_studio.data.fs.qa import promote_file_library_upload

    if not db.get_project(pid):
        return JSONResponse({"error": "project not found"}, status_code=404)

    promoted: list[dict] = []
    failed: list[dict] = []
    if req.promote_missing:
        sources = pfs.list_qa_sources(pid)
        known_paths = set()
        for s in sources:
            known_paths.add(str(s.get("data_path") or ""))
            known_paths.add(str(s.get("path") or ""))
        for f in fl.list_files(pid):
            if req.file_ids and f["id"] not in req.file_ids:
                continue
            try:
                versions = fl.list_versions(pid, f["id"])
            except Exception:  # noqa: BLE001
                log.warning("rag quick: no versions for file %s", f.get("original_name"))
                continue
            raw_path = str(versions[0].get("raw_path") or "") if versions else ""
            if not raw_path or raw_path in known_paths:
                continue
            try:
                src = promote_file_library_upload(
                    pid, f["id"], mime_type=f.get("mime_type") or "",
                    filename=f.get("original_name"),
                )
                promoted.append({"file_id": f["id"], "name": f.get("original_name"),
                                 "source_id": src.get("id"), "status": src.get("status")})
                if src.get("status") != "ready":
                    failed.append({"name": f.get("original_name"),
                                   "error": src.get("error") or "parse incomplete"})
            except Exception as e:  # noqa: BLE001
                failed.append({"name": f.get("original_name"), "error": str(e)})

    build = await rag_build(pid, BuildRequest())
    if isinstance(build, JSONResponse):
        build_dict = {"ok": False, **json.loads(build.body)}
    else:
        build_dict = build
    return {"ok": bool(build_dict.get("ok", True)),
            "promoted": len(promoted), "promoted_files": promoted,
            "failed": failed, "build": build_dict}


def _rag_build_snapshot(pid: str, *, elapsed_s: int = 0) -> dict:
    """One progress snapshot for SSE frames and the /build/status poll."""
    corpus = _corpus_dir(pid)
    project_files_dir = project_files_root(pid, create=False)
    total_files = (
        sum(1 for f in project_files_dir.rglob("*.txt") if f.is_file())
        if project_files_dir.exists() else 0
    )
    sources_dir = corpus / "sources"
    manifest_exists = (corpus / "manifest.json").exists()
    chunks_path = corpus / "chunks.parquet"
    files_done = 0
    chunks_count = 0
    if manifest_exists:
        try:
            m = json.loads((corpus / "manifest.json").read_text())
            files_done = int(m.get("documents", 0) or 0)
            chunks_count = int(m.get("chunks", 0) or 0)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            files_done = 0
            chunks_count = 0
    elif sources_dir.exists():
        # Mid-build fallback only — prefer manifest once written.
        files_done = sum(1 for _ in sources_dir.glob("*.txt"))
    if manifest_exists:
        phase = "done"
    elif chunks_path.exists():
        phase = "embedding"
    elif files_done > 0:
        phase = "chunking"
    else:
        phase = "queued"
    return {
        "phase": phase,
        "files_done": files_done,
        "files_total": total_files,
        "chunks": chunks_count,
        "elapsed_s": elapsed_s,
    }


@router.get("/{pid}/rag/build/status")
async def rag_build_status(pid: str):
    """One-shot corpus-build progress (SSE silent fallback for /build/progress)."""
    missing = _project_404(pid)
    if missing is not None:
        return missing
    return _rag_build_snapshot(pid)


@router.get("/{pid}/rag/build/progress")
async def rag_build_progress(pid: str):
    """Server-Sent Events stream that reports corpus build progress.

    Emits JSON events:
        {"phase":"queued|chunking|embedding|done|timeout",
         "files_done":N,"files_total":M,"chunks":K,"elapsed_s":S}
    Stream terminates once ``phase`` is ``done`` or ``timeout``,
    or after 10 min. Prefer this over polling; clients may fall back to
    ``GET .../rag/build/status`` via fts.subscribe (fallbackMs ≥ 5s).
    """
    missing = _project_404(pid)
    if missing is not None:
        return missing

    async def gen():
        start = asyncio.get_event_loop().time()
        deadline = start + 600  # 10 min
        try:
            while asyncio.get_event_loop().time() < deadline:
                elapsed = int(asyncio.get_event_loop().time() - start)
                payload = _rag_build_snapshot(pid, elapsed_s=elapsed)
                yield sse_data(payload)
                if payload["phase"] == "done":
                    return
                await asyncio.sleep(1.0)
            yield sse_data({
                "phase": "timeout",
                "files_total": _rag_build_snapshot(pid).get("files_total", 0),
                "files_done": 0,
                "chunks": 0,
                "elapsed_s": 600,
            })
        except asyncio.CancelledError:
            return

    return sse_response(gen())


@router.post("/{pid}/rag/rebuild-vectors")
async def rag_rebuild_vectors(pid: str, req: RebuildVectorsRequest):
    """Re-embed with (possibly new) embedder. Loads + replaces vectors.npy."""
    from finetune_studio.data.rag_portable import Manifest, PortableRAG
    rag = PortableRAG(_corpus_dir(pid))
    if not rag.exists():
        return JSONResponse({"error": "no corpus"}, status_code=404)
    embedder = req.embedder
    if not embedder:
        # ``POST /rag/settings`` writes a changed embedder to
        # ``rag_settings.embedder`` only (see rag_patch_settings) — the
        # active vectors still carry the OLD embedder in
        # ``embedding_model.name`` until a rebuild runs. Without this
        # fallback, rebuild_vectors(embedder=None) defaults to
        # ``embedding_model.name`` (store.py), so a settings-patched
        # embedder change is silently dropped on the very rebuild call
        # the settings endpoint's docstring says is required to apply it.
        try:
            m = Manifest.from_json(json.loads((_corpus_dir(pid) / "manifest.json").read_text()))
            embedder = m.rag_settings.embedder or None
        except Exception:  # noqa: BLE001
            embedder = None
    try:
        result = await asyncio.to_thread(rag.rebuild_vectors, embedder=embedder)
        return {"ok": True, **result}
    except Exception as e:
        log.exception("rebuild failed")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@router.get("/{pid}/rag/sources")
async def rag_list_sources(pid: str):
    """List all sources in the corpus."""
    from finetune_studio.data.rag_portable import PortableRAG
    rag = PortableRAG(_corpus_dir(pid))
    if not rag.exists():
        return JSONResponse({"error": "no corpus"}, status_code=404)
    try:
        q = rag.load()
        return {"sources": q.list_sources()}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=500)


@router.delete("/{pid}/rag/sources/{source_id}")
async def rag_delete_source(pid: str, source_id: str):
    """Remove a single source from the corpus."""
    from finetune_studio.data.rag_portable import PortableRAG
    rag = PortableRAG(_corpus_dir(pid))
    if not rag.exists():
        return JSONResponse({"error": "no corpus"}, status_code=404)
    try:
        rag.remove_source(source_id)
        return {"ok": True, "removed": source_id}
    except Exception as e:
        log.exception("remove source failed")
        return JSONResponse({"error": str(e)}, status_code=400)


@router.delete("/{pid}/rag/sources")
async def rag_clear_sources(pid: str):
    """Clear source text and all search indexes while retaining the corpus directory."""
    from finetune_studio.data.rag_portable import PortableRAG
    rag = PortableRAG(_corpus_dir(pid))
    if not rag.exists():
        return JSONResponse({"error": "no corpus"}, status_code=404)
    try:
        rag.clear_sources()
        return {"ok": True}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=400)


_STOP = frozenset(["what", "which", "when", "where", "whom", "whose", "does", "that", "this", "with", "from", "have", "about", "tell", "there", "their", "into"])


def weak_match(query: str, hits: list[dict]) -> bool:
    """True when the retrieved text covers too few of the query's content words."""
    terms = {w for w in re.findall(r"[a-z0-9]{4,}", query.lower()) if w not in _STOP}
    if not terms or not hits:
        return False
    blob = " ".join(str(h.get("text", "")) for h in hits).lower()
    found = sum(1 for t in terms if t in blob)
    return found / len(terms) < 0.6


@router.post("/{pid}/rag/search")
async def rag_search(pid: str, req: SearchRequest):
    from finetune_studio.data.rag_portable import PortableRAG
    rag = PortableRAG(_corpus_dir(pid))
    if not rag.exists():
        return JSONResponse({"error": "no corpus — build first"}, status_code=400)
    try:
        q = rag.load()
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"load failed: {e}"}, status_code=500)
    hits = q.search(req.query, top_k=req.top_k,
                    hybrid=req.hybrid, rerank=req.rerank,
                    rerank_top_n=req.rerank_top_n)
    weak = weak_match(req.query, hits)
    return {"hits": hits, "count": len(hits), "query": req.query, "weak": weak,
            "note": "These passages share few words with your question; the corpus may not cover it." if weak else ""}


class BundleRequest(BaseModel):
    name: str | None = None
    passphrase: str | None = None  # blank -> generated, returned once, never stored


_SAFE_BUNDLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.ftsrag$")


def _bundle_dir(pid: str) -> Path:
    return project_dir(pid) / "rag-bundles"


@router.post("/{pid}/rag/bundle")
async def rag_bundle(pid: str, body: BundleRequest | None = None):
    """Export the corpus as an encrypted ``.ftsrag`` file inside the project dir.

    AES-256-GCM, key derived from a passphrase that is returned once (when
    generated here) and never stored. Download it with ``GET .../rag/bundle/download``.
    """
    from finetune_studio.data.rag_portable import secure_bundle as sb

    missing = _project_404(pid)
    if missing is not None:
        return missing
    req = body or BundleRequest()
    corpus = _corpus_dir(pid)
    if not (corpus / "manifest.json").is_file():
        return JSONResponse({"error": "no corpus to bundle"}, status_code=404)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", req.name or "").strip(".-")[:48] or f"{pid}-corpus"
    out = _bundle_dir(pid) / f"{stem}{sb.BUNDLE_EXT}"
    try:
        path, generated = await asyncio.to_thread(
            sb.export_secure_bundle, corpus, out, req.passphrase or None)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:
        log.exception("bundle export failed")  # never includes the passphrase
        return JSONResponse({"error": f"bundle export failed: {type(e).__name__}"}, status_code=500)
    return JSONResponse(
        {"filename": path.name, "size": path.stat().st_size, "encrypted": True,
         "passphrase": generated,
         "download_url": f"/api/projects/{pid}/rag/bundle/download?file={path.name}"},
        headers={"Cache-Control": "no-store"},
    )


@router.get("/{pid}/rag/bundle/download")
async def rag_bundle_download(pid: str, file: str):
    from fastapi.responses import FileResponse

    missing = _project_404(pid)
    if missing is not None:
        return missing
    if not _SAFE_BUNDLE.match(file):
        return JSONResponse({"error": "invalid bundle filename"}, status_code=400)
    path = _bundle_dir(pid) / file
    if not path.is_file():
        return JSONResponse({"error": "bundle not found — export again"}, status_code=404)
    return FileResponse(path=str(path), media_type="application/octet-stream",
                        filename=file, headers={"Cache-Control": "no-store"})


@router.post("/{pid}/rag/import")
async def rag_import(pid: str, file: UploadFile, overwrite: bool = False,
                     passphrase: str = Form("")):
    """Import a corpus bundle: encrypted ``.ftsrag`` (needs ``passphrase``) or a
    legacy ``.tar/.tar.gz/.zip``.

    The upload is staged inside the project directory (never the system temp
    dir) and removed afterwards. The corpus is replaced only when
    ``overwrite=true`` or none exists. Returns the same stats as
    :func:`rag_status`.
    """
    from finetune_studio.data.rag_portable import PortableRAG
    from finetune_studio.data.rag_portable import secure_bundle as sb
    from finetune_studio.data.rag_portable.rag_container import ContainerError

    missing = _project_404(pid)
    if missing is not None:
        return missing
    filename = file.filename or "bundle"
    name_lower = filename.lower()
    # Path.suffix only sees the last dot, so "x.tar.gz" reports ".gz" —
    # match compound suffixes explicitly.
    if name_lower.endswith(sb.BUNDLE_EXT):
        suffix = sb.BUNDLE_EXT
    elif name_lower.endswith((".tar.gz", ".tgz")):
        suffix = ".tar.gz"
    elif name_lower.endswith(".tar"):
        suffix = ".tar"
    elif name_lower.endswith(".zip"):
        suffix = ".zip"
    else:
        return JSONResponse(
            {"error": f"unsupported archive format: {Path(filename).suffix.lower()}"},
            status_code=400,
        )

    staging = project_dir(pid) / "rag-import-staging"
    staging.mkdir(parents=True, exist_ok=True)
    tmp_path = staging / f"upload-{time.time_ns()}{suffix}"

    def _stage() -> None:
        with open(tmp_path, "wb") as tmp:
            shutil.copyfileobj(file.file, tmp)

    await asyncio.to_thread(_stage)

    rag = PortableRAG(_corpus_dir(pid))
    try:
        if suffix == sb.BUNDLE_EXT:
            if not passphrase:
                return JSONResponse({"error": "this bundle is encrypted; a passphrase is required"},
                                    status_code=400)
            if rag.exists() and not overwrite:
                raise FileExistsError(
                    f"Corpus already exists at {rag.dir}; pass overwrite=True to replace.")
            await asyncio.to_thread(sb.import_secure_bundle, tmp_path, passphrase,
                                    rag.dir, overwrite=True)
            stats = rag.load().manifest.to_json()
            stats = {"documents": stats.get("documents", 0), "chunks": stats.get("chunks", 0)}
        else:
            stats = rag.import_bundle(tmp_path, overwrite=overwrite)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except FileExistsError as e:
        return JSONResponse({"error": str(e)}, status_code=409)
    except ContainerError as e:  # wrong passphrase / tampered / not a bundle
        return JSONResponse({"error": str(e)}, status_code=400)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        log.exception("bundle import failed")
        return JSONResponse({"error": f"bundle import failed: {type(e).__name__}"}, status_code=500)
    finally:
        tmp_path.unlink(missing_ok=True)

    # Persist a fresh activity-feed row so the user sees this in the drawer.
    try:
        from finetune_studio import db
        db.record_activity_event(
            kind="rag_build",
            operation=f"POST /api/projects/{pid}/rag/import",
            method="POST",
            path=f"/api/projects/{pid}/rag/import",
            project_id=pid,
            status="done",
            http_status=200,
            message=(
                f"Imported RAG bundle: {stats.get('documents', 0)} docs, "
                f"{stats.get('chunks', 0)} chunks"
            ),
        )
    except Exception:
        log.exception("failed to record import activity event")

    return stats


@router.get("/shared-models/stats")
async def shared_model_stats():
    """Dashboard stats for the shared model pool. Shows which models are stored,
    how much disk they use, and how often they're referenced by corpora."""
    from finetune_studio.data.shared_models import stats as sm_stats
    return sm_stats()


def _active_n_ctx() -> int:
    """Window of the active chat provider (0 when none is loaded or the provider has no known window)."""
    from finetune_studio.models.manager import get_manager

    active = get_manager().active() or {}
    return int(active.get("n_ctx") or 0)


@router.post("/{pid}/rag/chat")
async def rag_chat(pid: str, req: ChatRequest):
    """RAG-augmented chat. Retrieves top-k from project corpus, prepends to
    the conversation, runs the loaded model.

    Returns: {reply, sources: [{filename, score, chunk_text}], messages_full}
    """
    from finetune_studio.data.rag_portable import PortableRAG
    from finetune_studio.data.rag_portable.prompt import (
        build_messages,
        context_char_budget,
    )
    from finetune_studio.webui.app import inference_engine

    rag = PortableRAG(_corpus_dir(pid))
    if not rag.exists():
        return JSONResponse({"error": "no corpus — build first"}, status_code=400)
    try:
        q = rag.load()
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"load failed: {e}"}, status_code=500)

    # Use the last user message as the search query
    user_messages = [m for m in req.messages if m.get("role") == "user"]
    if not user_messages:
        return JSONResponse({"error": "no user message to search for"}, status_code=400)
    last_user = user_messages[-1]["content"]

    hits = q.search(last_user, top_k=req.top_k)
    n_ctx = inference_engine.n_ctx if inference_engine.model is not None else (_active_n_ctx() or None)
    history_chars = sum(len(str(m.get("content", ""))) for m in req.messages) + len(req.system_prompt or "")
    context = q.format_context(hits, max_chars=context_char_budget(
        n_ctx, max_new_tokens=req.max_tokens, history_chars=history_chars))

    # Layout lives in rag_portable.prompt — shared with the context-grounded
    # training rows (data/prep/grounding.py); do not inline it here again.
    msgs = build_messages(req.messages, context, req.system_prompt)

    def _generate() -> str:
        if inference_engine.model is not None:
            return inference_engine.generate(
                msgs, max_tokens=req.max_tokens,
                temperature=req.temperature, top_p=0.9,
            ).strip()
        from finetune_studio.models.manager import get_manager
        mgr = get_manager()
        if mgr.active() is None:
            raise ValueError("no model loaded")
        return mgr.chat(
            msgs, max_tokens=req.max_tokens,
            temperature=req.temperature, top_p=0.9,
        ).strip()

    try:
        async with ENGINE_LOCK:
            reply = await asyncio.to_thread(_generate)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # Strip thinking block for UI
    from finetune_studio.webui.thinking import split_thinking
    parts = split_thinking(reply)
    clean = parts["response"].strip() if parts.get("response") else reply

    sources = [{
        "filename": h["filename"], "source": h["source"],
        "score": h.get("ce_score", h.get("rrf_score", 0.0)),
        "rank": h["rank"], "chunk_text_preview": h["text"][:400]
    } for h in hits]
    return {"reply": clean, "sources": sources,
            "raw_reply_thinking": parts.get("thinking", ""),
            "hits": hits}
