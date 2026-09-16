"""Routes for the per-project RAG.

Endpoints the UI uses (mirror of the existing style):

  GET  /api/projects/{pid}/rag              -- status + manifest summary
  GET  /api{pid}/rag/settings      -- settings (embedder, reranker, hybrid, etc.)
  POST /api{pid}/rag/settings      -- update settings (no rebuild)
  POST /api{pid}/rag/build        -- (re)build corpus from project's files dir
  POST /api{pid}/rag/rebuild-vectors  -- re-embed with current settings
  POST /api{pid}/rag/search       -- {query, top_k, hybrid?, rerank?, rerank_top_n?}
                                              -> {hits: [{rank, score, text, filename, ...}]}
  POST /api{pid}/rag/chat         -- {messages, top_k?} -> {reply, sources, citations}
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from finetune_studio.webui.live_sse import sse_data, sse_response

log = logging.getLogger(__name__)
router = APIRouter()

_CORPORA = Path.home() / ".finetune-studio" / "rag_corpora"


def _corpus_dir(pid: str) -> Path:
    return _CORPORA / pid


def _build_meta(pid: str, name: str) -> dict:
    """Sidecar DB-like info stored next to the project. For now: just counts."""
    return {"name": name, "pid": pid}


# ── DTOs ────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    hybrid: bool | None = None
    rerank: bool | None = None
    rerank_top_n: int | None = None


class ChatRequest(BaseModel):
    messages: list[dict]
    system_prompt: str = ""
    top_k: int = 5
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
        return JSONResponse({"error": "no corpus"}, status_code=404)
    m = rag.load().manifest
    return {
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
    return {"ok": True, "updated": req.dict(exclude_none=True)}


@router.post("/{pid}/rag/build")
async def rag_build(pid: str, req: BuildRequest):
    """(Re)build the project's RAG from the project's files dir.

    The corpus lives at `~/.finetune-studio/rag_corpora/<pid>/`. We source
    chunks from the project's structured filesystem (parsed.txt files), NOT
    from raw uploads — these are the OCR/parsed outputs that the data-prep
    pipeline already produced.
    """
    from finetune_studio.data.rag_portable import PortableRAG

    # Find source dir: project's files/<sha>/*.txt
    project_files_dir = Path.home() / ".finetune-studio" / "projects" / pid / "files"
    if not project_files_dir.exists():
        return JSONResponse({"error": "no parsed files in this project yet — upload & parse first"},
                            status_code=400)
    # Use the project name as corpus name if available
    from finetune_studio import db
    proj = db.get_project(pid)
    name = proj["name"] if proj else pid

    corpus = _corpus_dir(pid)
    if req.reset and corpus.exists():
        import shutil
        shutil.rmtree(corpus)
    rag = PortableRAG(corpus)

    # Synchronously count .txt files we are about to feed — gives the UI a
    # real "queued N files" message instead of the '?' fallback while the
    # background embedder spins up (which can take minutes on CPU).
    txt_files = sorted(project_files_dir.rglob("*.txt"))
    queued_files = sum(1 for f in txt_files if f.is_file())
    queued_chars = sum(f.stat().st_size for f in txt_files)

    # Run synchronously — embedding 45 small files takes ~30s.
    # Background tasks were silently failing because they don't inherit
    # the HF_HOME environment variable set by the systemd unit.
    try:
        result = rag.build_from_directory(
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
    return {
        "ok": True,
        "building": True,
        "corpus_dir": str(corpus),
        "queued_files": queued_files,
        "queued_chars": queued_chars,
    }


def _rag_build_snapshot(pid: str, *, elapsed_s: int = 0) -> dict:
    """One progress snapshot for SSE frames and the /build/status poll."""
    corpus = _corpus_dir(pid)
    project_files_dir = (
        Path.home() / ".finetune-studio" / "projects" / pid / "files"
    )
    total_files = (
        sum(1 for f in project_files_dir.rglob("*.txt") if f.is_file())
        if project_files_dir.exists() else 0
    )
    sources_dir = corpus / "sources"
    files_done = (
        sum(1 for _ in sources_dir.glob("*.txt")) if sources_dir.exists() else 0
    )
    manifest_exists = (corpus / "manifest.json").exists()
    chunks_path = corpus / "chunks.parquet"
    if manifest_exists:
        phase = "done"
    elif chunks_path.exists():
        phase = "embedding"
    elif files_done > 0:
        phase = "chunking"
    else:
        phase = "queued"
    chunks_count = 0
    if manifest_exists:
        try:
            m = json.loads((corpus / "manifest.json").read_text())
            chunks_count = int(m.get("chunks", 0) or 0)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            chunks_count = 0
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
    return _rag_build_snapshot(pid)


@router.get("/{pid}/rag/build/progress")
async def rag_build_progress(pid: str):
    """Server-Sent Events stream that reports corpus build progress.

    Emits JSON events:
        {"phase":"queued|chunking|embedding|done|error|timeout",
         "files_done":N,"files_total":M,"chunks":K,"elapsed_s":S}
    Stream terminates once ``phase`` is ``done``/``error``/``timeout``,
    or after 10 min. Prefer this over polling; clients may fall back to
    ``GET .../rag/build/status`` via fts.subscribe (fallbackMs ≥ 5s).
    """
    import asyncio

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
    from finetune_studio.data.rag_portable import PortableRAG
    rag = PortableRAG(_corpus_dir(pid))
    if not rag.exists():
        return JSONResponse({"error": "no corpus"}, status_code=404)
    try:
        result = rag.rebuild_vectors(embedder=req.embedder)
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
    """Clear all sources from the corpus (requires rebuild)."""
    from finetune_studio.data.rag_portable import PortableRAG
    rag = PortableRAG(_corpus_dir(pid))
    if not rag.exists():
        return JSONResponse({"error": "no corpus"}, status_code=404)
    try:
        rag.clear_sources()
        return {"ok": True}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=400)


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
    return {"hits": hits, "count": len(hits), "query": req.query}


@router.get("/{pid}/rag/bundle")
async def rag_bundle(pid: str, name: str | None = None,
                     fmt: str = "tar",
                     include_models: str = "true"):
    """Download a self-contained archive of the corpus.

    Query params:
      - name: custom filename stem (default: <project>-bundle-<date>)
      - fmt: 'tar' (fast, ~2.2GB) or 'tar.gz' (slow, ~2.0GB)
      - include_models: 'true' to copy the embedder + reranker INTO the
        archive (so recipient can run fully offline); 'false' to keep
        `shared:...` references and use the recipient's existing model store.

    Recipients unpack + install numpy + pandas + pyarrow + sentence-transformers
    and run `PortableRAG(dir).load()` to query. The RAG is fully offline-capable
    if `include_models=true` was used; otherwise the recipient needs to either
    have a model at the same shared path OR have network to fetch it.
    """
    from fastapi.responses import FileResponse

    from finetune_studio.data.rag_portable import PortableRAG
    rag = PortableRAG(_corpus_dir(pid))
    if not rag.exists():
        return JSONResponse({"error": "no corpus to bundle"}, status_code=404)
    inc_models = str(include_models).lower() not in ("0", "false", "no", "")
    try:
        out = rag.export_bundle(name=name, fmt=fmt, include_models=inc_models)
    except Exception as e:
        log.exception("bundle export failed")
        return JSONResponse({"error": f"bundle export failed: {e}"}, status_code=500)
    if not out.exists():
        return JSONResponse({"error": "bundle export produced no file"}, status_code=500)
    return FileResponse(
        path=str(out),
        media_type="application/x-tar" if fmt != "zip" else "application/zip",
        filename=out.name,
    )


@router.get("/shared-models/stats")
async def shared_model_stats():
    """Dashboard stats for the shared model pool. Shows which models are stored,
    how much disk they use, and how often they're referenced by corpora."""
    from finetune_studio.data.shared_models import stats as sm_stats
    return sm_stats()


@router.post("/{pid}/rag/chat")
async def rag_chat(pid: str, req: ChatRequest):
    """RAG-augmented chat. Retrieves top-k from project corpus, prepends to
    the conversation, runs the loaded model.

    Returns: {reply, sources: [{filename, score, chunk_text}], messages_full}
    """
    from finetune_studio.data.rag_portable import PortableRAG
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
    context = q.format_context(hits, max_chars=4000)

    sys_prompt = req.system_prompt or (
        "You are a knowledgeable assistant. Answer using ONLY the context below. "
        "If the answer isn't in the context, say so. Quote the source filename in [brackets] when relevant."
    )
    full_system = f"{sys_prompt}\n\nCONTEXT:\n{context}"
    msgs = [{"role": "system", "content": full_system}] + [
        {"role": m["role"], "content": m["content"]} for m in req.messages
    ]

    if inference_engine.model is not None:
        reply = inference_engine.generate(
            msgs, max_tokens=req.max_tokens,
            temperature=req.temperature, top_p=0.9,
        ).strip()
    else:
        from finetune_studio.models.manager import get_manager
        mgr = get_manager()
        if mgr.active() is None:
            return JSONResponse({"error": "no model loaded"}, status_code=400)
        reply = mgr.chat(
            msgs, max_tokens=req.max_tokens,
            temperature=req.temperature, top_p=0.9,
        ).strip()

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
