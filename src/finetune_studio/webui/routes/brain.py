"""Smart Brain routes: the project's retrieval index (status, settings, build, search, chat).

Mounted under ``/api/projects``. Methods behind search: semantic search (embeddings) + keyword search (BM25),
combined by rank fusion, then reranking; the answer gate declines when the hits do not cover the question.

  GET    /{pid}/brain                  -- status, manifest summary, what is stale (knowledge.index.state)
  GET    /{pid}/brain/settings         -- embedder, reranker, keyword search, fusion
  POST   /{pid}/brain/settings         -- update settings (no rebuild)
  POST   /{pid}/brain/build            -- (re)build the index from the project's parsed files
  POST   /{pid}/brain/quick            -- parse unparsed files, then build
  GET    /{pid}/brain/build/status     -- one-shot build progress
  GET    /{pid}/brain/build/progress   -- SSE build progress
  POST   /{pid}/brain/rebuild-vectors  -- re-embed with current settings
  POST   /{pid}/brain/search           -- {query, top_k, hybrid?, rerank?, rerank_top_n?}
  POST   /{pid}/brain/chat             -- {messages, top_k?} -> {reply, sources, hits}
  GET    /{pid}/brain/coverage         -- is every parsed file in the index with its current text?
Documents in the index: ``brain_documents``; bundles, import and the standalone server: ``brain_export``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from finetune_studio.data.fs.paths import brain_index_dir
from finetune_studio.knowledge.errors import KnowledgeError
from finetune_studio.smart_brain.constants import DEFAULT_TOP_K
from finetune_studio.webui.engine_guard import ENGINE_LOCK
from finetune_studio.webui.errors import ApiError
from finetune_studio.webui.live_sse import sse_data, sse_response

log = logging.getLogger(__name__)
router = APIRouter()


def _corpus_dir(pid: str) -> Path:
    return brain_index_dir(pid)


def _project_404(pid: str) -> JSONResponse | None:
    """Return a 404 response when the project does not exist, else None.

    This module answers errors with ``JSONResponse`` (see the corpus 404s
    below), so the guard matches that style. Called before any corpus
    directory is touched — otherwise a bad pid gets a real ``corpus_dir``
    path back with a 200.
    """
    from finetune_studio import db
    if not db.get_project(pid):
        return JSONResponse({"detail": "project not found"}, status_code=404)
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
    embedder: str | None = None
    reset: bool = False
    fact_cards: bool | None = None   # index approved pairs as fact cards; None keeps the last build's choice


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

@router.get("/{pid}/brain")
async def brain_status(pid: str):
    missing = _project_404(pid)
    if missing is not None:
        return missing
    from finetune_studio.knowledge import index as knowledge_index
    from finetune_studio.smart_brain import BrainIndex
    state = knowledge_index.state(pid).as_dict()
    if not BrainIndex(_corpus_dir(pid)).exists():
        return {"exists": False, "corpus_dir": str(_corpus_dir(pid)), "state": state}
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
        "state": state,
    }


@router.get("/{pid}/brain/settings")
async def brain_get_settings(pid: str):
    from finetune_studio.smart_brain import BrainIndex
    index = BrainIndex(_corpus_dir(pid))
    if not index.exists():
        # Not an error: the page asks before any index exists.
        return {"exists": False}
    m = index.load().manifest
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


@router.post("/{pid}/brain/settings")
async def brain_patch_settings(pid: str, req: SettingsPatch):
    """Update settings. Changes to embedder/rerank_top_n don't break anything,
    but changing embedder will make existing vectors invalid → caller must
    rebuild vectors afterwards."""
    from finetune_studio.smart_brain import (
        BrainIndex,
        Manifest,
        write_json,
    )
    index = BrainIndex(_corpus_dir(pid))
    if not index.exists():
        return JSONResponse({"detail": "no corpus"}, status_code=404)
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


@router.post("/{pid}/brain/build")
async def brain_build(pid: str, req: BuildRequest):
    """(Re)build the project's index from every parsed source (``knowledge.index.build``).

    The request stays open until the build finishes, in a worker thread so the event loop can still serve
    /build/progress. A thread inherits HF_HOME from the systemd unit; detached background tasks did not.
    """
    from finetune_studio.knowledge import index

    missing = _project_404(pid)
    if missing is not None:
        return missing
    queued = len(index.documents(pid))
    try:
        result = await asyncio.to_thread(index.build, pid, embedder=req.embedder, reset=req.reset,
                                         fact_cards=req.fact_cards)
    except KnowledgeError:
        raise
    except Exception as e:
        log.exception("index build failed")
        return JSONResponse({"ok": False, "detail": str(e)}, status_code=500)
    # Build is synchronous: the corpus is already written when we return.
    # ``building: false`` keeps the UI from claiming an in-flight embed.
    return {
        "ok": True,
        "building": False,
        "corpus_dir": result["corpus_dir"],
        "queued_files": queued,
        "documents": int(result.get("documents") or 0),
        "chunks": int(result.get("chunks") or 0),
        "vector_dim": result.get("vector_dim"),
        "skipped": int(result.get("skipped") or 0),
        "reset": bool(req.reset),
        "fact_cards": bool(result.get("fact_cards")),
        "fact_card_count": int(result.get("fact_card_count") or 0),
    }


class QuickRequest(BaseModel):
    file_ids: list[str] | None = None
    promote_missing: bool = True


@router.post("/{pid}/brain/quick")
async def brain_quick(pid: str, req: QuickRequest):
    """⚡ Quick index: parse every readable library file that is not parsed yet, then build the index from all
    parsed text — one click, no leaving the page. ``file_ids`` restricts the parse step.
    """
    from finetune_studio import db
    from finetune_studio.data.fs import file_library as fl
    from finetune_studio.knowledge import sources

    if not db.get_project(pid):
        return JSONResponse({"detail": "project not found"}, status_code=404)

    promoted: list[dict] = []
    failed: list[dict] = []
    if req.promote_missing:
        for f in fl.list_files(pid):
            if (req.file_ids and f["id"] not in req.file_ids) or not sources.is_parseable(f["original_name"]):
                continue
            src = sources.stage(pid, f["id"])
            if src.ready:
                continue
            src = await asyncio.to_thread(sources.parse, pid, src.id)
            promoted.append({"file_id": f["id"], "name": src.filename, "source_id": src.id, "status": src.status})
            if not src.ready:
                failed.append({"name": src.filename, "error": src.error or "parse incomplete"})

    build = await brain_build(pid, BuildRequest())
    if isinstance(build, JSONResponse):
        # The parse step's work is kept; the build failure still answers with the build's own status.
        reason = json.loads(build.body).get("detail") or "index build failed"
        raise ApiError(build.status_code, f"converted {len(promoted)} file(s), then the index build failed: {reason}",
                       code="build_failed", promoted=len(promoted), promoted_files=promoted, failed=failed)
    return {"ok": True, "promoted": len(promoted), "promoted_files": promoted, "failed": failed, "build": build}


def _build_snapshot(pid: str, *, elapsed_s: int = 0) -> dict:
    """One progress snapshot for SSE frames and the /build/status poll."""
    from finetune_studio.knowledge.sources import list_sources

    corpus = _corpus_dir(pid)
    total_files = sum(1 for s in list_sources(pid) if s.ready)
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


@router.get("/{pid}/brain/build/status")
async def brain_build_status(pid: str):
    """One-shot corpus-build progress (SSE silent fallback for /build/progress)."""
    missing = _project_404(pid)
    if missing is not None:
        return missing
    return _build_snapshot(pid)


@router.get("/{pid}/brain/build/progress")
async def brain_build_progress(pid: str):
    """Server-Sent Events stream that reports corpus build progress.

    Emits JSON events:
        {"phase":"queued|chunking|embedding|done|timeout",
         "files_done":N,"files_total":M,"chunks":K,"elapsed_s":S}
    Stream terminates once ``phase`` is ``done`` or ``timeout``,
    or after 10 min. Prefer this over polling; clients may fall back to
    ``GET …/brain/build/status`` via fts.subscribe (fallbackMs ≥ 5s).
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
                payload = _build_snapshot(pid, elapsed_s=elapsed)
                yield sse_data(payload)
                if payload["phase"] == "done":
                    return
                await asyncio.sleep(1.0)
            yield sse_data({
                "phase": "timeout",
                "files_total": _build_snapshot(pid).get("files_total", 0),
                "files_done": 0,
                "chunks": 0,
                "elapsed_s": 600,
            })
        except asyncio.CancelledError:
            return

    return sse_response(gen())


@router.post("/{pid}/brain/rebuild-vectors")
async def brain_rebuild_vectors(pid: str, req: RebuildVectorsRequest):
    """Re-embed with (possibly new) embedder. Loads + replaces vectors.npy."""
    from finetune_studio.smart_brain import BrainIndex, Manifest
    index = BrainIndex(_corpus_dir(pid))
    if not index.exists():
        return JSONResponse({"detail": "no corpus"}, status_code=404)
    embedder = req.embedder
    if not embedder:
        # ``POST /brain/settings`` writes a changed embedder to
        # ``rag_settings.embedder`` only (see brain_patch_settings) — the
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
        result = await asyncio.to_thread(index.rebuild_vectors, embedder=embedder)
        return {"ok": True, **result}
    except Exception as e:
        log.exception("rebuild failed")
        return JSONResponse({"ok": False, "detail": str(e)}, status_code=400)


_STOP = frozenset(["what", "which", "when", "where", "whom", "whose", "does", "that", "this", "with", "from", "have", "about", "tell", "there", "their", "into"])


def weak_match(query: str, hits: list[dict]) -> bool:
    """True when the retrieved text covers too few of the query's content words."""
    terms = {w for w in re.findall(r"[a-z0-9]{4,}", query.lower()) if w not in _STOP}
    if not terms or not hits:
        return False
    blob = " ".join(str(h.get("text", "")) for h in hits).lower()
    found = sum(1 for t in terms if t in blob)
    return found / len(terms) < 0.6


@router.post("/{pid}/brain/search")
async def brain_search(pid: str, req: SearchRequest):
    from finetune_studio.smart_brain import BrainIndex
    index = BrainIndex(_corpus_dir(pid))
    if not index.exists():
        return JSONResponse({"detail": "no corpus — build first"}, status_code=400)
    try:
        q = index.load()
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"detail": f"load failed: {e}"}, status_code=500)
    hits = q.search(req.query, top_k=req.top_k,
                    hybrid=req.hybrid, rerank=req.rerank,
                    rerank_top_n=req.rerank_top_n)
    weak = weak_match(req.query, hits)
    return {"hits": hits, "count": len(hits), "query": req.query, "weak": weak,
            "note": "These passages share few words with your question; the corpus may not cover it." if weak else ""}


def _active_n_ctx() -> int:
    """Window of the active chat provider (0 when none is loaded or the provider has no known window)."""
    from finetune_studio.models.manager import get_manager

    active = get_manager().active() or {}
    return int(active.get("n_ctx") or 0)


@router.post("/{pid}/brain/chat")
async def brain_chat(pid: str, req: ChatRequest):
    """RAG-augmented chat. Retrieves top-k from project corpus, prepends to
    the conversation, runs the loaded model.

    Returns: {reply, sources: [{filename, score, chunk_text}], messages_full}
    """
    from finetune_studio.smart_brain import BrainIndex
    from finetune_studio.smart_brain.prompt import (
        build_messages,
        context_char_budget,
    )
    from finetune_studio.webui.app import inference_engine

    index = BrainIndex(_corpus_dir(pid))
    if not index.exists():
        return JSONResponse({"detail": "no corpus — build first"}, status_code=400)
    try:
        q = index.load()
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"detail": f"load failed: {e}"}, status_code=500)

    # Use the last user message as the search query
    user_messages = [m for m in req.messages if m.get("role") == "user"]
    if not user_messages:
        return JSONResponse({"detail": "no user message to search for"}, status_code=400)
    last_user = user_messages[-1]["content"]

    hits = q.search(last_user, top_k=req.top_k)
    n_ctx = inference_engine.n_ctx if inference_engine.model is not None else (_active_n_ctx() or None)
    history_chars = sum(len(str(m.get("content", ""))) for m in req.messages) + len(req.system_prompt or "")
    context = q.format_context(hits, max_chars=context_char_budget(
        n_ctx, max_new_tokens=req.max_tokens, history_chars=history_chars))

    # Layout lives in smart_brain.prompt — shared with the context-grounded
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
        return JSONResponse({"detail": str(e)}, status_code=400)

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


@router.get("/{pid}/brain/coverage")
async def brain_coverage(pid: str):
    """100%-facts gate for the Smart Brain: is every parsed source in the index, with its current text?

    ``knowledge.index.state``: a source whose text changed since the build (hand edit, Vision reading, reparse)
    counts as missing until the index is rebuilt.
    """
    from finetune_studio import db
    from finetune_studio.knowledge import index
    from finetune_studio.knowledge.sources import list_sources

    if not db.get_project(pid):
        return JSONResponse({"detail": "project not found"}, status_code=404)
    sources = [s for s in list_sources(pid) if s.ready]
    # Empty states are normal (Quick work polls this on every load), so they are
    # 200 with ``state`` rather than a 400 the browser logs as an error.
    empty = {"project": pid, "covered": 0, "coverage_pct": 0.0}
    if not sources:
        return {**empty, "state": "no_sources", "parsed_sources": 0, "missing_sources": []}
    from finetune_studio.data.fs.paths import brain_index_dir

    corpus_dir = brain_index_dir(pid)
    st = index.state(pid)
    rows = [{"source_id": s.id, "filename": s.filename, "declared_chunks": s.chunk_count,
             "reason": "changed since the build" if s.id in st.stale else "not indexed"} for s in sources]
    if not st.built:
        return {**empty, "state": "not_built", "parsed_sources": len(rows), "missing_sources": rows}
    missing = [r for r in rows if r["source_id"] not in st.indexed]
    return {
        "state": "partial" if missing else "ok",
        "project": pid, "corpus_dir": str(corpus_dir),
        "parsed_sources": len(rows), "covered": len(rows) - len(missing),
        "missing_sources": missing,
        "coverage_pct": round(100.0 * (len(rows) - len(missing)) / len(rows), 1) if rows else 0.0,
    }
