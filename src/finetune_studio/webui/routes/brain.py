"""Smart Brain routes: the project's retrieval index (status, settings, build, search, chat).

Mounted under ``/api/projects``. Methods behind search: semantic search (embeddings) + keyword search (BM25),
combined by rank fusion, then reranking; the answer gate declines when the hits do not cover the question.

  GET    /{pid}/brain                  -- status, manifest summary, what is stale (knowledge.index.state)
  GET    /{pid}/brain/settings         -- embedder, reranker, keyword search, fusion
  POST   /{pid}/brain/settings         -- update settings (no rebuild)
  POST   /{pid}/brain/build            -- start a (re)build of the index (a job process) -> 202 + build_id
  POST   /{pid}/brain/quick            -- parse unparsed files, then start the build -> 202 + build_id
  GET    /{pid}/brain/builds/{bid}     -- one build (the build routes answer 202 with its id; poll this)
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
from pydantic import BaseModel, Field

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
    # Follow-up searches the model may ask for before answering (a question that reaches its answer through another
    # item: the site of an asset, then that site's manager). 0 = one search, answer from it.
    follow_up: int = Field(default=1, ge=0, le=3)


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
        # Not an error: the page asks before any index exists. The reranker a first build will get depends on the
        # machine (GPU: bge-reranker-v2-m3, CPU only: MiniLM), so the form shows that one.
        from finetune_studio.smart_brain.rerankers import default_reranker
        return {"exists": False, "default_reranker": await asyncio.to_thread(default_reranker)}
    m = await asyncio.to_thread(index.read_manifest)
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


async def _start_build(pid: str, *, mode: str, embedder: str | None = None, reset: bool = False,
                       fact_cards: bool | None = None, file_ids: list[str] | None = None) -> JSONResponse:
    """Start a build job in ``mode`` (``webui.brain_build_jobs``) and answer 202 with its ``build_id``.

    Raises ApiError: 409 a build of this project is running (``build_running``) or the GPU is busy
    (``gpu_busy``); the IndexBuildError 404/400 of the plan step pass through (the app's error handler)."""
    from finetune_studio.jobs import launch
    from finetune_studio.webui import brain_build_jobs

    try:
        plan = await asyncio.to_thread(brain_build_jobs.start, pid, embedder=embedder, reset=reset,
                                       fact_cards=fact_cards, mode=mode, file_ids=file_ids)
    except brain_build_jobs.BuildBusy as e:
        raise ApiError(409, str(e), code="build_running", build_id=e.build_id) from e
    except launch.JobRejected as e:
        raise ApiError(e.status if e.status >= 400 else 503, f"the index build could not start: {e}",
                       code=e.code or None) from e
    bid = plan["build_id"]
    return JSONResponse(status_code=202, content={
        "ok": True,
        "building": True,
        "build_id": bid,
        "status": "queued",
        "status_url": f"/api/projects/{pid}/brain/builds/{bid}",
        "corpus_dir": str(_corpus_dir(pid)),
        "mode": plan["mode"],
        "queued_files": plan["queued_files"],
        "pending_parses": plan.get("pending_parses", 0),
        "reset": bool(reset),
        "fact_cards": bool(plan["fact_cards"]),
        "gpu": bool(plan["gpu"]),
    })


@router.post("/{pid}/brain/build")
async def brain_build(pid: str, req: BuildRequest):
    """Start a (re)build of the project's index from every parsed source: a job process
    (``webui.brain_build_jobs``), so a web restart does not end it. Answers 202 with the ``build_id``; poll
    ``GET …/brain/builds/{build_id}`` until ``finished`` (``ok`` false = failed, ``error`` says why).

    404 unknown project, 400 nothing parsed yet, 409 a build of this project is running or the GPU is busy.
    """
    missing = _project_404(pid)
    if missing is not None:
        return missing
    return await _start_build(pid, mode="build", embedder=req.embedder, reset=req.reset,
                              fact_cards=req.fact_cards)


@router.get("/{pid}/brain/builds/{build_id}")
async def brain_build_get(pid: str, build_id: str):
    """One build: ``status`` queued | running | done | error, ``finished``, ``ok``, ``documents``, ``chunks``,
    ``error`` (the real one when it failed)."""
    from finetune_studio.webui import brain_build_jobs

    missing = _project_404(pid)
    if missing is not None:
        return missing
    out = await asyncio.to_thread(brain_build_jobs.status, pid, build_id)
    if out is None:
        raise ApiError(404, f"no index build {build_id} in this project", code="build_not_found")
    return out


class QuickRequest(BaseModel):
    file_ids: list[str] | None = None
    promote_missing: bool = True


@router.post("/{pid}/brain/quick")
async def brain_quick(pid: str, req: QuickRequest):
    """⚡ Quick index: parse every readable library file that is not parsed yet, then build the index from all
    parsed text — one click, no leaving the page. Both steps run in the build job (the parse step used to run in
    this request: a long one, ended by a web restart). ``file_ids`` restricts the parse step. Answers 202 with the
    started build (``build.build_id``: poll it like ``/brain/build``; the finished build reports ``promoted`` /
    ``failed`` for the parse step and, while parsing, ``parsing: {done, total}``). A build that cannot start is a
    ``build_failed`` error with the reason.
    """
    missing = _project_404(pid)
    if missing is not None:
        return missing
    try:
        build = await _start_build(pid, mode="quick" if req.promote_missing else "build", file_ids=req.file_ids)
    except ApiError as e:   # _start_build already says "the index build could not start: ..." where it applies
        detail = str(e.detail)
        if not detail.startswith("the index build could not start"):
            detail = f"the index build could not start: {detail}"
        raise ApiError(e.status_code, detail, code="build_failed",
                       **{k: v for k, v in e.extra.items() if k != "code"}) from e
    except KnowledgeError as e:
        raise ApiError(e.status, f"the index build could not start: {e}", code="build_failed") from e
    body = json.loads(build.body)
    return JSONResponse(status_code=202, content={
        "ok": True, "pending_parses": body["pending_parses"], "build": body, "build_id": body["build_id"]})


def _epoch(value: object) -> float:
    """A build row timestamp (epoch seconds; ``brain_builds``) as a float, now when unreadable."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return time.time()


def _build_snapshot(pid: str, *, elapsed_s: int = 0) -> dict:
    """One progress snapshot for SSE frames and the /build/status poll, from the project's latest build row.

    ``phase``: queued | parsing (the quick index's parse step: ``files_done`` / ``files_total`` count its files) |
    embedding (a build job is at work) | done (an index exists, no build running) | error (the latest build
    failed; ``message`` says why) | idle (never built)."""
    from finetune_studio import db
    from finetune_studio.knowledge.sources import list_sources
    from finetune_studio.webui import brain_build_jobs

    corpus = _corpus_dir(pid)
    total_files = sum(1 for s in list_sources(pid) if s.ready)
    files_done = chunks_count = 0
    manifest = corpus / "manifest.json"
    if manifest.exists():
        try:
            m = json.loads(manifest.read_text())
            files_done = int(m.get("documents", 0) or 0)
            chunks_count = int(m.get("chunks", 0) or 0)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            files_done = chunks_count = 0
    active = brain_build_jobs.active_build(pid)
    latest = active or db.brain_builds.latest_for_project(pid)
    message = ""
    passages_done = passages_total = 0
    if active is not None:
        phase = "queued" if active.get("status") == "queued" else "embedding"
        files_done = chunks_count = 0
        st = brain_build_jobs.status(pid, active["id"]) or {}
        if st.get("parsing"):
            phase, files_done, total_files = "parsing", int(st["parsing"]["done"]), int(st["parsing"]["total"])
        elif st.get("embedding"):
            passages_done, passages_total = int(st["embedding"]["done"]), int(st["embedding"]["total"])
        started = active.get("started_at") or active.get("created_at")
        if started:
            elapsed_s = max(0, int(time.time() - _epoch(started)))
    elif latest is not None and latest.get("status") in ("error", "failed"):
        phase, message = "error", str(latest.get("error") or "the index build failed")
    elif manifest.exists():
        phase = "done"
    else:
        phase = "idle"
    out = {
        "phase": phase,
        "files_done": files_done,
        "files_total": total_files,
        "chunks": chunks_count,
        "passages_done": passages_done,
        "passages_total": passages_total,
        "elapsed_s": elapsed_s,
        "build_id": (latest or {}).get("id"),
    }
    if message:
        out["message"] = message
    return out


@router.get("/{pid}/brain/build/status")
async def brain_build_status(pid: str):
    """One-shot corpus-build progress (SSE silent fallback for /build/progress)."""
    missing = _project_404(pid)
    if missing is not None:
        return missing
    return await asyncio.to_thread(_build_snapshot, pid)


@router.get("/{pid}/brain/build/progress")
async def brain_build_progress(pid: str):
    """Server-Sent Events stream that reports corpus build progress.

    Emits JSON events:
        {"phase":"queued|parsing|embedding|done|error|idle|timeout",
         "files_done":N,"files_total":M,"chunks":K,"elapsed_s":S,"build_id":...}
    Stream terminates once ``phase`` is ``done``, ``error``, ``idle`` or ``timeout``,
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
                payload = await asyncio.to_thread(_build_snapshot, pid)
                yield sse_data(payload)
                if payload["phase"] in ("done", "error", "idle"):
                    return
                await asyncio.sleep(1.0)
            yield sse_data({
                "phase": "timeout",
                "files_total": (await asyncio.to_thread(_build_snapshot, pid)).get("files_total", 0),
                "files_done": 0,
                "chunks": 0,
                "elapsed_s": 600,
            })
        except asyncio.CancelledError:
            return

    return sse_response(gen())


@router.post("/{pid}/brain/build/stop")
async def brain_build_stop(pid: str):
    """Stop the project's running build (the page's Stop button). It ends at the next embedded batch, a few seconds,
    and the index stays as it was before the build. 404 unknown project, 409 no build is running."""
    from finetune_studio.webui import brain_build_jobs

    missing = _project_404(pid)
    if missing is not None:
        return missing
    out = await asyncio.to_thread(brain_build_jobs.stop, pid)
    if out is None:
        raise ApiError(409, "no index build is running", code="no_build_running")
    return out


@router.post("/{pid}/brain/rebuild-vectors")
async def brain_rebuild_vectors(pid: str, req: RebuildVectorsRequest):
    """Re-embed the index with the (possibly new) embedder: the documents and chunks stay, the vectors are
    replaced. A build job like ``/brain/build`` (the embedder loads in the job process, never here): 202 with the
    ``build_id``; the finished build reports the ``embedder`` used. No ``embedder`` = the one the settings ask for
    (``POST /brain/settings`` writes it to ``rag_settings.embedder``; the vectors carry the old one until this
    runs). 404 no index yet, 409 a build is running or the GPU is busy.
    """
    missing = _project_404(pid)
    if missing is not None:
        return missing
    return await _start_build(pid, mode="re-embed", embedder=req.embedder or None)


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
        # Off the event loop: load() brings up the embedder and search() runs it and the reranker (7 s on a first
        # search, every other request waited).
        q = await asyncio.to_thread(index.load)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"detail": f"load failed: {e}"}, status_code=500)
    hits = await asyncio.to_thread(q.search, req.query, top_k=req.top_k,
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


# A GPU job (an index build whose embedder runs on the GPU, training, an export) unloads the chat model first: say so,
# a bare "no model loaded" right after a Quick index read like a broken chat.
NO_CHAT_MODEL = ("No chat model is loaded. A GPU job (an index build on the GPU, training or an export) unloads it: "
                 "load a model from the model pill in the top bar, then ask again.")


class _NoChatModel(Exception):
    pass


def _chat_model_loaded() -> bool:
    from finetune_studio.models.manager import get_manager
    from finetune_studio.webui.app import inference_engine

    return inference_engine.model is not None or get_manager().active() is not None


@router.post("/{pid}/brain/chat")
async def brain_chat(pid: str, req: ChatRequest):
    """RAG-augmented chat. Retrieves top-k from project corpus, prepends to
    the conversation, runs the loaded model.

    Returns: {reply, sources: [{filename, score, chunk_text}], messages_full}
    """
    from finetune_studio.smart_brain import BrainIndex
    from finetune_studio.smart_brain.prompt import context_char_budget
    from finetune_studio.smart_brain.standalone_server import answer_with_follow_ups
    from finetune_studio.webui.app import inference_engine

    index = BrainIndex(_corpus_dir(pid))
    if not index.exists():
        return JSONResponse({"detail": "no corpus — build first"}, status_code=400)
    # Use the last user message as the search query
    user_messages = [m for m in req.messages if m.get("role") == "user"]
    if not user_messages:
        return JSONResponse({"detail": "no user message to search for"}, status_code=400)
    if not await asyncio.to_thread(_chat_model_loaded):   # before the embedder load + search it would waste
        raise ApiError(409, NO_CHAT_MODEL, code="no_model")
    last_user = user_messages[-1]["content"]
    try:
        q = await asyncio.to_thread(index.load)   # embedder load + search off the event loop, like brain_search
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"detail": f"load failed: {e}"}, status_code=500)

    hits = await asyncio.to_thread(q.search, last_user, top_k=req.top_k)
    n_ctx = inference_engine.n_ctx if inference_engine.model is not None else (_active_n_ctx() or None)
    history_chars = sum(len(str(m.get("content", ""))) for m in req.messages) + len(req.system_prompt or "")
    budget = context_char_budget(n_ctx, max_new_tokens=req.max_tokens, history_chars=history_chars)
    last_index = max(i for i, m in enumerate(req.messages) if m.get("role") == "user")
    history = [{"role": m["role"], "content": m["content"]} for m in req.messages[:last_index]]

    def _generate(msgs: list[dict]) -> str:
        if inference_engine.model is not None:
            return inference_engine.generate(
                msgs, max_tokens=req.max_tokens,
                temperature=req.temperature, top_p=0.9,
            ).strip()
        from finetune_studio.models.manager import get_manager
        mgr = get_manager()
        if mgr.active() is None:
            raise _NoChatModel
        return mgr.chat(
            msgs, max_tokens=req.max_tokens,
            temperature=req.temperature, top_p=0.9,
        ).strip()

    from finetune_studio.webui.thinking import split_thinking
    raw: list[str] = []

    def _answer() -> dict:
        def chat(msgs: list[dict]) -> str:
            raw.append(_generate(msgs))
            parts = split_thinking(raw[-1])
            return parts["response"].strip() if parts.get("response") else raw[-1]
        # The prompt layout is smart_brain.prompt's (the context-grounded training rows use it too); the follow-up
        # loop is the standalone server's /ask, so the studio and an exported package answer the same way.
        return answer_with_follow_ups(last_user, hits, lambda query: q.search(query, top_k=req.top_k), chat,
                                      follow_ups=req.follow_up, max_chars=budget,
                                      system_prompt=req.system_prompt, history=history)

    try:
        async with ENGINE_LOCK:
            result = await asyncio.to_thread(_answer)
    except _NoChatModel:
        raise ApiError(409, NO_CHAT_MODEL, code="no_model") from None   # unloaded while the search ran
    except ValueError as e:
        return JSONResponse({"detail": str(e)}, status_code=400)

    clean = result["answer"]
    hits = result["hits"]
    parts = split_thinking(raw[-1]) if raw else {}

    sources = [{
        "filename": h["filename"], "source": h["source"],
        "score": h.get("ce_score", h.get("rrf_score", 0.0)),
        "rank": h["rank"], "chunk_text_preview": h["text"][:400]
    } for h in hits]
    return {"reply": clean, "sources": sources,
            "raw_reply_thinking": parts.get("thinking", ""),
            "follow_up_searches": result["follow_up_searches"],
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
