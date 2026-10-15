"""Per-file mining API: queue/cancel jobs, read a file's inventory with its coverage, exclude items, inspect rounds.

    POST /api/projects/{pid}/mining/jobs                      {"source_ids": [...], "kind": "mine" | "fill"}
    GET  /api/projects/{pid}/mining/jobs?active=1&source_id=
    POST /api/projects/{pid}/mining/jobs/{job_id}/cancel
    GET  /api/projects/{pid}/mining/files/{source_id}/items   the checklist: every item with its state and covering pairs
    POST /api/projects/{pid}/mining/files/{source_id}/proposals/confirm  exclude what the helper proposed to skip
    POST /api/projects/{pid}/mining/files/{source_id}/reset   drop inventory, rounds and UNREVIEWED pairs (mine again)
    POST /api/projects/{pid}/mining/items/{item_id}/exclude   {"reason": "..."}; an empty reason includes the item again
    GET  /api/projects/{pid}/mining/files/{source_id}/rounds?job_id=
    GET  /api/projects/{pid}/mining/rounds/{round_id}         one round with its prompt and raw reply
    GET  /api/projects/{pid}/mining/pairs/{qa_id}/items       the items a pair states (review: "covers", last-cover warning)

Errors are real statuses: 404 unknown project/file/item/job, 409 a request the current state refuses (helper not usable, a
job already running, a file not mined yet), 400 malformed input.
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


def _project_404(pid: str) -> JSONResponse | None:
    from finetune_studio import db
    if not db.get_project(pid):
        return JSONResponse({"detail": "project not found"}, status_code=404)
    return None


def _source_404(pid: str, source_id: str) -> JSONResponse | None:
    from finetune_studio.knowledge.sources import get_source
    if get_source(pid, source_id) is None:
        return JSONResponse({"detail": "file not found"}, status_code=404)
    return None


async def _json(request: Request) -> dict[str, Any] | JSONResponse:
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({"detail": "invalid JSON"}, status_code=400)
    return body if isinstance(body, dict) else JSONResponse({"detail": "expected a JSON object"}, status_code=400)


@router.post("/api/projects/{pid}/mining/jobs")
async def submit_jobs(pid: str, request: Request):
    if (missing := _project_404(pid)) is not None:
        return missing
    body = await _json(request)
    if isinstance(body, JSONResponse):
        return body
    from finetune_studio.data.mining import MiningError, get_scheduler
    ids = body.get("source_ids")
    if not isinstance(ids, list):
        return JSONResponse({"detail": "source_ids must be a list"}, status_code=400)
    try:
        jobs = await asyncio.to_thread(get_scheduler().submit, pid, [str(i) for i in ids], str(body.get("kind") or "mine"))
    except MiningError as e:
        return JSONResponse({"detail": str(e)}, status_code=e.status)
    return {"ok": True, "jobs": jobs}


@router.get("/api/projects/{pid}/mining/jobs")
async def list_jobs(pid: str, active: bool = False, source_id: str | None = None, limit: int = 100):
    if (missing := _project_404(pid)) is not None:
        return missing
    from finetune_studio import db
    from finetune_studio.data.mining import helper_info
    jobs = await asyncio.to_thread(db.mining.list_jobs, pid, source_id=source_id, active=active, limit=max(1, min(limit, 500)))
    return {"jobs": jobs, "helper": await asyncio.to_thread(helper_info)}


@router.post("/api/projects/{pid}/mining/jobs/{job_id}/cancel")
async def cancel_job(pid: str, job_id: str):
    if (missing := _project_404(pid)) is not None:
        return missing
    from finetune_studio.data.mining import MiningError, get_scheduler
    try:
        return {"ok": True, "job": await asyncio.to_thread(get_scheduler().cancel, pid, job_id)}
    except MiningError as e:
        return JSONResponse({"detail": str(e)}, status_code=e.status)


@router.post("/api/projects/{pid}/mining/files/{source_id}/reset")
async def reset_file(pid: str, source_id: str):
    if (missing := _project_404(pid) or _source_404(pid, source_id)) is not None:
        return missing
    from finetune_studio.data.mining import MiningError, get_scheduler
    try:
        return {"ok": True, **await asyncio.to_thread(get_scheduler().reset, pid, source_id)}
    except MiningError as e:
        return JSONResponse({"detail": str(e)}, status_code=e.status)


def _checklist(pid: str, source_id: str) -> dict[str, Any]:
    from finetune_studio import db
    from finetune_studio.data.fs import qa as qa_fs
    from finetune_studio.data.mining import file_coverage

    items = db.mining.list_items(pid, source_id)
    pairs = qa_fs.list_qa_pairs(pid, source_id=source_id)
    cov = file_coverage(pid, source_id, items=items, pairs=pairs)
    by_id = {p["id"]: p for p in pairs}
    states = {s.item_id: s for s in cov.items}
    out = []
    for it in items:
        st = states.get(it["id"])
        if st is None:
            continue                                 # no checkable data left in it (coverage.item_states)
        out.append({
            "id": it["id"], "chunk_idx": it["chunk_idx"], "origin": it["origin"], "kind": it["kind"], "text": it["text"],
            "header": it["header"], "values": it["values"], "state": st.state, "excluded_reason": it["excluded_reason"],
            "proposed_reason": st.proposed,
            "pairs": [{"id": pid_, "status": by_id[pid_]["status"], "question": by_id[pid_]["question"]}
                      for pid_ in st.pair_ids if pid_ in by_id],
        })
    return {"source_id": source_id, "coverage": cov.as_dict(), "items": out}


@router.get("/api/projects/{pid}/mining/files/{source_id}/items")
async def file_items(pid: str, source_id: str):
    if (missing := _project_404(pid) or _source_404(pid, source_id)) is not None:
        return missing
    return await asyncio.to_thread(_checklist, pid, source_id)


@router.post("/api/projects/{pid}/mining/items/{item_id}/exclude")
async def exclude_item(pid: str, item_id: str, request: Request):
    if (missing := _project_404(pid)) is not None:
        return missing
    body = await _json(request)
    if isinstance(body, JSONResponse):
        return body
    from finetune_studio import db
    if not db.mining.get_item(pid, item_id):
        return JSONResponse({"detail": "item not found"}, status_code=404)
    item = await asyncio.to_thread(db.mining.set_item_exclusion, pid, item_id, str(body.get("reason") or "").strip()[:500])
    return {"ok": True, "item": item}


@router.post("/api/projects/{pid}/mining/files/{source_id}/proposals/confirm")
async def confirm_proposals(pid: str, source_id: str, request: Request):
    """Exclude the open items the helper proposed to skip: all of the file's, or ``{"item_ids": [...]}``."""
    if (missing := _project_404(pid) or _source_404(pid, source_id)) is not None:
        return missing
    body = await _json(request)
    if isinstance(body, JSONResponse):
        return body
    ids = body.get("item_ids")
    if ids is not None and not (isinstance(ids, list) and all(isinstance(i, str) for i in ids)):
        return JSONResponse({"detail": "item_ids must be a list of item ids"}, status_code=422)
    from finetune_studio import db
    n = await asyncio.to_thread(db.mining.confirm_proposals, pid, source_id, ids)
    return {"ok": True, "confirmed": n}


@router.get("/api/projects/{pid}/mining/files/{source_id}/rounds")
async def file_rounds(pid: str, source_id: str, job_id: str | None = None):
    if (missing := _project_404(pid) or _source_404(pid, source_id)) is not None:
        return missing
    from finetune_studio import db
    return {"rounds": await asyncio.to_thread(db.mining.list_rounds, pid, source_id, job_id)}


@router.get("/api/projects/{pid}/mining/rounds/{round_id}")
async def get_round(pid: str, round_id: str):
    if (missing := _project_404(pid)) is not None:
        return missing
    from finetune_studio import db
    row = await asyncio.to_thread(db.mining.get_round, pid, round_id)
    if not row:
        return JSONResponse({"detail": "round not found"}, status_code=404)
    return row


def _pair_items(pid: str, qa_id: str) -> dict[str, Any] | None:
    from finetune_studio import db
    from finetune_studio.data.fs import qa as qa_fs
    from finetune_studio.data.mining import item_states

    pair = next((p for p in qa_fs.list_qa_pairs(pid) if p.get("id") == qa_id), None)
    if pair is None:
        return None
    source_id = str(pair.get("source_id") or "")
    items = db.mining.list_items(pid, source_id)
    pairs = qa_fs.list_qa_pairs(pid, source_id=source_id)
    with_it = item_states(items, pairs)
    others = [p if p.get("id") != qa_id else {**p, "status": "rejected"} for p in pairs]
    without_it = {s.item_id: s for s in item_states(items, others)}
    text = {it["id"]: it["text"] for it in items}
    covers = [{"item_id": s.item_id, "text": text[s.item_id], "state": s.state,
               # rejecting this pair would leave the item stated by nothing: the reviewer is warned
               "last_cover": without_it[s.item_id].state == "open"}
              for s in with_it if qa_id in s.pair_ids]
    return {"qa_id": qa_id, "source_id": source_id, "items": covers}


@router.get("/api/projects/{pid}/mining/pairs/{qa_id}/items")
async def pair_items(pid: str, qa_id: str):
    if (missing := _project_404(pid)) is not None:
        return missing
    out = await asyncio.to_thread(_pair_items, pid, qa_id)
    if out is None:
        return JSONResponse({"detail": "pair not found"}, status_code=404)
    return out
