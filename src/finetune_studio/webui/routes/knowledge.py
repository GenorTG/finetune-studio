"""Files & Knowledge routes: search the project's facts and passages (``knowledge.search``).

Mounted under ``/api/projects``:

  GET /{pid}/knowledge/search?q=…&kinds=fact,passage&status=&limit=25
      -> {query, hits: [{kind, score, source_id, file_id, filename, chunk_idx, text, pair_id, question, answer, status}]}

A fact is edited with the pair routes (``PATCH /{pid}/data-prep/qa/{pair_id}``), a passage in its file's parsed-text
editor (``PUT /{pid}/files/{file_id}/parsed``).
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from finetune_studio.knowledge import search as knowledge_search

router = APIRouter()


@router.get("/{pid}/knowledge/search")
async def search_knowledge(pid: str, q: str = Query("", max_length=500), kinds: str = "fact,passage",
                           status: str | None = None, limit: int = Query(25, ge=1, le=200)) -> dict[str, Any]:
    from finetune_studio import db

    if not db.get_project(pid):
        raise HTTPException(status_code=404, detail="project not found")
    wanted = tuple(k for k in (x.strip() for x in kinds.split(",")) if k)
    bad = [k for k in wanted if k not in knowledge_search.KINDS]
    if bad or not wanted:
        raise HTTPException(status_code=400, detail=f"kinds must be a comma list of {', '.join(knowledge_search.KINDS)}")
    if status not in (None, "", "pending", "approved", "rejected"):
        raise HTTPException(status_code=400, detail="status must be pending, approved or rejected")
    if len(q.strip()) < 2:
        raise HTTPException(status_code=400, detail="type at least 2 characters to search")
    hits = await asyncio.to_thread(knowledge_search.search, pid, q, kinds=wanted, status=status or None, limit=limit)
    return {"query": q, "hits": [h.as_dict() for h in hits]}
