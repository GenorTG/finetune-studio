"""What the Pairs review page needs beyond a flat list: a slim paged queue, counts, a pair's full source chunk, reviewer-written pairs.

The flat ``list_qa_pairs`` ships every pair with its stored chunk text (9.5 MB for 5,000 pairs). The review queue sends only the
fields a row needs, in document order (file, chunk, creation), and the page fetches one pair's full chunk when it is opened.
"""
from __future__ import annotations

import time
import uuid

from finetune_studio.data.fs.qa import (
    list_qa_pairs,
    list_qa_sources,
    read_qa_source,
    write_qa_pair,
)

# Fields a queue row needs. chunk_text is left out on purpose: it is fetched per pair by pair_context().
_SLIM_FIELDS = ("id", "source_id", "chunk_idx", "question", "answer", "status", "origin", "note", "created_at", "updated_at",
                "reviewed_at", "score")
STATUSES = ("pending", "approved", "rejected")


def status_counts(pairs: list[dict]) -> dict[str, int]:
    counts = dict.fromkeys(STATUSES, 0)
    for qa in pairs:
        st = qa.get("status") or "pending"
        counts[st if st in counts else "pending"] += 1
    return counts


def _sort_key(qa: dict) -> tuple:
    return (str(qa.get("source_id") or ""), int(qa.get("chunk_idx") or 0), float(qa.get("created_at") or 0), str(qa.get("id")))


def review_queue(pid: str, *, status: str | None = None, source_id: str | None = None, text: str = "", offset: int = 0,
                 limit: int = 200) -> dict:
    """One page of slim pairs plus project-wide counts (overall and per source file), so the page never needs the full list."""
    everything = list_qa_pairs(pid)
    names = {s["id"]: s.get("filename") or s.get("name") or s["id"] for s in list_qa_sources(pid)}
    per_source: dict[str, dict] = {}
    for qa in everything:
        sid = str(qa.get("source_id") or "")
        row = per_source.setdefault(sid, {"source_id": sid, "filename": names.get(sid, sid), **dict.fromkeys(STATUSES, 0)})
        st = qa.get("status") or "pending"
        row[st if st in STATUSES else "pending"] += 1
    wanted = [qa for qa in everything
              if (not status or (qa.get("status") or "pending") == status) and (not source_id or qa.get("source_id") == source_id)]
    needle = text.strip().lower()
    if needle:
        wanted = [qa for qa in wanted if needle in f"{qa.get('question', '')} {qa.get('answer', '')}".lower()]
    # Order the review by document (file name, then chunk), not by arrival, so a reviewer reads a file front to back.
    wanted.sort(key=lambda qa: (names.get(str(qa.get("source_id") or ""), ""), *_sort_key(qa)))
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), 1000))
    page = []
    for qa in wanted[offset:offset + limit]:
        row = {k: qa.get(k) for k in _SLIM_FIELDS if k in qa}
        row["source_filename"] = names.get(str(qa.get("source_id") or ""), "")
        page.append(row)
    return {"items": page, "total": len(wanted), "offset": offset, "counts": status_counts(everything),
            "sources": sorted(per_source.values(), key=lambda r: str(r["filename"]))}


def pair_context(pid: str, qa_id: str) -> dict | None:
    """The pair with its whole source chunk (the stored copy is cut at 1,500 characters) and the chunk's position."""
    from finetune_studio.data.prep.ingest import load_existing_chunks

    qa = next((q for q in list_qa_pairs(pid) if q.get("id") == qa_id), None)
    if qa is None:
        return None
    chunks = load_existing_chunks(pid, str(qa.get("sha256") or ""))
    idx = int(qa.get("chunk_idx") or 0)
    full = chunks[idx - 1] if 1 <= idx <= len(chunks) and chunks[idx - 1] else str(qa.get("chunk_text", ""))
    siblings = [q["id"] for q in list_qa_pairs(pid, source_id=qa.get("source_id"))
                if int(q.get("chunk_idx") or 0) == idx and q["id"] != qa_id]
    return {"pair": qa, "chunk_text": full, "chunk_idx": idx, "chunk_total": len(chunks), "same_chunk_pairs": len(siblings)}


def add_reviewer_pair(pid: str, source_id: str, chunk_idx: int, question: str, answer: str) -> dict:
    """A pair the miner never wrote, authored by the reviewer: approved, origin=human_review, with the chunk as its context."""
    source = read_qa_source(pid, source_id)
    if not source:
        raise KeyError(f"unknown source {source_id}")
    from finetune_studio.data.prep.ingest import load_existing_chunks

    sha = str(source.get("sha256") or "")
    chunks = load_existing_chunks(pid, sha)
    if not 1 <= chunk_idx <= max(len(chunks), 1):
        raise ValueError(f"chunk_idx {chunk_idx} is outside 1..{len(chunks)}")
    chunk = chunks[chunk_idx - 1] if chunks else ""
    now = time.time()
    qa = {
        "id": uuid.uuid4().hex[:12], "source_id": source_id, "sha256": sha, "chunk_idx": chunk_idx, "chunk_text": chunk[:1500],
        "question": question.strip(), "answer": answer.strip(), "difficulty": "medium", "style": "factual", "score": 1.0,
        "status": "approved", "origin": "human_review", "created_at": now, "updated_at": now, "reviewed_at": now,
        "provenance": {"source_id": source_id, "filename": source.get("filename", ""), "chunk_idx": chunk_idx,
                       "generator": "human-review"},
        "validation": {"accepted": True, "version": "human", "reasons": []},
    }
    write_qa_pair(pid, qa)
    return qa
