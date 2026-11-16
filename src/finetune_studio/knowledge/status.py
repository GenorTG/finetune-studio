"""Per-file pipeline flags for the file browser: parsed, mined into pairs, in the Smart Brain index."""
from __future__ import annotations

from typing import Any

from finetune_studio.knowledge import index
from finetune_studio.knowledge.pictures import source_counts
from finetune_studio.knowledge.sources import READY, list_sources


def pipeline_status(pid: str) -> dict[str, dict[str, Any]]:
    """``file_id -> {has_parsed, source_id, status, error, chunk_count, parser, edited, in_brain, brain_stale,
    pictures_ocr, scanned_ocr, figures_read, pictures_job}`` (the ``*_ocr`` counts: figures only OCR read,
    ``knowledge.pictures``; ``pictures_job``: "" / "reading" / "waiting: <what holds the GPU>" for a file whose
    figures the project's pictures job is about to read)."""
    from finetune_studio.knowledge import pictures

    st = index.state(pid)
    try:
        job = pictures._running(pid)
    except Exception:  # noqa: BLE001 - the flags never fail on the job lookup
        job = None
    job_note = ""
    if job is not None:
        job_note = (f"waiting: {pictures._holder_label(str(job.get('waiting_for') or ''))}" if job.get("queued")
                    else "reading")

    def figures(s) -> dict[str, Any]:
        c = source_counts(pid, s)
        return {"pictures_ocr": c["pictures_ocr"], "scanned_ocr": c["scanned_ocr"], "figures_read": c["read"],
                "pictures_job": job_note if (c["pictures_ocr"] or c["scanned_ocr"]) else ""}

    return {
        s.file_id: {
            "has_parsed": s.status == READY,
            "source_id": s.id,
            "status": s.status,
            "error": s.error,
            "chunk_count": s.chunk_count,
            "parser": s.parser,
            "edited": s.edited_at is not None,
            "in_brain": s.id in st.indexed or s.id in st.stale,
            "brain_stale": s.id in st.stale,
            **figures(s),
        }
        for s in list_sources(pid)
    }
