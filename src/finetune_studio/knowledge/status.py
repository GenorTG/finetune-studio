"""Per-file pipeline flags for the file browser: parsed, mined into pairs, in the Smart Brain index."""
from __future__ import annotations

from typing import Any

from finetune_studio.knowledge import index
from finetune_studio.knowledge.pictures import source_counts
from finetune_studio.knowledge.sources import READY, list_sources


def pipeline_status(pid: str) -> dict[str, dict[str, Any]]:
    """``file_id -> {has_parsed, source_id, status, error, chunk_count, parser, edited, in_brain, brain_stale,
    pictures_ocr, scanned_ocr, figures_read}`` (the ``*_ocr`` counts: figures only OCR read, ``knowledge.pictures``)."""
    st = index.state(pid)

    def figures(s) -> dict[str, int]:
        c = source_counts(pid, s)
        return {"pictures_ocr": c["pictures_ocr"], "scanned_ocr": c["scanned_ocr"], "figures_read": c["read"]}

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
