"""Per-file pipeline flags for the file browser: parsed, mined into pairs, in the Smart Brain index."""
from __future__ import annotations

from typing import Any

from finetune_studio.knowledge import index
from finetune_studio.knowledge.sources import READY, list_sources


def pipeline_status(pid: str) -> dict[str, dict[str, Any]]:
    """``file_id -> {has_parsed, source_id, status, error, chunk_count, parser, edited, in_brain, brain_stale}``."""
    st = index.state(pid)
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
        }
        for s in list_sources(pid)
    }
