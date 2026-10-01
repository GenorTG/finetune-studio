"""Lazy source-backed data-prep runner for large per-file queues."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from finetune_studio.data.prep.runner import DataPrepRunner, PrepProgress


class QueuedSourcePrep:
    """Hold only a path while queued; read bytes when this job actually starts."""

    def __init__(
        self,
        *,
        pid: str,
        path: Path,
        filename: str,
        qa_per_chunk: int,
        difficulty: str,
        style: str,
        uploaded_by: str,
        progress_cb: Callable[[PrepProgress], None],
    ) -> None:
        self.pid = pid
        self.path = path
        self.filename = filename
        self.qa_per_chunk = qa_per_chunk
        self.difficulty = difficulty
        self.style = style
        self.uploaded_by = uploaded_by
        self.progress_cb = progress_cb
        self._queued_progress = PrepProgress(
            stage="queued", message="Waiting for earlier files"
        )
        self._delegate: DataPrepRunner | None = None
        self._cancelled = False

    def run(self) -> dict[str, Any]:
        if self._cancelled:
            return {"ok": False, "error": "cancelled"}
        data = self.path.read_bytes()
        self._delegate = DataPrepRunner(
            pid=self.pid,
            data=data,
            filename=self.filename,
            qa_per_chunk=self.qa_per_chunk,
            difficulty=self.difficulty,
            style=self.style,
            uploaded_by=self.uploaded_by,
            progress_cb=self.progress_cb,
        )
        return self._delegate.run()

    def cancel(self) -> None:
        if self._delegate is not None:
            self._delegate.cancel()
        else:
            self._cancelled = True
            self._queued_progress = PrepProgress(stage="error", message="Cancelled")

    @property
    def progress(self) -> PrepProgress:
        """Return live delegated progress or the lightweight queued state."""
        if self._delegate is not None:
            return self._delegate.progress
        return self._queued_progress
