"""Regression: an uploaded file's source must keep a resolvable raw path.

Bug 2026-09-18: the source row was rewritten after chunking without its path, so mining a file again failed with
"source file missing" after any restart. Upload is parse-only; the source points at the library's raw file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from finetune_studio.data.fs import qa as qa_fs
from finetune_studio.knowledge.sources import list_sources


def test_uploaded_source_keeps_raw_path(client: Any) -> None:
    """After an upload, the source has a raw path that exists, it is parsed and no pairs are mined yet."""
    pid = client.post("/api/projects", json={"name": "PathKeeper"}).json()["id"]
    data = b"# Lore\n\nThe disc floats above the Void Sea. Forty weavers sealed the shafts."
    r = client.post(f"/api/projects/{pid}/files/upload", files={"files": ("lore.txt", data, "text/plain")})
    assert r.status_code == 200, r.text
    [src] = list_sources(pid)
    assert Path(src.raw_path).is_file(), f"raw path does not exist on disk: {src.raw_path}"
    assert src.status == "ready" and src.chunk_count >= 1
    assert qa_fs.list_qa_pairs(pid) == []
