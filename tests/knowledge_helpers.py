"""Test helper: put a file into a project the way the app does (file library -> one parse).

Tests used to write ``qa/sources/*.json`` manifests by hand; that layer is gone (``knowledge.sources``). A source
now always has real bytes in the library, so a test that needs one calls ``add_source``.
"""
from __future__ import annotations

from finetune_studio.knowledge import sources
from finetune_studio.knowledge.sources import Source


def add_source(pid: str, filename: str, content: str | bytes, *, parse: bool = True) -> Source:
    """Upload ``content`` as ``filename`` into project ``pid`` and (by default) parse it. Returns the source."""
    from finetune_studio.data.fs import file_library as fl

    data = content.encode("utf-8") if isinstance(content, str) else content
    meta = fl.write_uploaded_file(pid, data, filename)
    source = sources.stage(pid, meta.file_id)
    return sources.parse(pid, source.id) if parse else source
