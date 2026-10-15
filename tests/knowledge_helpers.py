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


def wait_build(client, pid: str, response, timeout: float = 30.0) -> dict:
    """An index build answers 202 with its ``build_id`` (a job; in-process under the test suite): poll the build
    until it is finished and return it (``ok``, ``documents``, ``chunks``, ``error``...). Call it inside any
    ``patch(...)`` block the build needs: the job runs on a thread after the POST returned."""
    import time

    assert response.status_code == 202, response.text
    body = response.json()
    bid = body.get("build_id") or (body.get("build") or {}).get("build_id")
    assert bid, body
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/api/projects/{pid}/brain/builds/{bid}")
        assert r.status_code == 200, r.text
        if r.json()["finished"]:
            return r.json()
        time.sleep(0.05)
    raise AssertionError(f"build {bid} still running after {timeout} s: {r.json()}")
