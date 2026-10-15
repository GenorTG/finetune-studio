"""Regression: re-upload after soft-delete must succeed.

QABUG (2026-09-18, Vaelindrath stress pass): delete a file, then re-upload
the identical bytes. Upload dedup ignores deleted rows (correct), but the
insert path reused the content-derived file_id (first-16 sha256) which was
still held by the trash row → "UNIQUE constraint failed: project_files.id".
The re-upload of deliberately-deleted content was permanently poisoned.
"""

from __future__ import annotations

from typing import Any


def _upload(pid: str, fid_ref: list[str], name: str, payload: bytes, client: Any) -> dict:
    r = client.post(
        f"/api/projects/{pid}/files/upload",
        files={"files": (name, payload, "text/plain")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    rep = body["report"][0]
    if body["counts"]["uploaded"] == 1:
        fid_ref.append(rep["file_id"])
    return body


def test_reupload_after_softdelete_lands(client: Any) -> None:
    pid = client.post("/api/projects", json={"name": "ReUp"}).json()["id"]
    data = b"Vaelindrath annal: the Grellock Slide buried 2,400 moth-looms in 954."
    ids: list[str] = []
    first = _upload(pid, ids, "annal.txt", data, client)
    assert first["counts"]["uploaded"] == 1
    assert len(ids) == 1, first
    fid = ids[0]

    # Soft-delete it
    r = client.delete(f"/api/projects/{pid}/files/{fid}")
    assert r.status_code == 200, r.text

    # Re-upload identical bytes — must land as a NEW row, not 500/422.
    second = _upload(pid, ids, "annal.txt", data, client)
    assert second["counts"]["uploaded"] == 1, second
    assert second["counts"]["duplicates_skipped"] == 0
