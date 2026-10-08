"""Pairs review page backend: slim paged queue, cached listing, per-pair context, human verdict stamp, reviewer-written pairs."""
from __future__ import annotations

import json
import time
import uuid

from finetune_studio.data.fs import qa as qa_fs
from finetune_studio.data.fs.chunks import write_chunks


def _source(pid: str, sid: str, name: str, sha: str = "") -> None:
    qa_fs.write_qa_source(pid, {"id": sid, "filename": name, "name": name, "sha256": sha, "status": "ready", "uploaded_at": time.time()})


def _pair(pid: str, sid: str, chunk: int, q: str, status: str = "pending", **extra) -> str:
    qid = uuid.uuid4().hex[:12]
    qa_fs.write_qa_pair(pid, {"id": qid, "source_id": sid, "chunk_idx": chunk, "question": q, "answer": f"A to {q}",
                              "chunk_text": "x" * 1500, "status": status, "created_at": time.time(), **extra})
    return qid


def test_queue_is_slim_ordered_by_file_and_chunk_and_counts_everything(project_env) -> None:
    client, pid, _ = project_env
    _source(pid, "s-b", "b_file.txt")
    _source(pid, "s-a", "a_file.txt")
    late = _pair(pid, "s-b", 1, "b1?")
    _pair(pid, "s-a", 2, "a2?", status="approved")
    first = _pair(pid, "s-a", 1, "a1?", status="rejected")
    body = client.get(f"/api/projects/{pid}/data-prep/qa/queue").json()
    assert [r["question"] for r in body["items"]] == ["a1?", "a2?", "b1?"]
    assert all("chunk_text" not in r for r in body["items"])
    assert body["items"][0]["id"] == first and body["items"][0]["source_filename"] == "a_file.txt"
    assert body["counts"] == {"pending": 1, "approved": 1, "rejected": 1} and body["total"] == 3
    assert {s["filename"]: (s["pending"], s["approved"], s["rejected"]) for s in body["sources"]} == {
        "a_file.txt": (0, 1, 1), "b_file.txt": (1, 0, 0)}
    only = client.get(f"/api/projects/{pid}/data-prep/qa/queue", params={"status": "pending"}).json()
    assert [r["id"] for r in only["items"]] == [late] and only["counts"]["approved"] == 1  # counts stay project-wide


def test_queue_pages_filters_by_source_and_searches_text(project_env) -> None:
    client, pid, _ = project_env
    _source(pid, "s-a", "a.txt")
    _source(pid, "s-b", "b.txt")
    for i in range(5):
        _pair(pid, "s-a", i + 1, f"alpha {i}?")
    _pair(pid, "s-b", 1, "needle question?")
    page = client.get(f"/api/projects/{pid}/data-prep/qa/queue", params={"limit": 2, "offset": 2}).json()
    assert [r["question"] for r in page["items"]] == ["alpha 2?", "alpha 3?"] and page["total"] == 6
    by_src = client.get(f"/api/projects/{pid}/data-prep/qa/queue", params={"source_id": "s-b"}).json()
    assert [r["question"] for r in by_src["items"]] == ["needle question?"]
    found = client.get(f"/api/projects/{pid}/data-prep/qa/queue", params={"q": "NEEDLE"}).json()
    assert found["total"] == 1


def test_listing_sees_edits_made_outside_the_cache_and_hands_out_copies(project_env) -> None:
    _client, pid, projects = project_env
    qid = _pair(pid, "s-x", 1, "before?")
    listed = qa_fs.list_qa_pairs(pid)
    listed[0]["question"] = "mutated by a caller"
    assert qa_fs.list_qa_pairs(pid)[0]["question"] == "before?"
    path = projects / pid / "qa" / "pairs" / f"{qid}.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "question": "edited on disk by another process"}), encoding="utf-8")
    assert qa_fs.list_qa_pairs(pid)[0]["question"] == "edited on disk by another process"
    path.unlink()
    assert qa_fs.list_qa_pairs(pid) == []


def test_bulk_stamps_a_human_verdict_keeps_the_reason_and_returns_counts(project_env) -> None:
    client, pid, _ = project_env
    a, b = _pair(pid, "s", 1, "a?"), _pair(pid, "s", 1, "b?")
    r = client.post(f"/api/projects/{pid}/data-prep/qa/bulk", json={"ids": [a], "action": "reject", "note": "wrong value"}).json()
    assert r["updated"] == 1 and r["counts"] == {"pending": 1, "approved": 0, "rejected": 1}
    rejected = next(p for p in qa_fs.list_qa_pairs(pid) if p["id"] == a)
    assert rejected["status"] == "rejected" and rejected["note"] == "wrong value" and rejected["reviewed_at"] > 0
    assert "reviewed_at" not in next(p for p in qa_fs.list_qa_pairs(pid) if p["id"] == b)
    undone = client.post(f"/api/projects/{pid}/data-prep/qa/bulk", json={"ids": [a, "missing-id"], "action": "reset"}).json()
    assert undone["updated"] == 1 and undone["counts"]["pending"] == 2


def test_patch_with_a_status_stamps_the_verdict_but_a_text_edit_does_not(project_env) -> None:
    client, pid, _ = project_env
    qid = _pair(pid, "s", 1, "q?")
    edited = client.patch(f"/api/projects/{pid}/data-prep/qa/{qid}", json={"answer": "better answer"}).json()
    assert edited["answer"] == "better answer" and "reviewed_at" not in edited
    done = client.patch(f"/api/projects/{pid}/data-prep/qa/{qid}", json={"status": "approved"}).json()
    assert done["status"] == "approved" and done["reviewed_at"] > 0


def test_context_returns_the_whole_chunk_not_the_truncated_copy(project_env) -> None:
    client, pid, _ = project_env
    sha = "ab" * 32
    _source(pid, "s1", "doc.txt", sha)
    long_chunk = "fact " * 600  # 3,000 chars; the pair stores only the first 1,500
    write_chunks(pid, sha, [long_chunk, "second"])
    qid = _pair(pid, "s1", 1, "q?", sha256=sha)
    ctx = client.get(f"/api/projects/{pid}/data-prep/qa/{qid}/context").json()
    assert ctx["chunk_text"] == long_chunk and ctx["chunk_idx"] == 1 and ctx["chunk_total"] == 2
    assert ctx["pair"]["id"] == qid
    assert client.get(f"/api/projects/{pid}/data-prep/qa/nope/context").status_code == 404


def test_reviewer_written_pair_is_approved_human_review_with_the_chunk_as_context(project_env) -> None:
    client, pid, _ = project_env
    sha = "cd" * 32
    _source(pid, "s2", "doc.txt", sha)
    write_chunks(pid, sha, ["The gate closes at 22:00."])
    r = client.post(f"/api/projects/{pid}/data-prep/qa",
                    json={"source_id": "s2", "chunk_idx": 1, "question": "When does the gate close?", "answer": "At 22:00."})
    assert r.status_code == 200, r.text
    made = r.json()
    assert made["status"] == "approved" and made["origin"] == "human_review" and made["reviewed_at"] > 0
    assert made["chunk_text"] == "The gate closes at 22:00."
    assert client.post(f"/api/projects/{pid}/data-prep/qa", json={"source_id": "s2", "chunk_idx": 9, "question": "q", "answer": "a"}
                       ).status_code == 400
    assert client.post(f"/api/projects/{pid}/data-prep/qa", json={"source_id": "nope", "chunk_idx": 1, "question": "q", "answer": "a"}
                       ).status_code == 404
    assert client.post(f"/api/projects/{pid}/data-prep/qa", json={"source_id": "s2"}).status_code == 400


def test_queue_routes_404_for_an_unknown_project(project_env) -> None:
    client, _pid, _ = project_env
    assert client.get("/api/projects/does-not-exist/data-prep/qa/queue").status_code == 404


def test_review_page_is_wired_for_keyboard_review_without_refetching_per_verdict() -> None:
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates/data_prep.html").read_text(encoding="utf-8")
    assert "/qa/queue" in src and "/context" in src and "rqVerdict" in src and "rqKey" in src
    verdict = src.split("function rqVerdict", 1)[1].split("function rqPatchSourceCounts", 1)[0]
    assert "rqLoad(" not in verdict and "prepRefreshResults(" not in verdict  # a verdict updates the screen locally, then saves
    assert "removeEventListener('keydown', rqKey)" in src  # the SPA must not keep the shortcuts after leaving the page
