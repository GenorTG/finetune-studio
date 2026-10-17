"""``knowledge.search``: find facts (Q&A pairs) and passages before training or indexing, then edit them in place."""
from __future__ import annotations

import pytest

from finetune_studio import db
from finetune_studio.data.fs import qa as qa_fs
from finetune_studio.knowledge import search
from tests.knowledge_helpers import add_source

GATE = "The gate of Saltmere closes at 22:00 every night.\n\nThe keeper is Ada Morrow, appointed in 1841."
LEDGER = "The ledger lists 41 barrels of salt and 3 casks of wax for the winter store."


@pytest.fixture
def pid() -> str:
    p = db.create_project(name="search", description="")["id"]
    gate = add_source(p, "gate.txt", GATE)
    add_source(p, "ledger.txt", LEDGER)
    qa_fs.write_qa_pair(p, {"id": "q1", "source_id": gate.id, "chunk_idx": 1, "question": "When does the gate close?",
                            "answer": "At 22:00 every night.", "status": "approved"})
    qa_fs.write_qa_pair(p, {"id": "q2", "source_id": gate.id, "chunk_idx": 1, "question": "Who is the keeper?",
                            "answer": "Ada Morrow.", "status": "pending"})
    return p


def test_facts_and_passages_are_ranked_together(pid: str) -> None:
    hits = search.search(pid, "when does the gate close")
    assert hits[0].kind == "fact" and hits[0].pair_id == "q1" and hits[0].filename == "gate.txt"
    passage = next(h for h in hits if h.kind == "passage")
    assert passage.filename == "gate.txt" and passage.chunk_idx == 1 and "22:00" in passage.text and passage.file_id
    assert all(h.filename != "ledger.txt" for h in hits)          # no shared word, no hit


def test_kind_and_status_filters(pid: str) -> None:
    assert {h.kind for h in search.search(pid, "keeper Ada", kinds=("passage",))} == {"passage"}
    assert search.search(pid, "keeper Ada", kinds=("fact",), status="approved") == []
    assert [h.pair_id for h in search.search(pid, "keeper Ada", kinds=("fact",), status="pending")] == ["q2"]
    assert search.search(pid, "   ") == []


def test_route_validates_and_answers(pid: str, client) -> None:
    r = client.get(f"/api/projects/{pid}/knowledge/search", params={"q": "salt barrels"})
    assert r.status_code == 200, r.text
    hits = r.json()["hits"]
    assert hits and hits[0]["kind"] == "passage" and hits[0]["filename"] == "ledger.txt"
    assert client.get(f"/api/projects/{pid}/knowledge/search", params={"q": "x"}).status_code == 400
    assert client.get(f"/api/projects/{pid}/knowledge/search", params={"q": "salt", "kinds": "chunk"}).status_code == 400
    assert client.get("/api/projects/nope/knowledge/search", params={"q": "salt"}).status_code == 404


def test_a_fact_found_by_search_is_edited_in_place(pid: str, client) -> None:
    hit = client.get(f"/api/projects/{pid}/knowledge/search", params={"q": "keeper", "kinds": "fact"}).json()["hits"][0]
    r = client.patch(f"/api/projects/{pid}/data-prep/qa/{hit['pair_id']}",
                     json={"answer": "Ada Morrow, keeper since 1841.", "status": "approved"})
    assert r.status_code == 200
    again = client.get(f"/api/projects/{pid}/knowledge/search", params={"q": "keeper 1841", "kinds": "fact"}).json()
    assert again["hits"][0]["answer"] == "Ada Morrow, keeper since 1841." and again["hits"][0]["status"] == "approved"


def test_files_page_searches_the_knowledge_and_edits_facts(client) -> None:
    pid = db.create_project(name="search-ui", description="")["id"]
    body = client.get(f"/projects/{pid}/data").text
    assert "search inside files" in body and "/knowledge/search?limit=40&q=" in body
    assert "window.knSave" in body and "/data-prep/qa/" in body and "Edit parsed text" in body
    assert "/files/search-content" not in body
