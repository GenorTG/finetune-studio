"""Near-duplicate pair detection + the API route that rejects pending duplicates."""
from __future__ import annotations

from typing import Any

from finetune_studio.data.prep.dedupe import find_near_duplicates


def _p(i: str, q: str, a: str, chunk: int = 1, src: str = "s1") -> dict[str, Any]:
    return {"id": i, "question": q, "answer": a, "chunk_idx": chunk, "source_id": src}


def test_reworded_same_fact_is_a_duplicate_but_a_new_value_is_not() -> None:
    pairs = [
        _p("a", "What is the stage of the Nordkjøl Seafood AS deal (OPP-24-0302)?", "The stage of the Nordkjøl Seafood AS deal (OPP-24-0302) is Negotiation."),
        _p("b", "What is the stage of the Nordkjøl Seafood AS opportunity (OPP-24-0302)?", "The stage of Nordkjøl Seafood AS (OPP-24-0302) is Negotiation."),
        _p("c", "What is the close_date for Nordkjøl Seafood AS (OPP-24-0302)?", "The close_date for Nordkjøl Seafood AS (OPP-24-0302) is 2024-09-27."),
        _p("d", "What is the stage of the Nordkjøl Seafood AS deal (OPP-24-0302)?", "The stage is Negotiation and the amount is 63685.20."),
    ]
    assert find_near_duplicates(pairs) == [("b", "a")]


def test_other_chunks_and_sources_are_never_merged() -> None:
    q, a = "What is the notice period for managers under the handbook?", "The notice period for managers is 3 months."
    assert find_near_duplicates([_p("a", q, a, chunk=1), _p("b", q, a, chunk=2), _p("c", q, a, src="s2")]) == []


def test_route_dry_run_then_apply(client: Any) -> None:
    from finetune_studio import db
    from finetune_studio.data import project_filesystem as pfs

    pid = db.create_project(name="dd", description="")["id"]
    for i, (q, a) in enumerate([
        ("What is the stage of the Acme deal (OPP-1)?", "The stage of the Acme deal (OPP-1) is Won."),
        ("What is the stage of the Acme opportunity (OPP-1)?", "The stage of Acme (OPP-1) is Won."),
    ]):
        pfs.write_qa_pair(pid, {"id": f"p{i}", "source_id": "s", "chunk_idx": 1, "question": q, "answer": a,
                                "status": "pending", "created_at": float(i)})
    dry = client.post(f"/api/projects/{pid}/data-prep/dedupe", json={"dry_run": True}).json()
    assert dry["rejected"] == 1 and {p["status"] for p in pfs.list_qa_pairs(pid)} == {"pending"}
    done = client.post(f"/api/projects/{pid}/data-prep/dedupe", json={}).json()
    assert done["rejected"] == 1
    by = {p["id"]: p for p in pfs.list_qa_pairs(pid)}
    assert by["p0"]["status"] == "pending" and by["p1"]["status"] == "rejected" and "p0" in by["p1"]["note"]
