"""``knowledge.index``: the project index reads the same parse training uses, and knows when it is stale."""
from __future__ import annotations

import pytest

from finetune_studio import db
from finetune_studio.data.fs import file_library as fl
from finetune_studio.knowledge import index, sources
from finetune_studio.knowledge.errors import IndexBuildError
from tests.knowledge_helpers import add_source, wait_build
from tests.test_brain_rebuild_sources import _fake_get_embedder


@pytest.fixture
def pid(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr("finetune_studio.smart_brain.store.get_embedder", _fake_get_embedder)
    return db.create_project(name="index", description="")["id"]


def _indexed_text(pid: str) -> str:
    from finetune_studio.data.fs.paths import brain_index_dir
    from finetune_studio.smart_brain.io import try_import_pandas

    df = try_import_pandas().read_parquet(brain_index_dir(pid) / "chunks.parquet")
    return "\n".join(df["text"].tolist())


def test_build_indexes_each_ready_source_once_under_its_own_name(pid: str) -> None:
    a = add_source(pid, "helios_notes.txt", "Helios notes: the reactor runs at 41 MW. " * 10)
    b = add_source(pid, "ops_runbook.md", "# Runbook\n\nRestart the pumps before 06:00. " * 10)
    add_source(pid, "later.txt", "Not parsed yet.", parse=False)
    result = index.build(pid, embedder="fake-deterministic")
    assert result["documents"] == 2
    from finetune_studio.smart_brain import BrainIndex

    listed = BrainIndex(result["corpus_dir"]).list_sources()
    assert {s["id"] for s in listed} == {a.id, b.id}
    assert {s["filename"] for s in listed} == {"helios_notes.txt", "ops_runbook.md"}
    st = index.state(pid)
    assert st.built and st.fresh and set(st.indexed) == {a.id, b.id}
    rags = db.list_brain_indexes(pid)
    assert len(rags) == 1 and rags[0]["doc_count"] == 2           # the chat attaches to this row


def test_the_index_reads_vision_readings_and_hand_edits(pid: str) -> None:
    """HANDOFF 2026-10-14 next step 1: the index re-parsed files and never saw what the Vision reader wrote."""
    src = add_source(pid, "annual.txt", "[Figure 1 · page 3 · picture · read by OCR, not checked]\n84.3 68.8\n[End]")
    sources.set_text(pid, src.id, "Revenue 2021 was EUR 84.3m; 2020 was EUR 68.8m.", parser="pdf_v1+vision")
    index.build(pid, embedder="fake-deterministic")
    assert "Revenue 2021 was EUR 84.3m" in _indexed_text(pid)


def test_state_reports_stale_missing_and_removed_until_rebuilt(pid: str) -> None:
    a = add_source(pid, "a.txt", "File A says the gate closes at 22:00.")
    b = add_source(pid, "b.txt", "File B says the keeper is Ada Morrow.")
    assert index.state(pid).as_dict()["built"] is False
    index.build(pid, embedder="fake-deterministic")
    sources.set_text(pid, a.id, "File A now says the gate closes at 22:30.")
    c = add_source(pid, "c.txt", "File C arrived after the build.")
    fl.soft_delete_file(pid, b.file_id)
    st = index.state(pid)
    assert (st.stale, st.missing, st.removed, st.fresh) == ([a.id], [c.id], [b.id], False)
    index.build(pid, embedder="fake-deterministic")
    st = index.state(pid)
    assert st.fresh and set(st.indexed) == {a.id, c.id}


def test_nothing_parsed_is_a_400_and_an_unknown_project_a_404(pid: str) -> None:
    add_source(pid, "later.txt", "Not parsed yet.", parse=False)
    with pytest.raises(IndexBuildError) as e:
        index.build(pid)
    assert e.value.status == 400 and "no parsed files" in str(e.value)
    with pytest.raises(IndexBuildError) as e:
        index.build("nope")
    assert e.value.status == 404


def test_build_routes_answer_with_the_reason(pid: str, client) -> None:
    r = client.post(f"/api/projects/{pid}/brain/build", json={})
    assert r.status_code == 400 and "no parsed files" in r.json()["detail"]
    add_source(pid, "a.txt", "File A says the gate closes at 22:00.")
    r = client.post(f"/api/projects/{pid}/brain/build", json={"embedder": "fake-deterministic"})
    b = wait_build(client, pid, r)
    assert b["ok"] and b["documents"] == 1, b
    status = client.get(f"/api/projects/{pid}/brain").json()
    assert status["exists"] and status["state"]["fresh"]


def test_fact_cards_index_approved_pairs_and_report_changes(pid: str) -> None:
    from finetune_studio.data.fs import qa as qa_fs

    src = add_source(pid, "gate.txt", "The gate of Saltmere closes at 22:00 every night.")
    qa_fs.write_qa_pair(pid, {"id": "q1", "source_id": src.id, "chunk_idx": 1, "question": "When does the gate close?",
                              "answer": "At 22:00.", "status": "approved"})
    qa_fs.write_qa_pair(pid, {"id": "q2", "source_id": src.id, "chunk_idx": 1, "question": "Is it unreviewed?",
                              "answer": "Yes.", "status": "pending"})
    off = index.build(pid, embedder="fake-deterministic")
    assert off["fact_cards"] is False and off["fact_card_count"] == 0
    assert "Q: When does the gate close?" not in _indexed_text(pid)

    on = index.build(pid, embedder="fake-deterministic", fact_cards=True)
    assert on["fact_card_count"] == 1                                  # approved only
    text = _indexed_text(pid)
    assert "Q: When does the gate close?\nA: At 22:00." in text and "unreviewed" not in text
    st = index.state(pid)
    assert st.fresh and st.fact_cards and st.indexed == [src.id]       # the cards never count as a file

    qa_fs.update_qa_pair(pid, "q2", status="approved")
    assert index.state(pid).facts_changed and not index.state(pid).fresh
    again = index.build(pid, embedder="fake-deterministic")            # no flag: keeps fact cards on
    assert again["fact_cards"] is True and again["fact_card_count"] == 2 and index.state(pid).fresh


def test_brain_page_offers_fact_cards_off_by_default(pid: str, client) -> None:
    body = client.get(f"/projects/{pid}/brain").text
    toggle = body.split('id="b-fact-cards"', 1)[0].rsplit("<input", 1)[1] + body.split('id="b-fact-cards"', 1)[1].split(">", 1)[0]
    assert "checked" not in toggle and "fact_cards: !!" in body and "reviewed facts changed" in body


def test_the_index_rows_are_the_mining_passages(pid: str) -> None:
    """One chunking for the app (A/B 2026-10-14: top-5 recall 95/102 with the passages vs 89/102 with 400-token chunks)."""
    import json

    from finetune_studio.data.fs.paths import brain_index_dir
    from finetune_studio.smart_brain.io import try_import_pandas

    long_text = "\n\n".join(f"Section {i}: the keeper of gate {i} is warden number {i * 7}." * 6 for i in range(1, 12))
    src = add_source(pid, "long.txt", long_text)
    index.build(pid, embedder="fake-deterministic")
    rows = try_import_pandas().read_parquet(brain_index_dir(pid) / "chunks.parquet")["text"].tolist()
    assert rows == [p for p in sources.passages(pid, src.sha256) if p.strip()]
    manifest = json.loads((brain_index_dir(pid) / "manifest.json").read_text())
    assert manifest["chunk_settings"]["splitter"] == "passages"


def test_a_rebuild_keeps_the_chosen_embedder_and_retrieval_settings(pid: str, client) -> None:
    """Regression (2026-10-21): every Build / Quick index without an explicit embedder re-embedded with
    DEFAULT_EMBEDDER and reset rerank/hybrid to defaults, so a project switched to another embedder (Settings, then
    Re-embed) silently went back to e5 on its next rebuild."""
    import json

    from finetune_studio.data.fs.paths import brain_index_dir

    add_source(pid, "a.txt", "File A says the gate closes at 22:00.")
    index.build(pid, embedder="fake-deterministic")
    r = client.post(f"/api/projects/{pid}/brain/settings",
                    json={"embedder": "fake-chosen", "rerank_enabled": False, "rerank_top_n": 20})
    assert r.status_code == 200, r.text
    for reset in (False, True):
        index.build(pid, reset=reset)
        m = json.loads((brain_index_dir(pid) / "manifest.json").read_text())
        rs = m["rag_settings"]
        assert m["embedding_model"]["name"] == "fake-chosen" and rs["embedder"] == "fake-chosen", (reset, m)
        assert rs["rerank_enabled"] is False and rs["rerank_top_n"] == 20

