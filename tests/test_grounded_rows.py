"""Context-grounded training rows (data/prep/grounding.py + export + route).

Evidence for the feature: .tmp/rag-grounding/RESULTS.md — a model tuned on
plain QA rows ignores rag/chat context (2/5); rows carrying the pair's own
source passage in the rag/chat prompt layout lifted it to 4/5.
"""
from __future__ import annotations

import json
import time
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db
from finetune_studio.config import settings
from finetune_studio.data.fs import qa as qa_fs
from finetune_studio.data.prep import grounding as g
from finetune_studio.data.prep.export import _to_jsonl, build_qa_export
from finetune_studio.data.rag_portable.prompt import (
    CONTEXT_MARKER,
    DEFAULT_SYSTEM_PROMPT,
    build_messages,
    build_system_prompt,
    format_context_blocks,
    synthetic_hit,
)
from finetune_studio.data.rag_portable.query import PortableRAGQuery
from finetune_studio.training.data import format_for_sft
from finetune_studio.webui.app import app

FILES = {"s1": "warranty.txt", "s2": "specs.txt", "s3": "history.txt"}


def _pair(i: int, source: str = "s1") -> dict:
    return {
        "id": f"p{i}", "source_id": source, "chunk_idx": i,
        "chunk_text": f"Chunk {i} of {source}: the fact number {i} is VALUE-{i}.",
        "question": f"What is fact {i} of {source}?", "answer": f"VALUE-{i}.",
        "category": "source-grounded", "status": "approved",
    }


def _items(n: int = 20) -> list[dict]:
    return [_pair(i, f"s{1 + i % 3}") for i in range(n)]


# ── shared layout ──────────────────────────────────────────────────────

def test_context_block_format_matches_portable_query() -> None:
    hits = [synthetic_hit(1, "a.txt", "alpha"), synthetic_hit(2, "b.txt", "beta")]
    assert PortableRAGQuery.format_context(SimpleNamespace(), hits) == format_context_blocks(hits)


def test_layout_is_instructions_then_context_marker_then_blocks() -> None:
    ctx = format_context_blocks([synthetic_hit(1, "warranty.txt", "Three years.")])
    system = build_system_prompt(ctx)
    assert system == f"{DEFAULT_SYSTEM_PROMPT}\n\n{CONTEXT_MARKER}{ctx}"
    assert ctx.startswith("[1] (source: warranty.txt, score 0.033)\nThree years.")
    msgs = build_messages([{"role": "user", "content": "q"}], ctx, "Be terse.")
    assert msgs[0] == {"role": "system", "content": f"Be terse.\n\n{CONTEXT_MARKER}{ctx}"}
    assert msgs[1:] == [{"role": "user", "content": "q"}]


# ── apply_grounding ────────────────────────────────────────────────────

def test_share_selects_exact_count_and_leaves_answers_unchanged() -> None:
    items = _items(20)
    out, stats = g.apply_grounding(items, g.GroundingOptions(share=0.4), FILES)
    assert (stats.rows, stats.grounded, stats.no_chunk_text) == (20, 8, 0)
    assert sum(1 for o in out if o.get("grounded")) == 8
    for src, new in zip(items, out, strict=True):
        assert (new["question"], new["answer"]) == (src["question"], src["answer"])
        if new.get("grounded"):
            assert new["system"].startswith(DEFAULT_SYSTEM_PROMPT + "\n\n" + CONTEXT_MARKER)
            assert src["chunk_text"] in new["system"]
            assert f"(source: {FILES[src['source_id']]}," in new["system"]
        else:
            assert "system" not in new


def test_selection_is_deterministic_and_seed_dependent() -> None:
    items = _items(30)
    a, _ = g.apply_grounding(items, g.GroundingOptions(share=0.4, seed=1), FILES)
    b, _ = g.apply_grounding(list(reversed(items)), g.GroundingOptions(share=0.4, seed=1), FILES)
    c, _ = g.apply_grounding(items, g.GroundingOptions(share=0.4, seed=2), FILES)
    def ids(rows: list[dict]) -> set[str]:
        return {r["id"] for r in rows if r.get("grounded")}

    assert ids(a) == ids(b)  # independent of input order
    assert ids(a) != ids(c)
    assert a == g.apply_grounding(items, g.GroundingOptions(share=0.4, seed=1), FILES)[0]


def test_share_extremes() -> None:
    items = _items(10)
    assert g.apply_grounding(items, g.GroundingOptions(share=1.0), FILES)[1].grounded == 10
    assert g.apply_grounding(items, g.GroundingOptions(share=0.0), FILES)[1].grounded == 0


def test_pair_without_chunk_text_stays_plain_and_is_counted() -> None:
    items = _items(4)
    for it in items:
        it["chunk_text"] = "  "
    out, stats = g.apply_grounding(items, g.GroundingOptions(share=1.0), FILES)
    assert (stats.grounded, stats.no_chunk_text) == (0, 4)
    assert out == items


def test_distractors_come_from_other_sources_and_gold_always_present() -> None:
    items = _items(18)
    out, stats = g.apply_grounding(items, g.GroundingOptions(share=1.0, distractors=2), FILES)
    assert stats.with_distractors == 18
    for src, new in zip(items, out, strict=True):
        system = new["system"]
        assert src["chunk_text"] in system
        assert system.count("(source: ") == 3
        own = FILES[src["source_id"]]
        assert system.count(f"(source: {own},") == 1  # the other two cite other files
        assert len(system.split(CONTEXT_MARKER, 1)[1]) < 4000


def test_distractors_skipped_when_only_one_source() -> None:
    items = [_pair(i, "s1") for i in range(5)]
    out, stats = g.apply_grounding(items, g.GroundingOptions(share=1.0, distractors=2), FILES)
    assert (stats.grounded, stats.with_distractors) == (5, 0)
    assert all(o["system"].count("(source: ") == 1 for o in out)


def test_long_chunk_is_clipped_under_the_context_cap() -> None:
    item = {**_pair(1), "chunk_text": "x" * 9000}
    out, _ = g.apply_grounding([item], g.GroundingOptions(share=1.0), FILES)
    ctx = out[0]["system"].split(CONTEXT_MARKER, 1)[1]
    assert 3000 < len(ctx) < 4000
    assert ctx.startswith("[1] (source: warranty.txt")


def test_options_validate() -> None:
    for bad in ({"share": 1.5}, {"share": -0.1}, {"distractors": 3}, {"distractors": -1}):
        with pytest.raises(ValueError):
            g.GroundingOptions(**bad)


def test_resolve_auto_follows_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(g, "corpus_built", lambda pid: True)
    assert g.resolve_grounding("p").share == g.DEFAULT_SHARE
    assert g.resolve_grounding("p", share=0) is None
    assert g.resolve_grounding("p", share=0.25, distractors=1) == g.GroundingOptions(0.25, 1)
    monkeypatch.setattr(g, "corpus_built", lambda pid: False)
    assert g.resolve_grounding("p") is None
    assert g.resolve_grounding("p", share=0.5).share == 0.5  # explicit wins without a corpus
    with pytest.raises(ValueError):
        g.resolve_grounding("p", share=2)


# ── serialisation + trainer round trip ─────────────────────────────────

@pytest.mark.parametrize("fmt", ["sharegpt", "openai", "alpaca"])
def test_every_format_carries_the_system_turn(fmt: str) -> None:
    out, _ = g.apply_grounding(_items(4), g.GroundingOptions(share=1.0), FILES)
    rows = [json.loads(ln) for ln in _to_jsonl(out, fmt).splitlines()]
    assert all(r["grounded"] is True for r in rows)
    if fmt == "sharegpt":
        assert rows[0]["conversations"][0]["from"] == "system"
        assert CONTEXT_MARKER in rows[0]["conversations"][0]["value"]
        assert rows[0]["conversations"][-1] == {"from": "gpt", "value": "VALUE-0."}
    elif fmt == "openai":
        assert [m["role"] for m in rows[0]["messages"]] == ["system", "user", "assistant"]
    else:
        assert CONTEXT_MARKER in rows[0]["system"]
        assert rows[0]["output"] == "VALUE-0."


def test_plain_rows_serialise_exactly_as_before() -> None:
    row = json.loads(_to_jsonl(_items(1), "sharegpt"))
    assert [t["from"] for t in row["conversations"]] == ["human", "gpt"]
    assert "grounded" not in row


def test_trainer_sees_system_user_assistant_for_grounded_and_plain() -> None:
    out, _ = g.apply_grounding(_items(10), g.GroundingOptions(share=0.5), FILES)
    rows = [json.loads(ln) for ln in _to_jsonl(out, "sharegpt").splitlines()]
    formatted = format_for_sft(rows)
    grounded = [f for f, r in zip(formatted, rows, strict=True) if r.get("grounded")]
    plain = [f for f, r in zip(formatted, rows, strict=True) if not r.get("grounded")]
    assert len(grounded) == 5 and len(plain) == 5
    assert all([m["role"] for m in f["messages"]] == ["system", "user", "assistant"] for f in grounded)
    assert all([m["role"] for m in f["messages"]] == ["user", "assistant"] for f in plain)


def test_build_qa_export_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    from finetune_studio.data.prep import export as ex

    monkeypatch.setattr(ex.pfs, "list_qa_pairs", lambda pid, status=None: _items(10))
    monkeypatch.setattr(ex.pfs, "list_qa_sources",
                        lambda pid: [{"id": k, "filename": v} for k, v in FILES.items()])
    res = build_qa_export("p", "openai", grounding=g.GroundingOptions(share=0.4))
    assert (res.rows, res.grounding.grounded) == (10, 4)
    assert res.body.count('"role": "system"') == 4
    plain = build_qa_export("p", "openai")
    assert plain.grounding is None and '"system"' not in plain.body
    sub = build_qa_export("p", "openai", source_ids=["s1"], grounding=g.GroundingOptions(share=1.0))
    assert sub.rows == sum(1 for i in _items(10) if i["source_id"] == "s1")


# ── route ──────────────────────────────────────────────────────────────

@pytest.fixture
def client_and_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "fts_test.db"))
    root = tmp_path / "fts_root"
    projects = root / "projects"
    projects.mkdir(parents=True)
    monkeypatch.setattr("finetune_studio.data.fs.paths._ROOT", root)
    monkeypatch.setattr("finetune_studio.data.fs.paths._PROJECTS", projects)
    monkeypatch.setattr(qa_fs, "project_dir", lambda pid: projects / pid)
    db.init_db()
    monkeypatch.setattr("finetune_studio.data.prep.coverage_fill.fill_all_project_gaps",
                        lambda pid: {"uncovered_chunks": []})
    client = TestClient(app)
    r = client.post("/api/projects", json={"name": f"grounded-{uuid.uuid4().hex[:6]}", "base_model": "x/test"})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    (projects / pid / "qa" / "pairs").mkdir(parents=True, exist_ok=True)
    for it in _items(10):
        qa_fs.write_qa_pair(pid, {**it, "id": uuid.uuid4().hex[:12], "created_at": time.time()})
    return client, pid


def _export(client: TestClient, pid: str, **params):
    return client.get(f"/api/projects/{pid}/data-prep/export", params={"fmt": "openai", **params})


def test_route_auto_is_on_with_corpus_and_off_without(client_and_pid, monkeypatch) -> None:
    client, pid = client_and_pid
    monkeypatch.setattr(g, "corpus_built", lambda p: True)
    r = _export(client, pid)
    assert r.status_code == 200, r.text
    assert (r.headers["X-Rows"], r.headers["X-Grounded-Rows"]) == ("10", "4")
    assert r.text.count('"role": "system"') == 4
    ds = client.get(f"/api/projects/{pid}/datasets").json()["datasets"]
    assert "4 with retrieved context" in ds[0]["name"]

    monkeypatch.setattr(g, "corpus_built", lambda p: False)
    r = _export(client, pid)
    assert r.headers["X-Grounded-Rows"] == "0" and '"role": "system"' not in r.text
    assert "retrieved context" not in client.get(f"/api/projects/{pid}/datasets").json()["datasets"][0]["name"]


def test_route_explicit_share_and_off_switch(client_and_pid, monkeypatch) -> None:
    client, pid = client_and_pid
    monkeypatch.setattr(g, "corpus_built", lambda p: True)
    assert _export(client, pid, grounded_share=0).headers["X-Grounded-Rows"] == "0"
    r = _export(client, pid, grounded_share=1, distractors=1)
    assert r.headers["X-Grounded-Rows"] == "10"
    assert all(ln.count("(source: ") == 2 for ln in r.text.splitlines())


def test_route_rejects_bad_values_with_400(client_and_pid) -> None:
    client, pid = client_and_pid
    assert _export(client, pid, grounded_share=1.5).status_code == 400
    assert _export(client, pid, distractors=5).status_code == 400


def test_data_prep_ui_has_honest_grounded_control(client_and_pid) -> None:
    client, pid = client_and_pid
    html = client.get(f"/projects/{pid}/data-prep").text
    assert 'id="dp-grounded"' in html and 'id="dp-grounded-pct"' in html
    assert "Only matters if you will chat with the trained model over this project's RAG corpus" in html
    assert "grounded_share=" in html and "X-Grounded-Rows" in html
