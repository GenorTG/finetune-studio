"""Export gate: extractive ``coverage_fill`` pairs are opt-in, never auto-approved.

Before: every export approved extractive pairs for each chunk without an approved pair, so
unreviewed text reached training. Now they stay pending and out of the dataset unless the
caller passes ``include_unreviewed_fill`` (route query / CLI flag / UI checkbox).
"""
from __future__ import annotations

import json

import pytest

from finetune_studio.data import project_filesystem as pfs
from finetune_studio.data.prep import dataset_build as dsb
from finetune_studio.data.prep.coverage_fill import (
    fill_all_project_gaps,
    fill_coverage_gaps,
)

CHARTER = (
    "Charter of the Ledger-Keepers of the Vaelindrath Concord. "
    "A Ledger-Keeper is sworn at the age of 231 and serves 158 years. "
    "Their oath-stone is carved from margin-stone. "
    "The Concord pays a Ledger-Keeper 1875 crowns per annum plus 71 measures of black-whiskey. "
    "Dismissal requires the countersignature of a Salt-Speaker and two Ledger-Keepers."
)
SHA = "optinsha0001"
SID = "optinsrc0001"


@pytest.fixture()
def env(project_env):
    """(client, pid): one source with two chunks; chunk 1 already has a reviewed, approved pair."""
    client, pid, _projects = project_env
    pfs.write_qa_source(pid, {
        "id": SID, "sha256": SHA, "filename": "charter.txt", "mime_type": "text/plain",
        "char_count": len(CHARTER) * 2, "chunk_count": 2, "parser": "text_v1", "status": "ready",
        "data_path": "", "path": "",
    })
    chunks = pfs.file_dir(pid, SHA) / "chunks"
    chunks.mkdir()
    (chunks / "0000.txt").write_text("Reviewed chunk. The reviewed fact is VALUE-ONE and nothing else.", encoding="utf-8")
    (chunks / "0001.txt").write_text(CHARTER, encoding="utf-8")
    pfs.write_qa_pair(pid, {
        "id": "reviewed01", "source_id": SID, "chunk_idx": 1, "question": "What is the reviewed fact?",
        "answer": "VALUE-ONE.", "status": "approved", "category": "source-grounded",
    })
    return client, pid


def _fill_rows(pid: str) -> list[dict]:
    return [r for r in pfs.list_qa_pairs(pid, source_id=SID) if r.get("origin") == "coverage_fill"]


# ── fill modes ────────────────────────────────────────────────────────

def test_default_fill_leaves_pairs_pending(env) -> None:
    _client, pid = env
    summary = fill_all_project_gaps(pid)
    rows = _fill_rows(pid)
    assert summary["pairs_created"] == len(rows) > 0
    assert {r["status"] for r in rows} == {"pending"}
    assert {r["chunk_idx"] for r in rows} == {2}          # chunk 1 was already covered
    assert summary["mode"] == "pending"


def test_repeated_default_passes_do_not_pile_up_pending_pairs(env) -> None:
    _client, pid = env
    fill_all_project_gaps(pid)
    before = len(_fill_rows(pid))
    for _ in range(3):
        assert fill_all_project_gaps(pid)["pairs_created"] == 0
    assert len(_fill_rows(pid)) == before


def test_count_mode_writes_nothing_and_matches_what_approving_adds(env) -> None:
    _client, pid = env
    fresh = fill_all_project_gaps(pid, "count")
    assert _fill_rows(pid) == [] and fresh["pairs_would_add"] > 0
    fill_all_project_gaps(pid)                              # now pending pairs exist
    waiting = fill_all_project_gaps(pid, "count")
    assert waiting["pairs_would_add"] == fresh["pairs_would_add"]
    assert len(_fill_rows(pid)) == fresh["pairs_would_add"]
    approved = fill_all_project_gaps(pid, "approve")
    assert approved["pairs_promoted"] == fresh["pairs_would_add"]


def test_approve_mode_promotes_pending_pairs_without_duplicates(env) -> None:
    _client, pid = env
    fill_all_project_gaps(pid)
    pending_ids = {r["id"] for r in _fill_rows(pid)}
    summary = fill_all_project_gaps(pid, "approve")
    rows = _fill_rows(pid)
    assert {r["id"] for r in rows} == pending_ids           # same pairs, flipped, none added
    assert {r["status"] for r in rows} == {"approved"}
    assert summary["pairs_promoted"] == len(pending_ids) and summary["pairs_created"] == 0


def test_approve_mode_on_a_fresh_project_writes_approved_pairs(env) -> None:
    _client, pid = env
    summary = fill_all_project_gaps(pid, "approve")
    assert summary["pairs_created"] > 0 and summary["pairs_promoted"] == 0
    assert {r["status"] for r in _fill_rows(pid)} == {"approved"}


def test_rejected_fill_pairs_stay_rejected(env) -> None:
    _client, pid = env
    fill_all_project_gaps(pid)
    for r in _fill_rows(pid):
        pfs.update_qa_pair(pid, r["id"], status="rejected")
    summary = fill_all_project_gaps(pid, "approve")
    assert summary["pairs_promoted"] == 0 and summary["pairs_created"] == 0
    assert {r["status"] for r in _fill_rows(pid)} == {"rejected"}


def test_unfillable_chunk_is_still_reported_in_every_mode(env) -> None:
    _client, pid = env
    for mode in ("pending", "approve", "count"):
        result = fill_coverage_gaps(pid, "noisesrc", chunk_texts={1: "~~~ *** ###"}, mode=mode)
        assert result.chunks_still_uncovered and result.pairs_created == 0


# ── gate + dataset build ──────────────────────────────────────────────

def test_gate_off_keeps_fill_out_of_the_dataset(env) -> None:
    _client, pid = env
    built = dsb.build_project_dataset(pid, "sharegpt", "approved", grounding=None)
    answers = [json.loads(ln)["conversations"][-1]["value"] for ln in built.export.body.splitlines()]
    assert answers == ["VALUE-ONE."]                         # only the reviewed pair
    assert {r["status"] for r in _fill_rows(pid)} == {"pending"}


def test_gate_on_approves_and_exports_the_fill(env) -> None:
    _client, pid = env
    built = dsb.build_project_dataset(pid, "sharegpt", "approved", grounding=None, include_unreviewed_fill=True)
    assert built.export.rows == 1 + len(_fill_rows(pid)) > 1
    assert {r["status"] for r in _fill_rows(pid)} == {"approved"}
    assert built.coverage["mode"] == "approve"


def test_preview_counts_without_writing(env) -> None:
    _client, pid = env
    preview = dsb.unreviewed_fill_preview(pid)
    assert preview["pairs"] > 0 and preview["chunks"] == 1
    assert _fill_rows(pid) == []


# ── HTTP route ────────────────────────────────────────────────────────

def test_route_default_export_excludes_unreviewed_fill(env) -> None:
    client, pid = env
    r = client.get(f"/api/projects/{pid}/data-prep/export", params={"fmt": "sharegpt", "only": "approved"})
    assert r.status_code == 200, r.text
    assert r.headers["X-Rows"] == "1" and r.headers["X-Unreviewed-Fill-Pairs"] == "0"
    assert {x["status"] for x in _fill_rows(pid)} == {"pending"}


def test_route_opt_in_exports_the_fill_and_reports_the_count(env) -> None:
    client, pid = env
    would_add = client.get(f"/api/projects/{pid}/data-prep/export/fill-preview").json()["pairs"]
    assert would_add > 0
    r = client.get(f"/api/projects/{pid}/data-prep/export",
                   params={"fmt": "sharegpt", "only": "approved", "include_unreviewed_fill": "true"})
    assert r.status_code == 200, r.text
    assert r.headers["X-Rows"] == str(1 + would_add)
    assert r.headers["X-Unreviewed-Fill-Pairs"] == str(would_add)
    # once approved there is nothing left for the option to add
    assert client.get(f"/api/projects/{pid}/data-prep/export/fill-preview").json()["pairs"] == 0


def test_fill_preview_route_is_read_only_and_404s_unknown_project(env) -> None:
    client, pid = env
    body = client.get(f"/api/projects/{pid}/data-prep/export/fill-preview").json()
    assert body["pairs"] > 0 and _fill_rows(pid) == []
    assert client.get("/api/projects/nope/data-prep/export/fill-preview").status_code == 404


def test_export_page_exposes_the_option_with_a_count(env) -> None:
    client, pid = env
    html = client.get(f"/projects/{pid}/data-prep").text
    assert 'id="dp-fill-unreviewed"' in html and 'id="dp-fill-count"' in html
    assert "include_unreviewed_fill=true" in html and "export/fill-preview" in html


# ── CLI ───────────────────────────────────────────────────────────────

def test_cli_flag_is_opt_in(env, monkeypatch, capsys) -> None:
    import sys

    from finetune_studio.cli import _registry

    _client, pid = env

    def run(*extra: str) -> dict:
        monkeypatch.setattr(sys, "argv", ["fts", "dataset", "build", "--project", pid,
                                          "--no-rag-grounding", "--json", *extra])
        _registry.main()
        return json.loads(capsys.readouterr().out)

    off = run()
    assert off["rows"] == 1 and off["coverage_fill"]["mode"] == "pending"
    on = run("--include-unreviewed-fill")
    assert on["rows"] > 1 and on["coverage_fill"]["mode"] == "approve"
