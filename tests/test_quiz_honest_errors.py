"""Quiz questions are whole clauses, a judge always explains itself or records why it could not, errors are honest."""
from __future__ import annotations

import json

from finetune_studio import db
from finetune_studio.data.prep.coverage_fill import _make_pairs_from_chunk
from finetune_studio.db import judgements as jdb
from finetune_studio.testing.judge import LoadedJudge
from finetune_studio.testing.judging import judge_benchmark


def test_extractive_questions_have_no_ellipsis_fragment():
    chunk = (
        "The Rhine river flows north from the Swiss Alps through Germany to the "
        "North Sea, and it is a major trade route for central Europe. "
        "Mount Everest is the highest mountain above sea level on Earth, "
        "located in the Himalayas on the border of Nepal and China."
    )
    pairs = _make_pairs_from_chunk(chunk, seen_questions=set())
    assert pairs
    for q, _a in pairs:
        assert "…" not in q and "..." not in q, q


def test_unusable_sentence_is_skipped_not_fragmented():
    assert _make_pairs_from_chunk("Yes. See above. OK then.", seen_questions=set()) == []


def _saved_one_case_run() -> tuple[str, str]:
    pid = db.create_project(name="p")["id"]
    rid = db.create_run(pid, "run")["id"]
    bid = db.create_benchmark(rid, "quiz", {}, status="running", kind="suite")["id"]
    cid = db.create_case(bid, rid, "c", "x", "q?", "Paris", "It is Paris.", [])
    return bid, cid


def test_a_failed_judge_call_records_why_instead_of_a_blank_reasoning():
    bid, cid = _saved_one_case_run()

    def down(_messages):
        raise ConnectionError("provider down")

    judge = LoadedJudge(chat=down, provider_id="fake", model="fake", label="Fake", concurrent=False)
    assert judge_benchmark(bid, judge).failed == 1
    row = jdb.list_judgements(case_id=cid)[0]
    assert row["verdict"] == "" and "provider down" in row["error"]
    assert db.get_case(cid)["verdict"] == ""  # unjudged, not failed


def test_a_judge_verdict_always_arrives_with_its_reasoning():
    bid, cid = _saved_one_case_run()
    reply = json.dumps({"reasoning": "key fact Paris is present", "verdict": "pass"})
    judge = LoadedJudge(chat=lambda _m: reply, provider_id="fake", model="fake", label="Fake", concurrent=False)
    judge_benchmark(bid, judge)
    assert db.get_case(cid)["judge_reasoning"].strip()


def test_there_is_no_scripted_reasoning_fallback_left():
    import finetune_studio.testing.suite as suite_mod

    assert not hasattr(suite_mod, "fallback_reasoning") and not hasattr(suite_mod, "apply_heuristic_judging")


def test_training_start_bad_body_is_400(client):
    r = client.post("/api/training/start", json=[1])
    assert r.status_code == 400
    assert r.json()["error"]


def test_testing_load_unknown_project_is_not_200(client):
    r = client.post("/api/projects/nope-does-not-exist/testing/load", json={})
    assert r.status_code >= 400
    body = r.json()
    assert body.get("error") or body.get("detail")


def test_memory_estimate_empty_body_is_not_200(client):
    r = client.post("/api/inference/memory-estimate", json={})
    assert r.status_code >= 400
    body = r.json()
    assert body.get("error") or body.get("detail")
