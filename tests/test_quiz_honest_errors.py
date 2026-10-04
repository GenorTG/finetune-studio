"""Quiz questions are whole clauses, judge reasoning is never blank, errors are honest."""
from __future__ import annotations

from finetune_studio.data.prep.coverage_fill import _make_pairs_from_chunk
from finetune_studio.testing.suite import (
    CaseResult,
    apply_heuristic_judging,
    fallback_reasoning,
)


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


def test_fallback_reasoning_variants():
    assert fallback_reasoning("pass", "because") == "because"
    assert "failed" in fallback_reasoning("", "", error="boom")
    assert "no verdict" in fallback_reasoning("", "")
    assert "without an explanation" in fallback_reasoning("pass", "", judge="heuristic")


def test_heuristic_judging_always_sets_reasoning():
    ok = CaseResult(
        case_name="c", category="x", question="q?", correct_answer="Paris",
        model_answer="It is Paris.",
    )
    bad = CaseResult(
        case_name="e", category="x", question="q?", correct_answer="Paris",
        model_answer="", error="engine down",
    )
    apply_heuristic_judging([ok, bad])
    assert ok.judge_reasoning.strip()
    assert bad.judge_reasoning.strip()


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
