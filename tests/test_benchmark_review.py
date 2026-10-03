"""Benchmark review from the UI: open any result, override verdicts, re-check, delete.

Gaps (UI coverage audit 2026-10-03): only the latest benchmark's cases were
viewable; the verdict-override, audit and delete routes had no UI. Bugs found
alongside: no verdict change except the heuristic re-judge recomputed the
benchmark's headline score (stale "Recent scores"), that re-judge dropped
training-eval metadata from the scores, any string was accepted as a verdict,
and deleting a benchmark left its cases orphaned.
"""

from __future__ import annotations

import pytest

from finetune_studio import db
from tests import test_bench_judge as _bj
from tests.test_bench_judge import _create_project_and_run

client_and_db = _bj.client_and_db  # re-exported pytest fixture


def _case(name: str, verdict: str, answer: str = "Paris") -> dict:
    return {"name": name, "question": f"{name}: capital of France?", "correct_answer": "Paris",
            "model_answer": answer, "verdict": verdict, "judge": "heuristic", "category": "geo"}


def _bench(rid: str, verdicts: list[str], **meta) -> dict:
    cases = [_case(f"case-{i}", v) for i, v in enumerate(verdicts)]
    passed = verdicts.count("pass")
    scores = {"total": len(cases), "judged": len(cases), "passed": passed,
              "pass_rate": round(passed / len(cases) * 100, 1), **meta}
    return db.create_benchmark(rid, "suite-x", scores, cases=cases)


def _scores(bid: str) -> dict:
    b = db.get_benchmark(bid)
    return b.get("scores") or {}


def test_human_verdict_override_rescores_and_keeps_metadata(client_and_db, tmp_path):
    client, _ = client_and_db
    pid, rid = _create_project_and_run(client, tmp_path)
    b = _bench(rid, ["pass", "fail"], eval_kind="training_set", leakage_warning="trained on these")
    cid = next(c["id"] for c in db.list_cases(b["id"]) if c["verdict"] == "fail")
    r = client.post(f"/api/benchmarks/projects/{pid}/benchmarks/{b['id']}/cases/{cid}/verdict",
                    json={"verdict": "pass", "reasoning": "answer is right, judge missed a synonym"})
    assert r.status_code == 200, r.text
    assert r.json()["scores"]["pass_rate"] == 100.0
    s = _scores(b["id"])
    assert s["pass_rate"] == 100.0 and s["passed"] == 2
    assert s["eval_kind"] == "training_set" and s["leakage_warning"] == "trained on these"
    case = next(c for c in db.list_cases(b["id"]) if c["id"] == cid)
    assert case["judge"] == "human" and "synonym" in case["judge_reasoning"]


@pytest.mark.parametrize("bad", ["", "maybe", None, 3])
def test_invalid_verdict_is_400(client_and_db, tmp_path, bad):
    client, _ = client_and_db
    pid, rid = _create_project_and_run(client, tmp_path)
    b = _bench(rid, ["pass"])
    cid = db.list_cases(b["id"])[0]["id"]
    r = client.post(f"/api/benchmarks/projects/{pid}/benchmarks/{b['id']}/cases/{cid}/verdict",
                    json={"verdict": bad})
    assert r.status_code == 400, r.text
    assert db.list_cases(b["id"])[0]["verdict"] == "pass"


def test_heuristic_rejudge_keeps_training_eval_metadata(client_and_db, tmp_path):
    client, _ = client_and_db
    pid, rid = _create_project_and_run(client, tmp_path)
    b = _bench(rid, ["fail"], eval_kind="holdout", dataset_name="faq")
    r = client.post(f"/api/benchmarks/projects/{pid}/benchmarks/{b['id']}/judge",
                    json={"judge_mode": "heuristic"})
    assert r.status_code == 200, r.text
    s = _scores(b["id"])
    assert s["eval_kind"] == "holdout" and s["dataset_name"] == "faq"


def test_delete_benchmark_removes_its_cases(client_and_db, tmp_path):
    client, _ = client_and_db
    pid, rid = _create_project_and_run(client, tmp_path)
    b = _bench(rid, ["pass", "fail"])
    r = client.delete(f"/api/benchmarks/projects/{pid}/benchmarks/{b['id']}")
    assert r.status_code == 200, r.text
    assert db.get_benchmark(b["id"]) is None
    assert db.list_cases(b["id"]) == []


def test_page_opens_any_benchmark_by_id(client_and_db, tmp_path):
    client, _ = client_and_db
    pid, rid = _create_project_and_run(client, tmp_path)
    older = _bench(rid, ["pass"])
    db.update_case(db.list_cases(older["id"])[0]["id"], question="OLDER-ONLY QUESTION")
    _bench(rid, ["fail"])  # newer: shown by default
    default = client.get(f"/projects/{pid}/benchmarks").text
    assert "OLDER-ONLY QUESTION" not in default
    html = client.get(f"/projects/{pid}/benchmarks?bid={older['id']}").text
    assert "OLDER-ONLY QUESTION" in html
    assert f"?bid={older['id']}" in html  # recent-scores rows link to their cases
    assert 'class="select verdict-select"' in html  # human override control, themed
    assert "Re-check score" in html and "Delete this result" in html


def test_unknown_bid_falls_back_to_latest(client_and_db, tmp_path):
    client, _ = client_and_db
    pid, rid = _create_project_and_run(client, tmp_path)
    _bench(rid, ["pass"])
    r = client.get(f"/projects/{pid}/benchmarks?bid=nope")
    assert r.status_code == 200 and "case-0" in r.text


# ── sample size: the UI offers a full run instead of pointing at the API ─────


def test_num_samples_full_means_full_run() -> None:
    from finetune_studio.webui.routes.benchmarks import _parse_sample_knobs

    assert _parse_sample_knobs({"num_samples": "full"})[:2] == (None, True)
    assert _parse_sample_knobs({"num_samples": "100"})[:2] == (100, False)
    assert _parse_sample_knobs({})[1] is False


def test_benchmarks_page_offers_full_run_without_api_jargon() -> None:
    from pathlib import Path

    from finetune_studio import webui

    html = (Path(webui.__file__).parent / "templates" / "benchmarks.html").read_text()
    assert html.count('<option value="full">') == 2  # trained-run and base-model rows
    assert "run API" not in html and "full_run via API" not in html
