"""Per-question McNemar comparison of two runs (testing/paired.py) and its place in Benchmarks -> Compare."""
from __future__ import annotations

import pytest

from finetune_studio.testing.paired import (
    case_outcomes,
    compare_outcomes,
    mcnemar_exact_p,
)


@pytest.mark.parametrize(
    ("gained", "lost", "expected"),
    [
        (0, 0, 1.0),
        (8, 8, 1.0),  # Korvane §20 A vs B: decline rows changed nothing
        (33, 0, 2 * 0.5**33),  # Korvane A vs C3
        (5, 0, 0.0625),
        (6, 1, 0.125),
    ],
)
def test_exact_p_values(gained: int, lost: int, expected: float) -> None:
    assert mcnemar_exact_p(gained, lost) == pytest.approx(expected)
    assert mcnemar_exact_p(lost, gained) == pytest.approx(expected)  # two-sided


def test_compare_counts_only_shared_questions_and_names_the_verdict() -> None:
    a = {"q1": True, "q2": False, "q3": False, "q4": True, "only_a": True}
    b = {"q1": True, "q2": True, "q3": False, "q4": False, "only_b": False}
    r = compare_outcomes(a, b)
    assert (r.paired, r.both_pass, r.gained, r.lost, r.both_fail) == (4, 1, 1, 1, 1)
    assert r.verdict == "could be noise"
    many = {f"q{i}": False for i in range(40)}
    better = {f"q{i}": i < 10 for i in range(40)}
    assert compare_outcomes(many, better).verdict == "better"
    assert compare_outcomes(better, many).verdict == "worse"
    assert compare_outcomes({}, {"x": True}).verdict == "no shared questions"


def test_case_outcomes_skip_unscored_and_count_only_pass() -> None:
    cases = [
        {"case_name": "c1", "question": "Q1", "verdict": "pass"},
        {"case_name": "c1", "question": "Q1b", "verdict": "partial"},  # same name, other question: kept apart
        {"case_name": "c2", "question": "Q2", "verdict": ""},  # not judged yet
        {"case_name": "", "question": "Q3", "verdict": "fail"},
    ]
    out = case_outcomes(cases)
    assert sorted(out.values()) == [False, False, True]
    assert len(out) == 3


def _project(client) -> str:
    return client.post("/api/projects", json={"name": "Paired", "base_model": "/m"}).json()["id"]


def _cases(passes: list[bool]) -> list[dict]:
    return [{"name": f"case-{i}", "question": f"Question {i}?", "verdict": "pass" if p else "fail"}
            for i, p in enumerate(passes)]


def test_compare_api_reports_paired_test_per_suite(client) -> None:
    from finetune_studio import db

    pid = _project(client)
    run_a = db.create_run(project_id=pid, name="a", base_model="/m", data_path="/d")
    run_b = db.create_run(project_id=pid, name="b", base_model="/m", data_path="/d")
    a = [False] * 30
    b = [i < 12 for i in range(30)]
    db.create_benchmark(run_a["id"], "quiz", {"pass_rate": 0.0}, cases=_cases(a))
    db.create_benchmark(run_b["id"], "quiz", {"pass_rate": 40.0}, cases=_cases(b))
    r = client.get(f"/api/benchmarks/projects/{pid}/compare", params={"run_a": run_a["id"], "run_b": run_b["id"]})
    assert r.status_code == 200, r.text
    paired = r.json()["suites"][0]["paired"]
    assert paired["gained"] == 12 and paired["lost"] == 0 and paired["paired"] == 30
    assert paired["verdict"] == "better" and paired["p_value"] < 0.001


def test_compare_page_offers_benchmarked_base_model_as_baseline(client) -> None:
    from finetune_studio import db
    from finetune_studio.db.runs import BASE_PROBE_RUN_NAME

    pid = _project(client)
    done = db.create_run(project_id=pid, name="tuned", base_model="/m", data_path="/d")
    db.update_run(done["id"], status="done")
    base = db.create_run(pid, BASE_PROBE_RUN_NAME, base_model="/m")
    db.update_run(base["id"], status="done")
    page = client.get(f"/projects/{pid}/benchmarks").text
    assert f'<option value="{base["id"]}"' not in page  # never benchmarked: nothing to compare against
    db.create_benchmark(base["id"], "quiz", {"pass_rate": 10.0}, cases=_cases([True, False]))
    page = client.get(f"/projects/{pid}/benchmarks").text
    assert f'<option value="{base["id"]}" selected>base model (untrained)</option>' in page
