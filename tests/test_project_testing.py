"""Tests for project testing page suite dropdown."""
from __future__ import annotations

import time


def _project(client) -> str:
    r = client.post("/api/projects", json={"name": "Testing Suite Dropdown"})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def test_testing_page_renders_suite_select(client) -> None:
    pid = _project(client)
    r = client.get(f"/projects/{pid}/testing")
    assert r.status_code == 200
    body = r.text
    assert 'id="t-suite"' in body
    assert "<select id=\"t-suite\"" in body or "<select id='t-suite'" in body
    assert "— pick a suite —" in body
    # The user's own quizzes are tested here (saved answers, judged afterwards): a local known suite is offered...
    assert 'value="data/benchmarks/default.json"' in body
    # ...public / built-in multiple-choice suites keep their official exact-match scoring on the Benchmarks page
    assert 'value="real://gsm8k"' not in body
    assert "MMLU-shaped" not in body and "GSM8K-shaped" not in body and "HellaSwag-shaped" not in body
    # Free-text path input must be gone
    assert 'placeholder="path/to/suite.json"' not in body
    # Two separate steps are visible: run, then judge
    assert "1 · Run a test" in body and "2 · Judge" in body
    assert 'id="t-auto-judge"' in body and 'id="t-judge-provider"' in body
    assert 'id="t-status"' in body
    # Model selector is project-scoped (not global HF discovery alone)
    assert 'id="t-model"' in body
    assert "— latest merged export —" in body
    # Readable labels, not raw suite JSON blobs in the select
    assert "<pre>" not in body.split('id="t-suite"')[1].split("</select>")[0]


def test_testing_page_lists_project_merged_export(client, tmp_path) -> None:
    """After merge-at-export, the merged path must appear in the selector."""
    from finetune_studio import db

    pid = _project(client)
    out = tmp_path / "run-out"
    merged = out / "merged"
    merged.mkdir(parents=True)
    (merged / "model.safetensors").write_bytes(b"x" * 64)
    (merged / "config.json").write_text("{}", encoding="utf-8")
    run = db.create_run(
        project_id=pid, name="merged-run", base_model="/m", data_path="/d"
    )
    db.update_run(run["id"], status="done", output_path=str(out))

    r = client.get(f"/projects/{pid}/testing")
    assert r.status_code == 200
    body = r.text
    assert str(merged) in body
    assert "merged-run/merged" in body
    assert "selected" in body  # default_model_path pre-selects latest merged
    assert "No project merged/GGUF exports yet" not in body


def test_testing_page_suite_options_are_the_judged_suites(client) -> None:
    from finetune_studio.webui.routes.benchmarks import _discover_suites

    pid = _project(client)
    suites = _discover_suites(pid)
    r = client.get(f"/projects/{pid}/testing")
    assert r.status_code == 200
    assert any(s["scoring"] == "judge" for s in suites) and any(s["scoring"] == "exact" for s in suites)
    for s in suites:
        present = f'value="{s["path"]}"' in r.text
        assert present == (s["scoring"] == "judge"), s["path"]


def test_testing_page_lists_runs_from_the_api_not_a_server_snapshot(client) -> None:
    pid = _project(client)
    r = client.get(f"/projects/{pid}/testing")
    assert 'id="t-runs"' in r.text and "/api/testing/projects/" in r.text


def test_benchmarks_for_project_lists_runs_newest_first_with_run_name(temp_db) -> None:
    from finetune_studio import db
    from finetune_studio.db.connection import cursor

    pid = db.create_project(name="Recent Suite Runs")["id"]
    run = db.create_run(project_id=pid, name="r1", base_model="/m", data_path="/d")
    base = time.time()
    for i in range(3):
        b = db.create_benchmark(run["id"], f"suite_{i}", {"pass_rate": float(i)}, time_ms=i)
        with cursor() as c:
            c.execute("UPDATE benchmark_runs SET ran_at = ? WHERE id = ?", (base + i, b["id"]))
    rows = db.list_benchmarks_for_project(pid, limit=2)
    assert [r["suite_name"] for r in rows] == ["suite_2", "suite_1"]
    assert rows[0]["run_name"] == "r1" and rows[0]["config"] == {} and rows[0]["status"] == "done"
