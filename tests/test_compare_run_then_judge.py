"""Compare tab = run, then judge (Genor 2026-10-09): several models answer the same quiz, nobody scores by matching.

Pins: a compare run saves each model's raw answers (one test run per model, kind ``compare``) with NO verdict and no
score; the AI judge (the Testing page's, loaded once for the group) or a person decides afterwards; the side-by-side
view lines the answers up next to the answer key; one failing model does not stop the others; stopping any row stops
the group; the old keyword scorer (``benchmarks.comparison`` / ``score_open_ended``) is gone. The engine and the judge
are faked (tests/testing_run_support.py); the jobs, the DB rows and the routes are the real ones.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db
from finetune_studio.compare import session
from finetune_studio.db import judgements as jdb
from finetune_studio.webui import testing_jobs
from tests.test_cli_error_paths import run_cli
from tests.testing_run_support import (
    FakeEngine,
    install_engine,
    install_judge,
    wait_until,
)

API = "/api/compare/projects"
MODEL_A, MODEL_B = "/m/a/merged", "/m/b/merged"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, object]:
    """Fresh job registry + engine lock, a private settings file, and a model load that records which model is resident."""
    monkeypatch.setattr(testing_jobs, "_ACTIVE", {})
    monkeypatch.setattr(testing_jobs, "ENGINE_LOCK", asyncio.Lock())
    monkeypatch.setattr("finetune_studio.webui.routes.settings.SETTINGS_PATH", tmp_path / "settings.json")
    import finetune_studio.webui.app  # noqa: F401  (loads the routes in app order; a bare string patch would hit a circular import)

    monkeypatch.setattr("finetune_studio.webui.routes.testing.local_model_missing", lambda _p: False)  # fake model paths below
    state: dict[str, object] = {"model": "", "loads": [], "fail": set()}

    def load(path: str) -> None:
        if path in state["fail"]:  # type: ignore[operator]
            raise RuntimeError(f"cannot load {path}")
        state["model"] = path
        state["loads"].append(path)  # type: ignore[attr-defined]

    monkeypatch.setattr(testing_jobs, "_load_model", load)
    return state


def _engine(monkeypatch: pytest.MonkeyPatch, state: dict[str, object], **kw) -> FakeEngine:
    """An engine whose answers name the resident model, so each model's answers are tell-apart."""
    return install_engine(monkeypatch, FakeEngine(lambda q: f"{Path(str(state['model'])).parent.name}:{q}", **kw))


def _project(client: TestClient) -> str:
    r = client.post("/api/projects", json={"name": "cmp", "base_model": "x/test"})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _suite(tmp_path: Path, n: int = 3) -> str:
    path = tmp_path / "quiz.json"
    path.write_text(json.dumps([
        {"name": f"c{i}", "category": "geo", "question": f"Q{i}", "correct_answer": f"key{i}", "keywords": [f"key{i}"]}
        for i in range(n)
    ]), encoding="utf-8")
    return str(path)


def _start(client: TestClient, pid: str, suite: str, models=(MODEL_A, MODEL_B), **extra) -> dict:
    r = client.post(f"{API}/{pid}/runs", json={"suite_path": suite, "models": list(models), "max_tokens": 16, **extra})
    assert r.status_code == 202, r.text
    return r.json()


def _finished(client: TestClient, pid: str, gid: str, *, judge: bool = False) -> dict:
    def done() -> bool:
        view = client.get(f"{API}/{pid}/groups/{gid}").json()
        ok = view["status"]["run"] != "running" and not testing_jobs.active_jobs()
        return ok and (not judge or view["status"]["judge"] not in ("", "running"))

    wait_until(done, what="the comparison to finish")
    return client.get(f"{API}/{pid}/groups/{gid}").json()


# ── run: raw answers per model, no verdict ───────────────────────────────────


def test_run_saves_every_models_answers_side_by_side_and_scores_nothing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _isolated: dict[str, object],
) -> None:
    pid = _project(client)
    engine = _engine(monkeypatch, _isolated)
    started = _start(client, pid, _suite(tmp_path))
    rows = started["benchmarks"]
    assert [r["kind"] for r in rows] == ["compare", "compare"] and {r["model_path"] for r in rows} == {MODEL_A, MODEL_B}
    assert all(r["config"]["compare"]["group_id"] == started["group_id"] for r in rows)

    view = _finished(client, pid, started["group_id"])
    assert view["status"] == {"run": "done", "judge": ""}
    assert _isolated["loads"] == [MODEL_A, MODEL_B]  # one model resident at a time, in the requested order
    assert engine.asked == ["Q0", "Q1", "Q2", "Q0", "Q1", "Q2"]  # model A finished before B was loaded
    a, b = (m["benchmark_id"] for m in view["models"])
    assert [c["question"] for c in view["cases"]] == ["Q0", "Q1", "Q2"]
    for i, c in enumerate(view["cases"]):
        assert c["correct_answer"] == f"key{i}" and set(c["answers"]) == {a, b}
        assert c["answers"][a]["model_answer"] == f"a:Q{i}" and c["answers"][b]["model_answer"] == f"b:Q{i}"
        assert c["answers"][a]["verdict"] == "" and c["answers"][b]["judgements"] == []  # nothing decided anything
        assert c["disagree"] is False
    for m in view["models"]:
        assert (m["scores"]["total"], m["scores"]["awaiting"], m["scores"]["pass_rate"]) == (3, 3, None)
        assert m["status"] == "done" and m["progress_done"] == 3
    assert jdb.list_judgements(benchmark_id=a) == [] and jdb.list_judgements(benchmark_id=b) == []


def test_an_answer_that_contains_the_key_is_not_a_pass_until_a_judge_or_person_says_so(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _isolated: dict[str, object],
) -> None:
    """The old tab passed any answer holding the keyword. Now the answer says 'key0' verbatim and is still awaiting."""
    pid = _project(client)
    install_engine(monkeypatch, FakeEngine(lambda q: f"it is not key{q[1:]}, I refuse"))
    install_judge(monkeypatch, {"Q0": "fail"})
    gid = _start(client, pid, _suite(tmp_path, 1))["group_id"]
    view = _finished(client, pid, gid)
    assert all(a["verdict"] == "" for a in view["cases"][0]["answers"].values())
    assert client.post(f"{API}/{pid}/groups/{gid}/judge", json={"provider_id": _api_provider()}).status_code == 202
    view = _finished(client, pid, gid, judge=True)
    assert {a["verdict"] for a in view["cases"][0]["answers"].values()} == {"fail"}  # the judge decided, not the matcher


# ── judge: the Testing page's judge, loaded once for the group ───────────────


def _api_provider() -> str:
    from tests.testing_run_support import add_api_provider

    return add_api_provider()


def test_group_judge_records_ai_verdicts_for_every_row_and_a_person_overrides(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _isolated: dict[str, object],
) -> None:
    pid = _project(client)
    _engine(monkeypatch, _isolated)
    opened: list[str] = []
    asked = install_judge(monkeypatch, {"Q1": "fail", "Q2": "partial"})
    import finetune_studio.webui.testing_jobs as tj

    real_open = tj.open_judge

    def counting_open(provider_id: str):
        opened.append(provider_id)
        return real_open(provider_id)

    monkeypatch.setattr(tj, "open_judge", counting_open)
    provider = _api_provider()
    gid = _start(client, pid, _suite(tmp_path))["group_id"]
    _finished(client, pid, gid)

    r = client.post(f"{API}/{pid}/groups/{gid}/judge", json={"provider_id": provider})
    assert r.status_code == 202 and len(r.json()["benchmarks"]) == 2
    view = _finished(client, pid, gid, judge=True)

    assert opened == [provider]  # ONE judge load for both models
    assert sorted(asked) == ["Q0", "Q0", "Q1", "Q1", "Q2", "Q2"]
    assert view["status"]["judge"] == "done"
    a, b = (m["benchmark_id"] for m in view["models"])
    for m in view["models"]:  # pass, fail, partial of 3 → 33.3 % pass
        assert m["scores"]["judged"] == 3 and m["scores"]["pass_rate"] == 33.3 and m["judge_model"] == "fake-judge"
    assert [c["answers"][a]["verdict"] for c in view["cases"]] == ["pass", "fail", "partial"]

    # a person disagrees on model B's first answer: the Testing page's verdict call, then the view shows who decided
    cid = view["cases"][0]["answers"][b]["case_id"]
    r = client.put(f"/api/testing/projects/{pid}/runs/{b}/cases/{cid}/verdict", json={"verdict": "fail", "reasoning": "wrong city"})
    assert r.status_code == 200, r.text
    view = client.get(f"{API}/{pid}/groups/{gid}").json()
    first = view["cases"][0]
    assert first["answers"][b]["verdict"] == "fail" and first["answers"][b]["judge"] == "human"
    assert first["answers"][a]["verdict"] == "pass" and first["disagree"] is True
    assert [j["kind"] for j in first["answers"][b]["judgements"]] == ["ai", "human"]  # the AI's opinion is kept
    assert view["models"][1]["scores"]["pass_rate"] == 0.0 and view["models"][0]["scores"]["pass_rate"] == 33.3
    assert view["agreement"][b]["judges"]["fake-judge"] == {"compared": 1, "agree": 0, "agreement": 0.0}


def test_auto_judge_option_judges_when_the_run_finishes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _isolated: dict[str, object],
) -> None:
    pid = _project(client)
    _engine(monkeypatch, _isolated)
    install_judge(monkeypatch)
    provider = _api_provider()
    gid = _start(client, pid, _suite(tmp_path, 2), auto_judge=True, judge_provider_id=provider)["group_id"]
    wait_until(lambda: client.get(f"{API}/{pid}/groups/{gid}").json()["status"]["judge"] == "done", what="auto-judge")
    view = client.get(f"{API}/{pid}/groups/{gid}").json()
    assert all(m["scores"]["pass_rate"] == 100.0 for m in view["models"])


def test_judging_is_refused_while_the_models_are_still_answering_and_for_unknown_groups(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _isolated: dict[str, object],
) -> None:
    pid = _project(client)
    gate_open = []
    _engine(monkeypatch, _isolated, on_generate=lambda _q: wait_until(lambda: bool(gate_open), what="the test to release"))
    install_judge(monkeypatch)
    gid = _start(client, pid, _suite(tmp_path, 1))["group_id"]
    r = client.post(f"{API}/{pid}/groups/{gid}/judge", json={"provider_id": _api_provider()})
    assert r.status_code == 400 and "still answering" in r.json()["error"]
    assert client.post(f"{API}/{pid}/groups/nope/judge", json={}).status_code == 404
    gate_open.append(1)
    _finished(client, pid, gid)


# ── failure, cancel, busy ────────────────────────────────────────────────────


def test_one_model_that_cannot_load_fails_alone_and_the_others_still_answer(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _isolated: dict[str, object],
) -> None:
    pid = _project(client)
    _engine(monkeypatch, _isolated)
    _isolated["fail"] = {MODEL_A}
    gid = _start(client, pid, _suite(tmp_path, 2))["group_id"]
    view = _finished(client, pid, gid)
    a, b = view["models"]
    assert a["status"] == "failed" and "cannot load" in a["error"] and a["progress_done"] == 0
    assert b["status"] == "done" and b["progress_done"] == 2
    assert view["status"]["run"] == "failed"
    assert [c["answers"].get(a["benchmark_id"]) for c in view["cases"]] == [None, None]  # no answer, not a fake one


def test_stopping_any_row_stops_the_whole_group_and_keeps_what_was_saved(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _isolated: dict[str, object],
) -> None:
    pid = _project(client)
    stopped = []

    def stop_after_first(_q: str) -> None:
        if not stopped:
            stopped.append(1)
            gid = session.list_groups(pid)[0]["group_id"]
            assert testing_jobs.cancel(session.group_rows(pid, gid)[1]["id"])  # the row that is only queued

    _engine(monkeypatch, _isolated, on_generate=stop_after_first)
    gid = _start(client, pid, _suite(tmp_path, 3))["group_id"]
    view = _finished(client, pid, gid)
    a, b = view["models"]
    assert (a["status"], a["progress_done"]) == ("cancelled", 1)  # finished the question it was on, then stopped
    assert (b["status"], b["progress_done"]) == ("cancelled", 0)
    assert view["status"]["run"] == "cancelled" and _isolated["loads"] == [MODEL_A]


def test_group_cancel_route_and_busy_gpu(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _isolated: dict[str, object],
) -> None:
    pid = _project(client)
    _engine(monkeypatch, _isolated)
    done = _start(client, pid, _suite(tmp_path, 1))["group_id"]
    _finished(client, pid, done)
    assert client.post(f"{API}/{pid}/groups/{done}/cancel").status_code == 409  # nothing running
    assert client.post(f"{API}/{pid}/groups/nope/cancel").status_code == 404

    testing_jobs._ACTIVE[("other", "run")] = testing_jobs._Job("other", "run", True)
    r = client.post(f"{API}/{pid}/runs", json={"suite_path": _suite(tmp_path), "models": [MODEL_A, MODEL_B]})
    assert r.status_code == 409 and r.json()["active"]["benchmark_id"] == "other"
    assert len(session.list_groups(pid)) == 1  # the refused request left no rows behind


# ── input validation ─────────────────────────────────────────────────────────


def test_start_validates_the_request(client: TestClient, tmp_path: Path) -> None:
    pid = _project(client)
    suite = _suite(tmp_path)
    base = f"{API}/{pid}/runs"
    h = {"content-type": "application/json"}
    assert client.post(base, content="x", headers=h).status_code == 400
    assert client.post(base, json=[1]).status_code == 400
    assert client.post(base, json={"models": [MODEL_A, MODEL_B]}).status_code == 400            # no suite
    assert client.post(base, json={"suite_path": suite, "models": [MODEL_A]}).status_code == 400  # one model is a test, not a compare
    assert client.post(base, json={"suite_path": suite, "models": [MODEL_A, MODEL_A]}).status_code == 400
    assert client.post(base, json={"suite_path": suite, "models": [MODEL_A, ""]}).status_code == 400
    assert client.post(base, json={"suite_path": suite, "models": [MODEL_A] * 9}).status_code == 400
    assert client.post(base, json={"suite_path": suite, "models": [MODEL_A, MODEL_B], "max_tokens": "x"}).status_code == 400
    assert client.post(base, json={"suite_path": suite, "models": [MODEL_A, MODEL_B], "auto_judge": "yes"}).status_code == 400
    assert client.post(base, json={"suite_path": str(tmp_path / "ghost.json"), "models": [MODEL_A, MODEL_B]}).status_code == 404
    (tmp_path / "empty.json").write_text("[]")
    assert client.post(base, json={"suite_path": str(tmp_path / "empty.json"), "models": [MODEL_A, MODEL_B]}).status_code == 400
    assert client.post(f"{API}/nope/runs", json={}).status_code == 404
    assert client.get(f"{API}/nope/groups").status_code == 404
    assert client.get(f"{API}/{pid}/groups/nope").status_code == 404
    assert client.get(f"{API}/{pid}/groups").json() == []
    assert not testing_jobs.active_jobs()


def test_the_untrained_base_can_be_one_of_the_models(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _isolated: dict[str, object],
) -> None:
    pid = _project(client)
    _engine(monkeypatch, _isolated)
    started = _start(client, pid, _suite(tmp_path, 1), models=["__base__", MODEL_A])
    view = _finished(client, pid, started["group_id"])
    assert view["models"][0]["model_path"] == "x/test" and view["models"][0]["label"].startswith("Untrained base")


# ── the view itself ──────────────────────────────────────────────────────────


def _saved_compare_row(rid: str, gid: str, idx: int, label: str, answers: list[tuple[str, str, str]]) -> str:
    bench = db.create_benchmark(rid, "quiz.json", {}, status="done", kind="compare",
                                config={"compare": {"group_id": gid, "index": idx, "label": label}})
    for name, q, ans in answers:
        db.create_case(bench["id"], rid, name, "general", q, f"key of {q}", ans, [])
    return bench["id"]


def test_side_by_side_lines_up_cases_by_name_even_with_repeats_and_missing_answers() -> None:
    pid = db.create_project(name=f"p-{time.time_ns()}")["id"]
    rid = db.create_run(pid, "train")["id"]
    a = _saved_compare_row(rid, "g1", 0, "A", [("dup", "Q1", "a1"), ("dup", "Q2", "a2"), ("tail", "Q3", "a3")])
    b = _saved_compare_row(rid, "g1", 1, "B", [("dup", "Q1", "b1"), ("dup", "Q2", "b2")])  # stopped before "tail"
    _saved_compare_row(rid, "g2", 0, "other group", [("dup", "Q1", "z")])
    view = session.side_by_side(pid, "g1")
    assert view is not None and [m["label"] for m in view["models"]] == ["A", "B"]
    assert [(c["case_name"], c["question"]) for c in view["cases"]] == [("dup", "Q1"), ("dup", "Q2"), ("tail", "Q3")]
    assert view["cases"][1]["answers"][a]["model_answer"] == "a2" and view["cases"][1]["answers"][b]["model_answer"] == "b2"
    assert b not in view["cases"][2]["answers"]
    assert session.side_by_side(pid, "nope") is None
    assert {g["group_id"] for g in session.list_groups(pid)} == {"g1", "g2"}


# ── the old keyword scorer is gone ───────────────────────────────────────────


def test_no_keyword_scorer_is_left_on_the_compare_path() -> None:
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("finetune_studio.benchmarks.comparison")
    from finetune_studio.benchmarks import scoring

    assert not hasattr(scoring.scorer, "score_open_ended")
    src = (Path(session.__file__).parent / "session.py").read_text() + Path(session.__file__).with_name("__init__.py").read_text()
    assert "keyword" not in src.lower().replace("keywords", "")  # the view carries no matcher


def test_the_runs_comparison_never_shows_a_case_count_as_a_score() -> None:
    from finetune_studio.webui.routes.benchmarks import _primary_score

    unjudged = {"total": 5, "judged": 0, "awaiting": 5, "passed": 0, "pass_rate": None, "avg_time_ms": 12.0}
    assert _primary_score(unjudged) is None
    assert _primary_score({**unjudged, "judged": 5, "pass_rate": 60.0}) == 60.0
    assert _primary_score({"score": 0.5}) == 0.5   # an older run that stored a score keeps working
    assert _primary_score({"total": 5}) is None


# ── page + CLI ───────────────────────────────────────────────────────────────


def test_compare_page_renders_and_is_in_the_project_nav(client: TestClient, tmp_path: Path) -> None:
    pid = _project(client)
    r = client.get(f"/projects/{pid}/compare")
    assert r.status_code == 200
    for needle in ('id="cp-run-btn"', 'id="cp-suite"', 'id="cp-models"', 'id="cp-judge-btn"', 'id="cp-cases"', "Nothing is scored by string matching"):
        assert needle in r.text, needle
    assert f'href="/projects/{pid}/testing"' in r.text and 'data-tab="compare"' in r.text
    assert client.get("/projects/nope/compare").status_code == 404


class _CliEngine:
    """Answers with the model it was loaded from (CLI path: one engine per model, unloaded after)."""

    path = ""

    def load(self, path: str, **_kw) -> None:
        self.path = path

    def generate(self, messages, **_kw) -> str:
        return f"{Path(self.path).name} says Paris"

    def unload(self) -> None:
        pass


def _cli_setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[str, str, str]:
    monkeypatch.setattr("finetune_studio.testing.inference.InferenceEngine", _CliEngine)
    suite = tmp_path / "s.json"
    suite.write_text('[{"name":"c","question":"Capital of France?","correct_answer":"Paris","keywords":["Paris"]}]')
    a, b = tmp_path / "ma", tmp_path / "mb"
    a.mkdir()
    b.mkdir()
    return str(suite), str(a), str(b)


def test_cli_compare_records_answers_and_judges_nothing_by_itself(monkeypatch, capsys, tmp_path) -> None:
    suite, a, b = _cli_setup(monkeypatch, tmp_path)
    report = tmp_path / "r.json"
    code, out = run_cli(monkeypatch, capsys, "compare", suite, "--models", f"a={a}", f"b={b}", "--report", str(report))
    assert code == 0, out
    assert "[unjudged] a" in out and "[unjudged] b" in out and "none judged" in out
    assert "accuracy" not in out and "% (" not in out
    data = json.loads(report.read_text())
    assert data["judge"] is None and data["models"] == ["a", "b"]
    case = data["cases"][0]
    assert case["correct_answer"] == "Paris" and case["answers"]["a"]["response"] == "ma says Paris"
    assert case["answers"]["b"]["judge"] is None


def test_cli_compare_judge_flag_uses_the_judge_provider_once(monkeypatch, capsys, tmp_path) -> None:
    from contextlib import contextmanager

    from finetune_studio.testing.judge import LoadedJudge

    suite, a, b = _cli_setup(monkeypatch, tmp_path)
    opened: list[str] = []
    asked: list[str] = []

    @contextmanager
    def fake_open(provider_id: str):
        opened.append(provider_id)
        yield LoadedJudge(chat=lambda m: (asked.append(m[-1]["content"]), '{"verdict": "pass", "reasoning": "ok"}')[1],
                          provider_id=provider_id, model="fake-judge", label="Fake", concurrent=False)

    monkeypatch.setattr("finetune_studio.testing.judge.open_judge", fake_open)
    monkeypatch.setattr("finetune_studio.testing.judge.default_judge_provider_id", lambda: "default-judge")
    code, out = run_cli(monkeypatch, capsys, "compare", suite, "--models", f"a={a}", f"b={b}", "--judge")
    assert code == 0, out
    assert opened == ["default-judge"] and len(asked) == 2
    assert "a: 1 pass, 0 partial, 0 fail, 0 not judged" in out and "b: 1 pass" in out


def test_cli_compare_rejects_repeated_names_and_missing_models(monkeypatch, capsys, tmp_path) -> None:
    suite, a, _b = _cli_setup(monkeypatch, tmp_path)
    code, out = run_cli(monkeypatch, capsys, "compare", suite, "--models", f"x={a}", f"x={a}")
    assert code == 1 and "used twice" in out and "Traceback" not in out
    code, out = run_cli(monkeypatch, capsys, "compare", suite, "--models", f"x={tmp_path / 'ghost'}")
    assert code == 1 and "Model not found" in out
