"""Test-run and judge jobs (webui/testing_jobs.py): the run records raw transcripts, judging is a separate step.

The inference engine and the judge are faked (see tests/testing_run_support.py); the job machinery, the DB rows and
the engine lock are the real ones. Pins: a run saves cases progressively and never a verdict; auto-judge is off by
default and chains a judge job only when asked (spec or saved setting); one GPU job at a time; cancel keeps what was
saved; every failure is written to the run row; a restart marks interrupted rows failed; an API judge needs no GPU.
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

from finetune_studio import db
from finetune_studio.db import benchmarks as benchmarks_db
from finetune_studio.db import judgements as jdb
from finetune_studio.testing.judge import JudgeUnavailable, default_judge_provider_id
from finetune_studio.webui import testing_jobs
from finetune_studio.webui.testing_jobs import (
    RunSpec,
    TestingBusy,
    start_judge_job,
    start_run_job,
)
from tests.test_rag_suite import FakeRag
from tests.testing_run_support import (
    FakeEngine,
    add_api_provider,
    await_finished,
    cases_n,
    install_engine,
    install_judge,
    make_project_run,
    saved_run,
)


@pytest.fixture(autouse=True)
def _isolated_jobs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """Fresh job registry + engine lock per test, a private settings file, and a model load that records itself."""
    monkeypatch.setattr(testing_jobs, "_ACTIVE", {})
    monkeypatch.setattr(testing_jobs, "ENGINE_LOCK", asyncio.Lock())
    monkeypatch.setattr("finetune_studio.webui.routes.settings.SETTINGS_PATH", tmp_path / "settings.json")
    loads: list[str] = []
    monkeypatch.setattr(testing_jobs, "_load_model", loads.append)
    return loads


def _spec(pid: str, n: int = 3, **kw: Any) -> RunSpec:
    return RunSpec(project_id=pid, kind="suite", suite_name="quiz.json", cases=cases_n(n), model_path="/m/merged",
                   max_tokens=16, **kw)


def _set_saved_auto_judge(value: bool) -> None:
    from finetune_studio.webui.routes import settings as settings_mod

    settings_mod._save({"test_auto_judge": value})


# ── a run records transcripts, never a verdict ───────────────────────────────


@pytest.mark.asyncio
async def test_run_persists_cases_progressively_and_ends_done_with_no_verdicts(
    monkeypatch: pytest.MonkeyPatch, _isolated_jobs: list[str],
) -> None:
    pid, _rid = make_project_run()
    seen: list[tuple[int, int]] = []

    def watch(_question: str) -> None:  # runs in the worker thread, right before each answer
        bid = testing_jobs.active_jobs()[0]["benchmark_id"]
        seen.append((len(db.list_cases(bid)), db.get_benchmark(bid)["progress_done"]))

    install_engine(monkeypatch, FakeEngine(on_generate=watch))
    asked = install_judge(monkeypatch)

    row = await start_run_job(_spec(pid, 3))
    assert row["status"] == "running" and row["progress_total"] == 3 and row["scoring"] == "judge"
    done = await await_finished(row["id"])

    assert seen == [(0, 0), (1, 1), (2, 2)]  # each case was on disk before the next question was asked
    assert done["status"] == "done" and done["error"] == "" and done["progress_done"] == 3
    assert done["model_path"] == "/m/merged" and _isolated_jobs == ["/m/merged"]
    cases = db.list_cases(row["id"])
    assert [c["question"] for c in cases] == ["Q0", "Q1", "Q2"]
    assert [c["model_answer"] for c in cases] == ["answer to Q0", "answer to Q1", "answer to Q2"]
    assert [c["correct_answer"] for c in cases] == ["key0", "key1", "key2"]
    assert all(c["verdict"] == "" and c["judge"] == "none" and c["transcript"] for c in cases)
    assert jdb.list_judgements(benchmark_id=row["id"]) == []
    assert done["judge_status"] == "" and asked == []  # off by default: nothing judged on its own
    scores = done["scores"]
    assert (scores["total"], scores["awaiting"], scores["pass_rate"]) == (3, 3, None)  # no verdict, no rate


@pytest.mark.asyncio
async def test_run_hangs_off_the_matching_training_run_or_the_hidden_evaluation_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    pid, rid = make_project_run()
    out = tmp_path / "out"
    db.update_run(rid, status="done", output_path=str(out))
    install_engine(monkeypatch)
    own = await start_run_job(_spec(pid, 1))  # model_path "/m/merged" is not under `out`
    await await_finished(own["id"])
    spec = _spec(pid, 1)
    spec.model_path = str(out / "merged")
    mine = await start_run_job(spec)
    await await_finished(mine["id"])
    assert mine["run_id"] == rid
    assert own["run_id"] != rid and db.get_run(own["run_id"])["name"] == "__evaluation__"


@pytest.mark.asyncio
async def test_a_failing_generation_is_recorded_on_the_case_and_the_run_still_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid, _rid = make_project_run()

    def answer(question: str) -> str:
        if question == "Q1":
            raise RuntimeError("cuda oom")
        return "fine"

    install_engine(monkeypatch, FakeEngine(answer))
    row = await start_run_job(_spec(pid, 3))
    done = await await_finished(row["id"])
    assert done["status"] == "done"
    cases = db.list_cases(row["id"])
    assert [bool(c["error"]) for c in cases] == [False, True, False] and "cuda oom" in cases[1]["error"]
    assert all(c["verdict"] == "" for c in cases)


@pytest.mark.asyncio
async def test_rag_run_saves_retrieval_provenance_next_to_each_case_and_no_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid, _rid = make_project_run()
    hit = {"rank": 1, "document_id": "doc-1", "source": "a.md", "filename": "a.md", "chunk_index": 0,
           "chunk_id": "c0", "score": 1.0, "text": "RK-04 is owned by Ada Smit."}
    monkeypatch.setattr("finetune_studio.testing.rag_suite.load_portable_rag_query",
                        lambda _p: FakeRag({"Who owns RK-04?": [hit]}))
    engine = install_engine(monkeypatch, FakeEngine(lambda _q: "Ada Smit", model_path="/m/merged"))
    from finetune_studio.testing.suite import BenchmarkCase

    spec = RunSpec(project_id=pid, kind="rag", suite_name="rag · s.json", model_path="/m/merged",
                   cases=[BenchmarkCase(name="rk04", question="Who owns RK-04?", correct_answer="Ada Smit",
                                        source_id="doc-1")],
                   rag={"top_k": 2, "corpus_path": "/corpus", "max_context_chars": 2000})
    row = await start_run_job(spec)
    done = await await_finished(row["id"])
    case = db.list_cases(row["id"])[0]
    assert done["status"] == "done" and case["verdict"] == "" and case["model_answer"] == "Ada Smit"
    assert case["judge_input"]["retrieval_hit"] is True and case["judge_input"]["context_text"]
    assert done["scores"]["retrieval"]["recall_at_k"] == 1.0 and done["scores"]["corpus_path"] == "/corpus"
    assert "Ada Smit" in engine.asked[0]  # the retrieved chunk is in the prompt the model saw


# ── auto-judge: off by default, explicit or from the saved setting ───────────


@pytest.mark.asyncio
async def test_auto_judge_in_the_spec_chains_a_judge_job_that_fills_the_verdicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid, _rid = make_project_run()
    provider = add_api_provider()
    install_engine(monkeypatch)
    asked = install_judge(monkeypatch, {"Q1": "fail"}, concurrent=True)

    row = await start_run_job(_spec(pid, 3, auto_judge=True, judge_provider_id=provider))
    done = await await_finished(row["id"], judge=True)

    assert done["status"] == "done" and done["judge_status"] == "done" and done["judge_error"] == ""
    assert done["judge_provider_id"] == provider and done["judge_model"] == "fake-judge"
    assert (done["judge_done"], done["judge_total"]) == (3, 3) and sorted(asked) == ["Q0", "Q1", "Q2"]
    verdicts = {c["question"]: (c["verdict"], c["judge"]) for c in db.list_cases(row["id"])}
    assert verdicts == {"Q0": ("pass", "ai"), "Q1": ("fail", "ai"), "Q2": ("pass", "ai")}
    assert (done["scores"]["passed"], done["scores"]["failed"], done["scores"]["awaiting"]) == (2, 1, 0)
    assert done["scores"]["by_judge"] == {"ai": 3}


@pytest.mark.asyncio
async def test_auto_judge_follows_the_saved_setting_and_an_explicit_false_beats_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid, _rid = make_project_run()
    install_engine(monkeypatch)
    asked = install_judge(monkeypatch)
    _set_saved_auto_judge(True)

    chained = await start_run_job(_spec(pid, 2))  # auto_judge unset -> the saved setting (on), default judge
    done = await await_finished(chained["id"], judge=True)
    assert done["judge_status"] == "done" and done["judge_provider_id"] == default_judge_provider_id()
    assert sorted(asked) == ["Q0", "Q1"]

    asked.clear()
    plain = await start_run_job(_spec(pid, 2, auto_judge=False))  # the run's own choice wins
    done = await await_finished(plain["id"])
    assert done["judge_status"] == "" and asked == []
    assert all(c["verdict"] == "" for c in db.list_cases(plain["id"]))


@pytest.mark.asyncio
async def test_auto_judge_is_off_when_the_setting_was_never_saved(monkeypatch: pytest.MonkeyPatch) -> None:
    pid, _rid = make_project_run()
    install_engine(monkeypatch)
    asked = install_judge(monkeypatch)
    row = await start_run_job(_spec(pid, 2))
    done = await await_finished(row["id"])
    await asyncio.sleep(0.1)  # a stray chained judge job would have started by now
    assert db.get_benchmark(row["id"])["judge_status"] == "" and asked == [] and done["status"] == "done"


@pytest.mark.asyncio
async def test_an_auto_judge_that_cannot_start_is_recorded_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    pid, _rid = make_project_run()
    install_engine(monkeypatch)
    install_judge(monkeypatch)
    row = await start_run_job(_spec(pid, 2, auto_judge=True, judge_provider_id="no-such-provider"))
    done = await await_finished(row["id"])
    assert done["status"] == "done" and done["progress_done"] == 2  # the run itself is complete and saved
    assert done["judge_status"] == "failed" and "no-such-provider" in done["judge_error"]


# ── one GPU job at a time ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_second_gpu_job_is_refused_while_one_is_active(monkeypatch: pytest.MonkeyPatch) -> None:
    pid, rid = make_project_run()
    gate = threading.Event()
    install_engine(monkeypatch, FakeEngine(on_generate=lambda _q: gate.wait(10)))
    install_judge(monkeypatch)
    finished = saved_run(rid, [("Q?", "k", "a")])
    try:
        first = await start_run_job(_spec(pid, 2))
        assert testing_jobs.is_active(first["id"], "run")

        with pytest.raises(TestingBusy) as busy:
            await start_run_job(_spec(pid, 1))
        assert busy.value.active == {"kind": "run", "benchmark_id": first["id"]}
        assert [b["id"] for b in db.list_benchmarks_for_project(pid)] == [first["id"], finished]  # no row for the refused run

        with pytest.raises(TestingBusy):  # a local-GGUF judge needs the same card
            await start_judge_job(finished, "local-default")
        assert db.get_benchmark(finished)["judge_status"] == ""
    finally:
        gate.set()
    await await_finished(first["id"])
    again = await start_run_job(_spec(pid, 1))  # the card is free again
    assert (await await_finished(again["id"]))["status"] == "done"


@pytest.mark.asyncio
async def test_a_test_never_takes_the_card_from_training_or_data_prep(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from finetune_studio.webui import app as app_mod

    pid, rid = make_project_run()
    install_engine(monkeypatch, FakeEngine())
    finished = saved_run(rid, [("Q?", "k", "a")])

    monkeypatch.setattr(app_mod.training_engine, "state", SimpleNamespace(status="training"))
    with pytest.raises(TestingBusy, match="training"):
        await start_run_job(_spec(pid, 1))
    with pytest.raises(TestingBusy, match="training"):  # a local judge would unload the trainer's model too
        await start_judge_job(finished, "local-default")
    assert [b["id"] for b in db.list_benchmarks_for_project(pid)] == [finished]  # no row for the refused run

    monkeypatch.setattr(app_mod.training_engine, "state", SimpleNamespace(status="idle"))
    monkeypatch.setattr(db, "list_stale_data_prep_runs", lambda: [{"id": "prep"}])
    with pytest.raises(TestingBusy, match="data-prep"):
        await start_run_job(_spec(pid, 1))

    monkeypatch.setattr(db, "list_stale_data_prep_runs", lambda: [])
    assert (await await_finished((await start_run_job(_spec(pid, 1)))["id"]))["status"] == "done"


@pytest.mark.asyncio
async def test_an_api_judge_runs_beside_a_gpu_run_but_not_twice_on_the_same_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid, rid = make_project_run()
    provider = add_api_provider()
    gate = threading.Event()
    install_engine(monkeypatch, FakeEngine(on_generate=lambda _q: gate.wait(10)))
    judge_gate = threading.Event()
    install_judge(monkeypatch, concurrent=True, on_chat=lambda _q: judge_gate.wait(10))
    finished = saved_run(rid, [("Q?", "k", "a")])
    try:
        run = await start_run_job(_spec(pid, 1))  # holds the GPU
        row = await start_judge_job(finished, provider)  # no GPU needed: accepted
        assert row["judge_status"] == "running" and row["judge_provider_id"] == provider
        with pytest.raises(TestingBusy):  # the same run is already being judged
            await start_judge_job(finished, provider)
    finally:
        judge_gate.set()
        gate.set()
    await await_finished(run["id"])
    assert (await await_finished(finished, judge=True))["judge_status"] == "done"


@pytest.mark.asyncio
async def test_the_engine_lock_is_released_after_a_run_even_when_it_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    pid, _rid = make_project_run()
    install_engine(monkeypatch)

    def boom(_path: str) -> None:
        raise RuntimeError("no such model")

    monkeypatch.setattr(testing_jobs, "_load_model", boom)
    row = await start_run_job(_spec(pid, 1))
    await await_finished(row["id"])
    assert not testing_jobs.ENGINE_LOCK.locked() and testing_jobs.active_jobs() == []


# ── cancel ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_stops_at_a_case_boundary_and_keeps_the_saved_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    pid, _rid = make_project_run()
    asked = install_judge(monkeypatch)

    def cancel_during_second_answer(question: str) -> None:
        if question == "Q1":
            assert testing_jobs.cancel(testing_jobs.active_jobs()[0]["benchmark_id"])

    install_engine(monkeypatch, FakeEngine(on_generate=cancel_during_second_answer))
    row = await start_run_job(_spec(pid, 5, auto_judge=True))
    done = await await_finished(row["id"])

    assert done["status"] == "cancelled" and done["error"] == "" and done["progress_done"] == 2
    cases = db.list_cases(row["id"])
    assert [c["question"] for c in cases] == ["Q0", "Q1"] and cases[1]["model_answer"] == "answer to Q1"
    assert done["scores"]["total"] == 2
    assert done["judge_status"] == "" and asked == []  # a cancelled run is not auto-judged
    assert not testing_jobs.cancel(row["id"])  # nothing left to cancel


@pytest.mark.asyncio
async def test_cancel_during_judging_keeps_the_verdicts_already_given(monkeypatch: pytest.MonkeyPatch) -> None:
    _pid, rid = make_project_run()
    bid = saved_run(rid, [(f"Q{i}", "k", "a") for i in range(4)])
    install_judge(monkeypatch, on_chat=lambda q: testing_jobs.cancel(bid) if q == "Q1" else None)
    await start_judge_job(bid, "local-default")
    done = await await_finished(bid, judge=True)
    assert done["judge_status"] == "cancelled" and done["judge_error"] == ""
    verdicts = [c["verdict"] for c in db.list_cases(bid)]
    assert verdicts == ["pass", "pass", "", ""] and done["scores"]["awaiting"] == 2


# ── failures are written to the run row ──────────────────────────────────────


@pytest.mark.asyncio
async def test_a_failing_model_load_marks_the_run_failed_with_the_error(monkeypatch: pytest.MonkeyPatch) -> None:
    pid, _rid = make_project_run()
    engine = install_engine(monkeypatch)
    asked = install_judge(monkeypatch)

    def boom(path: str) -> None:
        raise RuntimeError(f"cannot load {path}: CUDA out of memory")

    monkeypatch.setattr(testing_jobs, "_load_model", boom)
    row = await start_run_job(_spec(pid, 3, auto_judge=True))
    done = await await_finished(row["id"])
    assert done["status"] == "failed" and "CUDA out of memory" in done["error"] and "/m/merged" in done["error"]
    assert db.list_cases(row["id"]) == [] and engine.asked == []
    assert done["judge_status"] == "" and asked == []  # nothing to judge, nothing started
    assert testing_jobs.active_jobs() == []


@pytest.mark.asyncio
async def test_a_judge_that_returns_nothing_usable_fails_the_job_and_leaves_cases_unjudged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pid, rid = make_project_run()
    bid = saved_run(rid, [("Q1", "k", "a"), ("Q2", "k", "a")])
    install_judge(monkeypatch, reply=lambda _q: "looks good to me")
    await start_judge_job(bid, "local-default")
    done = await await_finished(bid, judge=True)
    assert done["judge_status"] == "failed" and "no usable verdict" in done["judge_error"]
    assert all(c["verdict"] == "" for c in db.list_cases(bid)) and done["scores"]["pass_rate"] is None
    assert all(j["error"] for j in jdb.list_judgements(benchmark_id=bid))  # each failure says why


@pytest.mark.asyncio
async def test_a_judge_that_cannot_be_opened_fails_the_job(monkeypatch: pytest.MonkeyPatch) -> None:
    _pid, rid = make_project_run()
    bid = saved_run(rid, [("Q1", "k", "a")])

    def refuse(provider_id: str) -> Any:
        raise JudgeUnavailable(f"could not load judge '{provider_id}': file missing")

    monkeypatch.setattr(testing_jobs, "open_judge", refuse)
    await start_judge_job(bid, "local-default")
    done = await await_finished(bid, judge=True)
    assert done["judge_status"] == "failed" and "file missing" in done["judge_error"]
    assert not testing_jobs.ENGINE_LOCK.locked()


def test_reconcile_stale_benchmarks_marks_running_rows_failed_and_keeps_their_cases() -> None:
    _pid, rid = make_project_run()
    running = saved_run(rid, [("Q1", "k", "a")], status="running")
    queued = saved_run(rid, [], status="queued")
    judging = saved_run(rid, [("Q1", "k", "a")], status="done")
    db.update_benchmark(judging, judge_status="running", judge_total=1)
    finished = saved_run(rid, [("Q1", "k", "a")], status="done")
    db.update_benchmark(finished, judge_status="done")

    assert {b["id"] for b in benchmarks_db.list_unfinished()} == {running, queued, judging}
    assert db.reconcile_stale_benchmarks() == 3

    for bid in (running, queued):
        row = db.get_benchmark(bid)
        assert row["status"] == "failed" and "interrupted" in row["error"]
    assert len(db.list_cases(running)) == 1  # what was saved before the crash is kept
    row = db.get_benchmark(judging)
    assert row["status"] == "done" and row["judge_status"] == "failed" and "interrupted" in row["judge_error"]
    row = db.get_benchmark(finished)
    assert (row["status"], row["judge_status"], row["error"]) == ("done", "done", "")
    assert benchmarks_db.list_unfinished() == [] and db.reconcile_stale_benchmarks() == 0


# ── the engine lock: local judge waits for it, API judge does not ────────────


@pytest.mark.asyncio
async def test_an_api_judge_does_not_take_the_engine_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    _pid, rid = make_project_run()
    provider = add_api_provider()
    bid = saved_run(rid, [("Q1", "k", "a"), ("Q2", "k", "a")])
    install_judge(monkeypatch, concurrent=True)
    async with testing_jobs.ENGINE_LOCK:  # the GPU is busy with something else
        await start_judge_job(bid, provider)
        done = await await_finished(bid, judge=True, timeout=5)
        assert testing_jobs.ENGINE_LOCK.locked()
    assert done["judge_status"] == "done" and done["scores"]["passed"] == 2


@pytest.mark.asyncio
async def test_a_local_judge_waits_for_the_engine_lock_then_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    _pid, rid = make_project_run()
    bid = saved_run(rid, [("Q1", "k", "a")])
    asked = install_judge(monkeypatch)
    async with testing_jobs.ENGINE_LOCK:
        await start_judge_job(bid, "local-default")
        await asyncio.sleep(0.3)
        row = db.get_benchmark(bid)
        assert row["judge_status"] == "running" and row["judge_done"] == 0 and asked == []
    done = await await_finished(bid, judge=True)
    assert done["judge_status"] == "done" and asked == ["Q1"]


# ── what may be judged ───────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["running", "queued"])
async def test_a_judge_job_for_a_run_that_is_still_running_is_rejected(status: str) -> None:
    _pid, rid = make_project_run()
    bid = saved_run(rid, [("Q1", "k", "a")], status=status)
    with pytest.raises(ValueError, match="still running"):
        await start_judge_job(bid, "local-default")
    row = db.get_benchmark(bid)
    assert row["judge_status"] == "" and not testing_jobs.is_active(bid)


@pytest.mark.asyncio
async def test_an_exact_scored_run_is_rejected_by_the_judge_job() -> None:
    _pid, rid = make_project_run()
    bid = saved_run(rid, [("Q1", "k", "a")], scoring="exact")
    with pytest.raises(ValueError, match="exact-match"):
        await start_judge_job(bid, "local-default")
    assert db.get_benchmark(bid)["judge_status"] == "" and jdb.list_judgements(benchmark_id=bid) == []


@pytest.mark.asyncio
async def test_judge_job_rejects_unknown_run_empty_run_and_unknown_provider() -> None:
    _pid, rid = make_project_run()
    with pytest.raises(LookupError):
        await start_judge_job("nope", "local-default")
    empty = saved_run(rid, [])
    with pytest.raises(ValueError, match="no saved cases"):
        await start_judge_job(empty, "local-default")
    bid = saved_run(rid, [("Q1", "k", "a")])
    with pytest.raises(JudgeUnavailable, match="unknown provider"):
        await start_judge_job(bid, "no-such-provider")
    assert db.get_benchmark(bid)["judge_status"] == "" and testing_jobs.active_jobs() == []


@pytest.mark.asyncio
async def test_judge_job_uses_the_configured_default_judge_when_no_provider_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from finetune_studio.webui.routes import settings as settings_mod

    _pid, rid = make_project_run()
    provider = add_api_provider()
    settings_mod._save({"test_judge_provider_id": provider})
    bid = saved_run(rid, [("Q1", "k", "a")])
    install_judge(monkeypatch, concurrent=True)
    await start_judge_job(bid)
    assert (await await_finished(bid, judge=True))["judge_provider_id"] == provider


@pytest.mark.asyncio
async def test_only_unjudged_leaves_a_human_verdict_alone_and_rejudging_never_overrides_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pid, rid = make_project_run()
    bid = saved_run(rid, [("Q1", "k", "a"), ("Q2", "k", "a")])
    first, _second = [c["id"] for c in db.list_cases(bid)]
    jdb.add_judgement(first, kind="human", verdict="fail", reasoning="mine")
    asked = install_judge(monkeypatch)

    await start_judge_job(bid, "local-default", only_unjudged=True)
    await await_finished(bid, judge=True)
    assert asked == ["Q2"]

    await start_judge_job(bid, "local-default", only_unjudged=False)
    done = await await_finished(bid, judge=True)
    assert asked == ["Q2", "Q1", "Q2"]
    cases = {c["id"]: c for c in db.list_cases(bid)}
    assert cases[first]["verdict"] == "fail" and cases[first]["judge"] == "human"  # the person's call stands
    assert len(jdb.list_judgements(case_id=first)) == 2  # the AI opinion is kept for comparison
    assert done["scores"]["by_judge"] == {"human": 1, "ai": 1}


@pytest.mark.asyncio
async def test_judge_progress_is_persisted_on_the_run_row(monkeypatch: pytest.MonkeyPatch) -> None:
    _pid, rid = make_project_run()
    bid = saved_run(rid, [(f"Q{i}", "k", "a") for i in range(3)])
    seen: list[tuple[str, int, int]] = []
    install_judge(monkeypatch, on_chat=lambda _q: seen.append(
        (db.get_benchmark(bid)["judge_status"], db.get_benchmark(bid)["judge_total"], db.get_benchmark(bid)["judge_done"])))
    await start_judge_job(bid, "local-default")
    done = await await_finished(bid, judge=True)
    assert seen[0] == ("running", 3, 0) and all(s[:2] == ("running", 3) for s in seen)
    assert (done["judge_done"], done["judge_total"], done["judge_model"]) == (3, 3, "fake-judge")
    assert jdb.list_judgements(benchmark_id=bid)[0]["reasoning"] == "judged pass"
