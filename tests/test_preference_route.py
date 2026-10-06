"""POST /api/projects/{pid}/data-prep/preference: statuses, shape, progress, and an unblocked event loop."""
from __future__ import annotations

import asyncio
import threading
import time

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from finetune_studio.db.datasets import list_datasets
from finetune_studio.webui.routes import data_prep_preference as route
from tests.preference_fakes import FakeModel, seed_project


@pytest.fixture
def api() -> TestClient:
    app = FastAPI()
    app.include_router(route.router, prefix="/api")
    return TestClient(app)


def _use_model(monkeypatch: pytest.MonkeyPatch, fake) -> None:
    monkeypatch.setattr("finetune_studio.data.prep.generator.resolve_generator",
                        lambda: lambda messages, **kw: fake(messages[-1]["content"]))


def _url(pid: str, tail: str = "") -> str:
    return f"/api/projects/{pid}/data-prep/preference{tail}"


def test_builds_and_returns_counts_and_dataset(api, monkeypatch) -> None:
    proj = seed_project()
    _use_model(monkeypatch, FakeModel())
    r = api.post(_url(proj["id"]), json={"kinds": ["hallucination", "abstain"], "max_pairs": 6, "seed": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pairs"] == 6 and body["by_kind"] == {"hallucination": 3, "abstain": 3}
    assert body["dataset"]["rows"] == 6 and body["dataset"]["name"] == "pref-docs · preference · 6 pairs"
    assert body["length"]["ratio"] > 0 and body["split"] == {"train": 5, "val": 1}
    assert body["train_url"] == f"/projects/{proj['id']}/training?dataset={body['dataset']['id']}&mode=dpo"
    (ds,) = list_datasets(proj["id"])
    assert ds["id"] == body["dataset"]["id"]


def test_defaults_build_both_kinds(api, monkeypatch) -> None:
    proj = seed_project()
    _use_model(monkeypatch, FakeModel())
    r = api.post(_url(proj["id"]), json={})
    assert r.status_code == 200 and set(r.json()["by_kind"]) == {"hallucination", "abstain"}


def test_unknown_project_is_404(api) -> None:
    assert api.post(_url("nope"), json={}).status_code == 404
    assert api.get(_url("nope", "/progress")).status_code == 404


def test_no_approved_pairs_is_409(api, monkeypatch) -> None:
    proj = seed_project(status="pending")
    _use_model(monkeypatch, FakeModel())
    r = api.post(_url(proj["id"]), json={})
    assert r.status_code == 409 and r.json()["code"] == "no_approved_pairs"
    assert list_datasets(proj["id"]) == []


def test_no_loaded_model_is_409_not_a_fabricated_dataset(api, monkeypatch) -> None:
    proj = seed_project()
    monkeypatch.setattr("finetune_studio.data.prep.generator.resolve_generator", lambda: None)
    r = api.post(_url(proj["id"]), json={})
    assert r.status_code == 409 and r.json()["code"] == "no_model" and r.json()["error"]
    assert list_datasets(proj["id"]) == []


def test_everything_gated_out_is_422_with_the_reasons(api, monkeypatch) -> None:
    proj = seed_project()
    _use_model(monkeypatch, FakeModel(hallucination="I don't know."))
    r = api.post(_url(proj["id"]), json={"kinds": ["hallucination"], "max_pairs": 4})
    assert r.status_code == 422 and r.json()["code"] == "no_usable_pairs"
    assert r.json()["dropped"]["hallucination"]["refusal_as_rejected"] >= 1


def test_model_failure_is_a_502(api, monkeypatch) -> None:
    proj = seed_project()

    def boom(prompt: str) -> str:
        raise RuntimeError("helper crashed")

    _use_model(monkeypatch, boom)
    r = api.post(_url(proj["id"]), json={"kinds": ["hallucination"]})
    assert r.status_code == 502 and "helper crashed" in r.json()["error"]


@pytest.mark.parametrize("payload", [{"kinds": []}, {"kinds": ["style"]}, {"max_pairs": 1},
                                     {"max_pairs": 999999}, {"seed": "x"}])
def test_bad_body_is_rejected_by_the_typed_model(api, payload) -> None:
    proj = seed_project()
    assert api.post(_url(proj["id"]), json=payload).status_code == 422


def test_build_runs_off_the_event_loop_and_reports_progress(monkeypatch) -> None:
    proj = seed_project()
    fake, seen = FakeModel(), {}

    def slow(messages, **kw):
        seen.setdefault("thread", threading.current_thread().name)
        time.sleep(0.05)
        return fake(messages[-1]["content"])

    monkeypatch.setattr("finetune_studio.data.prep.generator.resolve_generator", lambda: slow)
    app = FastAPI()
    app.include_router(route.router, prefix="/api")

    async def drive() -> tuple[int, list[dict], int]:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            build = asyncio.create_task(c.post(_url(proj["id"]), json={"max_pairs": 4}))
            await asyncio.sleep(0.3)
            polls, ticks = [], 0
            second = await c.post(_url(proj["id"]), json={})  # concurrent build for the same project
            while not build.done():
                polls.append((await c.get(_url(proj["id"], "/progress"))).json())
                await asyncio.sleep(0.05)
                ticks += 1
            assert (await build).status_code == 200
            return second.status_code, polls, ticks

    second, polls, ticks = asyncio.run(drive())
    assert seen["thread"] != threading.main_thread().name, "model calls ran on the event loop thread"
    assert second == 409 and ticks >= 3, "event loop was blocked by the build"
    running = [p for p in polls if p.get("running")]
    assert running and running[-1]["attempted"] > 0 and running[-1]["planned"] >= running[-1]["attempted"]


def test_progress_after_a_finished_build_is_not_running(api, monkeypatch) -> None:
    proj = seed_project()
    _use_model(monkeypatch, FakeModel())
    assert api.get(_url(proj["id"], "/progress")).json() == {"running": False}
    assert api.post(_url(proj["id"]), json={"max_pairs": 4}).status_code == 200
    done = api.get(_url(proj["id"], "/progress")).json()
    assert done["running"] is False and sum(done["kept"].values()) == 4
