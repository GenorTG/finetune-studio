"""HTTP contract of the async export job: 202 queue ack, 409, cancel, active, SSE."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from finetune_studio import db
from finetune_studio.webui import export_jobs
from tests._export_wait import wait_export
from tests._fake_llama import EngineStub, install_fake_llama, make_merged_run


@pytest.fixture
def setup(client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    install_fake_llama(tmp_path / "llama", monkeypatch)
    # the client fixture MagicMocks TrainingEngine; the fake toolchain needs a real export step
    monkeypatch.setattr("finetune_studio.training.engine.TrainingEngine", EngineStub)
    pid = client.post("/api/projects", json={"name": "Routes"}).json()["id"]
    run = make_merged_run(db, pid, tmp_path)
    yield client, pid, run, tmp_path
    deadline = time.monotonic() + 20
    while export_jobs.active_ids() and time.monotonic() < deadline:
        time.sleep(0.05)


def _post(client, pid, run, **body):
    return client.post(f"/api/projects/{pid}/runs/{run['id']}/export", json=body)


def test_post_acks_immediately_and_the_row_carries_the_job_state(setup, monkeypatch) -> None:
    client, pid, run, _ = setup
    monkeypatch.setenv("FAKE_CONVERT_SECONDS", "1")
    t0 = time.monotonic()
    r = _post(client, pid, run, format="gguf", quants=["f16"])
    assert time.monotonic() - t0 < 1.0
    assert r.status_code == 202
    body = r.json()
    assert body["ok"] is True and body["status"] == "queued"
    assert body["quants"] == ["f16"] and body["format"] == "gguf"
    row = client.get(f"/api/projects/{pid}/exports/{body['export_id']}").json()
    assert row["status"] in ("queued", "running")
    assert row["active"] is True and row["quants"] == ["f16"]
    assert row["server_now"] >= row["created_at"]
    final = wait_export(client, pid, body["export_id"])
    assert final["status"] == "done" and final["active"] is False
    assert final["output_path"].endswith("model-f16.gguf")


def test_single_quant_mode_also_queues_and_names_the_output(setup) -> None:
    client, pid, run, _ = setup
    r = _post(client, pid, run, format="gguf", quant="Q8_0")
    assert r.status_code == 202
    assert r.json()["output_path"].endswith("Qwen3-0.6B-Q8_0.gguf")
    final = wait_export(client, pid, r.json()["export_id"])
    assert final["status"] == "done"
    assert final["output_path"].endswith("Qwen3-0.6B-Q8_0.gguf")
    assert Path(final["output_path"]).stat().st_size > 0


def test_second_export_gets_409_naming_the_active_one(setup, monkeypatch) -> None:
    client, pid, run, _ = setup
    monkeypatch.setenv("FAKE_CONVERT_SECONDS", "1.5")
    first = _post(client, pid, run, format="gguf", quants=["f16"]).json()
    second = _post(client, pid, run, format="gguf", quants=["q8_0"])
    assert second.status_code == 409
    body = second.json()
    assert body["ok"] is False and body["status"] == "busy"
    assert body["active_export_id"] == first["export_id"]
    assert first["export_id"] in body["error"]
    wait_export(client, pid, first["export_id"])
    assert _post(client, pid, run, format="gguf", quants=["q8_0"]).status_code == 202


def test_failed_job_row_and_sse_carry_the_error(setup, monkeypatch) -> None:
    client, pid, run, _ = setup
    monkeypatch.setenv("FAKE_CONVERT_FAIL", "segfault in ggml_quantize")
    eid = _post(client, pid, run, format="gguf", quants=["f16"]).json()["export_id"]
    with client.stream("GET", f"/api/projects/{pid}/exports/{eid}/events") as resp:
        frames = [json.loads(ln[6:]) for ln in resp.iter_lines() if ln.startswith("data: ")]
    assert frames, "SSE produced no frames"
    last = frames[-1]
    assert last["status"] == "failed" and "segfault in ggml_quantize" in last["error"]
    assert {"phase", "phase_detail", "heartbeat_at", "server_now"} <= set(last)


def test_cancel_route(setup, monkeypatch) -> None:
    client, pid, run, _ = setup
    monkeypatch.setenv("FAKE_QUANT_SECONDS", "30")
    eid = _post(client, pid, run, format="gguf", quants=["q4_k_m"]).json()["export_id"]
    deadline = time.monotonic() + 20
    while db.get_export(eid)["phase"] != "quantizing" and time.monotonic() < deadline:
        time.sleep(0.05)
    r = client.post(f"/api/projects/{pid}/exports/{eid}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "cancelling"
    assert wait_export(client, pid, eid, timeout=20)["status"] == "cancelled"
    again = client.post(f"/api/projects/{pid}/exports/{eid}/cancel")
    assert again.status_code == 409
    assert client.post(f"/api/projects/{pid}/exports/nope/cancel").status_code == 404
    assert client.post("/api/projects/nope/exports/x/cancel").status_code == 404


def test_active_endpoint_lists_running_jobs_for_reattach(setup, monkeypatch) -> None:
    client, pid, run, _ = setup
    other = client.post("/api/projects", json={"name": "Other"}).json()["id"]
    assert client.get(f"/api/projects/{pid}/exports/active").json() == {"project": [], "other": []}
    monkeypatch.setenv("FAKE_CONVERT_SECONDS", "1.5")
    eid = _post(client, pid, run, format="gguf", quants=["f16"]).json()["export_id"]
    mine = client.get(f"/api/projects/{pid}/exports/active").json()
    assert [r["id"] for r in mine["project"]] == [eid] and mine["other"] == []
    theirs = client.get(f"/api/projects/{other}/exports/active").json()
    assert theirs["project"] == [] and [r["id"] for r in theirs["other"]] == [eid]
    wait_export(client, pid, eid)
    assert client.get(f"/api/projects/{pid}/exports/active").json()["project"] == []
    assert client.get("/api/projects/nope/exports/active").status_code == 404


def test_imatrix_quantize_is_a_tracked_job_too(setup, monkeypatch) -> None:
    client, pid, run, tmp = setup
    imatrix = tmp / "imatrix.dat"
    imatrix.write_bytes(b"im")
    monkeypatch.setenv("FAKE_QUANT_SECONDS", "0.5")
    r = client.post(f"/api/training/runs/{run['id']}/quantize",
                    json={"method": "imatrix", "imatrix_path": str(imatrix),
                          "quants": ["q4_k_m"]})
    assert r.status_code == 202, r.text
    eid = r.json()["export_id"]
    final = wait_export(client, pid, eid)
    assert final["status"] == "done", final
    assert final["format"] == "imatrix" and final["size_bytes"] > 0
    listed = client.get(f"/api/training/runs/{run['id']}/quant-exports").json()
    assert len(listed) == 1 and listed[0]["status"] == "done"
    # synchronous refusals keep their codes
    bad = client.post(f"/api/training/runs/{run['id']}/quantize",
                      json={"method": "imatrix", "imatrix_path": "/nope"})
    assert bad.status_code == 422 and "imatrix file not found" in bad.json()["error"]


def test_imatrix_all_quants_failing_marks_the_job_failed(setup, monkeypatch) -> None:
    client, pid, run, tmp = setup
    imatrix = tmp / "imatrix.dat"
    imatrix.write_bytes(b"im")
    monkeypatch.setenv("FAKE_QUANT_FAIL", "imatrix does not match model")
    r = client.post(f"/api/training/runs/{run['id']}/quantize",
                    json={"imatrix_path": str(imatrix), "quants": ["q4_k_m"]})
    final = wait_export(client, pid, r.json()["export_id"])
    assert final["status"] == "failed"
    assert "imatrix does not match model" in final["error"]
    assert client.get(f"/api/training/runs/{run['id']}/quant-exports").json() == []
