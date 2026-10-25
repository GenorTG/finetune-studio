"""HTTP contract of the async export job: 202 queue ack, 409, cancel, active, SSE."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from finetune_studio import db
from finetune_studio.webui import export_jobs
from tests._export_wait import wait_export
from tests._fake_llama import EngineStub, install_fake_llama, make_merged_run


@pytest.fixture
def setup(client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    install_fake_llama(tmp_path / "llama", monkeypatch)
    pid = client.post("/api/projects", json={"name": "Routes"}).json()["id"]
    run = make_merged_run(db, pid, tmp_path)
    # The client fixture MagicMocks TrainingEngine; the fake toolchain needs a real export step.
    # A ``patch`` context (not monkeypatch) so it unwinds BEFORE the client fixture's own patch;
    # monkeypatch would undo last and leave the MagicMock behind for every later test.
    with patch("finetune_studio.training.engine.TrainingEngine", EngineStub):
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
    assert first["export_id"] in body["detail"]
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


def _with_training_data(run: dict, tmp: Path) -> dict:
    data = tmp / "train.jsonl"
    data.write_text(json.dumps({"messages": [{"role": "user", "content": "What is the Korvane tag?"},
                                             {"role": "assistant", "content": "The KC-Tag 4."}]}) + "\n")
    db.update_run(run["id"], data_path=str(data))
    return db.get_run(run["id"])


def test_imatrix_export_measures_on_the_training_data_and_quantizes_with_it(setup) -> None:
    client, pid, run, tmp = setup
    run = _with_training_data(run, tmp)
    r = _post(client, pid, run, format="gguf", quants=["q4_k_m"], imatrix=True)
    assert r.status_code == 202, r.text
    final = wait_export(client, pid, r.json()["export_id"])
    assert final["status"] == "done", final
    out = Path(final["output_path"])
    assert out.parent.name == "gguf-imatrix" and out.name == "model-q4_k_m.gguf"
    imatrix = out.parent / "imatrix.dat"
    assert out.read_text().strip() == f"quantized imatrix={imatrix}"       # llama-quantize got --imatrix
    assert "The KC-Tag 4." in (out.parent / "imatrix.dat.seen-calibration").read_text()
    assert not (out.parent / "imatrix.dat.calibration.txt").exists()        # the temp text is cleaned up
    assert not (Path(run["output_path"]) / "gguf").exists()                  # the plain export is untouched


def test_imatrix_export_refusals_say_why(setup) -> None:
    client, pid, run, tmp = setup
    no_data = _post(client, pid, run, format="gguf", quants=["q4_k_m"], imatrix=True)   # data_path "/d" is gone
    assert no_data.status_code == 400 and "training data" in no_data.json()["detail"]
    run = _with_training_data(run, tmp)
    only_q8 = _post(client, pid, run, format="gguf", quants=["q8_0"], imatrix=True)
    assert only_q8.status_code == 400 and "llama-quantize" in only_q8.json()["detail"]
    merged = _post(client, pid, run, format="merged", imatrix=True)
    assert merged.status_code == 400 and "only applies to a GGUF" in merged.json()["detail"]
    assert _post(client, pid, run, format="gguf", quants=["q4_k_m"], imatrix="yes").status_code == 400


def test_imatrix_measuring_failure_fails_the_job_with_the_reason(setup, monkeypatch) -> None:
    client, pid, run, tmp = setup
    run = _with_training_data(run, tmp)
    monkeypatch.setenv("FAKE_IMATRIX_FAIL", "out of memory measuring")
    eid = _post(client, pid, run, format="gguf", quants=["q4_k_m"], imatrix=True).json()["export_id"]
    final = wait_export(client, pid, eid)
    assert final["status"] == "failed" and "out of memory measuring" in final["error"]


def test_old_quantize_routes_are_gone(client) -> None:
    assert client.post("/api/training/runs/x/quantize", json={}).status_code in (404, 405)
    assert client.get("/api/training/runs/x/quant-exports").status_code in (404, 405)
    assert client.post("/api/training/runs/x/set-output", json={"output_path": "/tmp"}).status_code in (404, 405)
