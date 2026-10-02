"""/api/training/start must reject bad input with a clean 4xx/error, never a 500."""
from __future__ import annotations

from tests.test_training_start_guard import (  # noqa: F401  (fixtures)
    _jsonl,
    _project,
    _start,
    fake_engine,
    fake_home,
)


def test_non_numeric_param_is_400(client, fake_engine, fake_home, tmp_path):
    pid = _project(client)
    model = tmp_path / "m"
    model.mkdir()
    r = _start(client, pid, _jsonl(tmp_path), str(model), num_epochs="abc")
    assert r.status_code == 400, r.text
    assert "invalid training parameter" in r.json()["error"]
    assert not fake_engine.started


def test_non_positive_param_is_400(client, fake_engine, fake_home, tmp_path):
    pid = _project(client)
    model = tmp_path / "m"
    model.mkdir()
    for key in ("num_epochs", "batch_size", "max_seq_length"):
        r = _start(client, pid, _jsonl(tmp_path), str(model), **{key: 0})
        assert r.status_code == 400, (key, r.text)
        assert key in r.json()["error"]
    assert not fake_engine.started


def test_bad_json_and_non_object_body_are_400(client, fake_engine, fake_home):
    r = client.post("/api/training/start", content="{bad", headers={"content-type": "application/json"})
    assert r.status_code == 400
    r = client.post("/api/training/start", json=[1])
    assert r.status_code == 400


def test_missing_local_model_path_is_rejected_synchronously(client, fake_engine, fake_home, tmp_path):
    pid = _project(client)
    r = _start(client, pid, _jsonl(tmp_path), "/nonexistent/model")
    assert "model path does not exist" in r.json().get("error", ""), r.text
    assert not fake_engine.started


def test_export_bad_body_is_400(client, fake_engine, fake_home):
    r = client.post("/api/training/runs/nope/export", content="x", headers={"content-type": "application/json"})
    assert r.status_code == 400
    r = client.post("/api/training/runs/nope/export", json=[1])
    assert r.status_code == 400
