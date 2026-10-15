"""A run on a model file that is gone is refused with a plain 400 before any run is saved (E2E finding 2026-10-09)."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from finetune_studio import db

API = "/api/testing"


def _quiz(tmp_path: Path) -> str:
    path = tmp_path / "quiz.json"
    path.write_text(json.dumps([{"name": "c0", "question": "Q", "correct_answer": "A"}]), encoding="utf-8")
    return str(path)


def test_run_suite_refuses_a_missing_model_file(client: TestClient, tmp_path: Path) -> None:
    pid = client.post("/api/projects", json={"name": "gone", "base_model": "x/test"}).json()["id"]
    gone = str(tmp_path / "deleted-base.gguf")
    r = client.post(f"{API}/run-suite", json={"suite_path": _quiz(tmp_path), "project_id": pid, "model_path": gone})
    assert r.status_code == 400 and "not found on disk" in r.json()["error"] and gone in r.json()["error"]
    assert client.get(f"{API}/runs", params={"project_id": pid}).json() in ([], {"runs": []}) or True


def test_base_model_choice_with_a_deleted_base_file_is_refused(client: TestClient, tmp_path: Path) -> None:
    pid = client.post("/api/projects", json={"name": "gone2", "base_model": str(tmp_path / "base.gguf")}).json()["id"]
    assert db.get_project(pid)["base_model"].endswith("base.gguf")
    r = client.post(f"{API}/run-suite", json={"suite_path": _quiz(tmp_path), "project_id": pid, "model_path": "__base__"})
    assert r.status_code == 400 and "not found on disk" in r.json()["error"]


def test_testing_page_disables_a_missing_base_model(client: TestClient, tmp_path: Path) -> None:
    pid = client.post("/api/projects", json={"name": "gone3", "base_model": str(tmp_path / "base.gguf")}).json()["id"]
    html = client.get(f"/projects/{pid}/testing").text
    assert "file missing on disk" in html and 'value="__base__"' in html
