"""/api/testing/* must answer malformed input with 400, not 500."""
import pytest


@pytest.fixture
def pid(client) -> str:
    return client.post("/api/projects", json={"name": "bad-input", "base_model": "x/test"}).json()["id"]


def test_evaluate_training_non_integer_max_cases_is_400(client, pid):
    r = client.post("/api/testing/evaluate-training", json={"project_id": pid, "max_cases": "abc"})
    assert r.status_code == 400


def test_evaluate_training_non_numeric_temperature_is_400(client, pid):
    r = client.post("/api/testing/evaluate-training", json={"project_id": pid, "temperature": "hot"})
    assert r.status_code == 400


def test_evaluate_training_unknown_project_is_404(client):
    r = client.post("/api/testing/evaluate-training", json={"project_id": "nope", "max_cases": "abc"})
    assert r.status_code == 404


@pytest.mark.parametrize("path", ["run-suite", "run-rag-suite", "evaluate-training"])
@pytest.mark.parametrize("raw", ["[1]", "not json"])
def test_run_routes_reject_a_non_object_body(client, path, raw):
    r = client.post(f"/api/testing/{path}", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 400
