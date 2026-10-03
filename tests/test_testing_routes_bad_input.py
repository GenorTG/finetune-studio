"""/api/testing/* must answer malformed input with 400, not 500."""


def test_evaluate_training_non_integer_max_cases_is_400(client):
    r = client.post("/api/testing/evaluate-training", json={"project_id": "p", "max_cases": "abc"})
    assert r.status_code == 400
