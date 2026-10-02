"""/api/testing/* must answer malformed input with 400, not 500."""


def test_chat_bad_json_and_non_object_are_400(client):
    r = client.post("/api/testing/chat", content="x", headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert client.post("/api/testing/chat", json=[1]).status_code == 400


def test_evaluate_training_non_integer_max_cases_is_400(client):
    r = client.post("/api/testing/evaluate-training", json={"project_id": "p", "max_cases": "abc"})
    assert r.status_code == 400
