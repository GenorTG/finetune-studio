"""/api/compare/compare/* must answer malformed input with 400, not 500."""


def test_compare_bad_body_is_400(client):
    h = {"content-type": "application/json"}
    assert client.post("/api/compare/compare/load", content="x", headers=h).status_code == 400
    assert client.post("/api/compare/compare/run", content="x", headers=h).status_code == 400
    assert client.post("/api/compare/compare/run", json=[1]).status_code == 400


def test_compare_run_malformed_case_is_400(client, monkeypatch):
    from finetune_studio.benchmarks import comparison

    monkeypatch.setattr(comparison.comparator, "engines", {"m": object()})
    r = client.post("/api/compare/compare/run", json={"test_suite": [{"question": "q"}]})
    assert r.status_code == 400
