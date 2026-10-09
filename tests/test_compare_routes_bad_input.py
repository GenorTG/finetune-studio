"""/api/compare/projects/* must answer malformed input with 400/404, not 500 (the keyword-scoring /compare/* routes are gone)."""


def test_compare_bad_body_is_400(client):
    pid = client.post("/api/projects", json={"name": "bad-body", "base_model": "x/test"}).json()["id"]
    h = {"content-type": "application/json"}
    assert client.post(f"/api/compare/projects/{pid}/runs", content="x", headers=h).status_code == 400
    assert client.post(f"/api/compare/projects/{pid}/runs", json=[1]).status_code == 400
    assert client.post(f"/api/compare/projects/{pid}/groups/g/judge", content="x", headers=h).status_code == 400


def test_old_keyword_scoring_routes_are_gone(client):
    for path in ("load", "run", "cleanup"):
        assert client.post(f"/api/compare/compare/{path}", json={}).status_code == 404
