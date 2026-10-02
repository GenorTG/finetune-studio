"""Loading a nonexistent local model path must give a clean error, not an HF repo-id message."""


def test_load_missing_local_path_is_clean_error(client):
    r = client.post("/api/inference/load", json={"model_path": "/nonexistent/x"})
    d = r.json()
    assert d["status"] == "error" and d["loaded"] is False
    assert "model path does not exist" in d["error"]
    assert "Repo id" not in d["error"]
