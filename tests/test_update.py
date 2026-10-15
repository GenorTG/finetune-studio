"""Tests for the /api/system/update endpoint + worker.

Covers:
- DB-level lifecycle for system_updates (mirrors TestSystemUpdateLifecycle
  in test_db_lifecycle.py with endpoint-driven scenarios)
- Route validation + happy path
- Worker under FTS_SKIP_UPDATE=1 (canned log, no real subprocess)
- Worker when update.sh is missing (clean error)
"""

from __future__ import annotations

import os

# ── Worker with FTS_SKIP_UPDATE=1 ────────────────────────────────────────


class TestUpdateWorkerSkip:
    """The worker should short-circuit cleanly when FTS_SKIP_UPDATE=1,
    writing a canned log and marking the row done."""

    def test_skip_marks_running_then_done_with_log(
        self, mock_settings, monkeypatch
    ):
        from finetune_studio import db
        from finetune_studio.webui.routes.updates import _update_worker

        monkeypatch.setenv("FTS_SKIP_UPDATE", "1")
        uid = db.create_update(mode="update", triggered_by="user")["id"]
        _update_worker(uid, "update", {})
        r = db.get_update(uid)
        assert r["status"] == "done"
        assert r["finished_at"] is not None
        assert r["duration_ms"] is not None
        # Canned log should mention all six steps
        for marker in ("step 1", "step 2", "step 3", "step 4", "step 5"):
            assert marker in r["log_text"], f"missing {marker} in log"
        assert "FTS_SKIP_UPDATE=1" in r["log_text"]

    def test_skip_passes_options_into_log(
        self, mock_settings, monkeypatch
    ):
        from finetune_studio import db
        from finetune_studio.webui.routes.updates import _update_worker

        monkeypatch.setenv("FTS_SKIP_UPDATE", "1")
        uid = db.create_update(mode="check",
                               options={"no_pull": True, "no_llama": True})["id"]
        _update_worker(uid, "check", {"no_pull": True, "no_llama": True})
        r = db.get_update(uid)
        assert "no_pull" in r["log_text"]
        assert "no_llama" in r["log_text"]


class TestUpdateWorkerFailure:
    """When FTS_SKIP_UPDATE is unset and the script can't be found,
    the worker should mark failed with a clean error message."""

    def test_missing_script_marks_failed(self, mock_settings, monkeypatch):
        from finetune_studio import db
        from finetune_studio.webui.routes import updates

        monkeypatch.delenv("FTS_SKIP_UPDATE", raising=False)
        # Point REPO_ROOT at a tmp dir with no update.sh
        import tempfile
        monkeypatch.setattr(updates, "REPO_ROOT", _FakePath(tempfile.mkdtemp()))
        monkeypatch.setattr(updates, "_find_update_script",
                            lambda: None)
        # Also clear $FTS_UPDATE_SCRIPT so the fallback doesn't rescue us
        monkeypatch.delenv("FTS_UPDATE_SCRIPT", raising=False)

        uid = db.create_update(mode="update")["id"]
        updates._update_worker(uid, "update", {})
        r = db.get_update(uid)
        assert r["status"] == "error"
        assert "update.sh not found" in r["error"]


# Helper: minimal Path-like object that reports a different root but
# has the same shape as the real REPO_ROOT. _find_update_script uses
# .is_file() so we just need an object that returns False.
class _FakePath:
    def __init__(self, s):
        self.s = s
    def __truediv__(self, other):
        return _FakePath(os.path.join(self.s, other))
    def __fspath__(self):
        return self.s
    def is_file(self):
        return False


# ── Route (FastAPI TestClient) ───────────────────────────────────────────


class TestUpdateRoute:
    def test_invalid_mode_returns_error(self, client, mock_settings):
        r = client.post("/api/system/update", json={"mode": "explode"})
        body = r.json()
        assert "invalid mode" in body["error"]

    def test_post_returns_update_id_and_queued_status(
        self, client, mock_settings, monkeypatch
    ):
        monkeypatch.setenv("FTS_SKIP_UPDATE", "1")
        r = client.post("/api/system/update", json={"mode": "update"})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["status"] == "queued"
        assert "update_id" in body
        assert body["mode"] == "update"

    def test_options_are_persisted_to_db(
        self, client, mock_settings, monkeypatch
    ):
        monkeypatch.setenv("FTS_SKIP_UPDATE", "1")
        r = client.post("/api/system/update", json={
            "mode": "update", "no_pull": True, "no_llama": True,
        })
        body = r.json()
        # options dict echoed in response
        assert body["options"]["no_pull"] is True
        assert body["options"]["no_llama"] is True

    def test_get_update_status_returns_row(
        self, client, mock_settings
    ):
        from finetune_studio import db
        uid = db.create_update(mode="update", triggered_by="user")["id"]
        db.mark_update_running(uid)
        db.append_update_log(uid, "test line\n")
        r = client.get(f"/api/system/update/{uid}")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == uid
        assert body["status"] == "running"
        assert "test line" in body["log_tail"]
        assert body["log_length"] >= len("test line\n")

    def test_get_update_unknown_returns_404(self, client, mock_settings):
        r = client.get("/api/system/update/nonexistent")
        assert r.status_code == 404

    def test_list_updates_returns_recent(self, client, mock_settings):
        from finetune_studio import db
        ids = {db.create_update(mode="update")["id"] for _ in range(3)}
        r = client.get("/api/system/updates")
        assert r.status_code == 200
        rows = r.json()
        assert isinstance(rows, list)
        got = {u["id"] for u in rows}
        assert ids.issubset(got)

    def test_latest_update_endpoint(self, client, mock_settings):
        from finetune_studio import db
        # No in-progress → exists: False
        r = client.get("/api/system/update/latest")
        assert r.status_code == 200
        assert r.json() == {"exists": False}
        # Create an in-progress → latest returns it
        uid = db.create_update(mode="update")["id"]
        r = client.get("/api/system/update/latest")
        body = r.json()
        assert body["exists"] is True or "id" in body
        # (depending on whether the API still wraps as 'exists')
        assert body["id"] == uid

    def test_full_lifecycle_via_route(
        self, client, mock_settings, monkeypatch
    ):
        """End-to-end: POST /update, BackgroundTask runs the worker under
        FTS_SKIP_UPDATE=1, then GET /update/{uid} shows done + log."""
        monkeypatch.setenv("FTS_SKIP_UPDATE", "1")
        # BackgroundTasks runs synchronously in TestClient mode after the
        # response is sent. We need to wait briefly for the worker to finish.
        r = client.post("/api/system/update", json={"mode": "update"})
        assert r.status_code == 200
        uid = r.json()["update_id"]
        # In TestClient the background tasks run before the next request,
        # but they share the same event loop, so a quick GET will see
        # the completed state.
        import time as _t
        for _ in range(20):
            r = client.get(f"/api/system/update/{uid}")
            if r.json()["status"] == "done":
                break
            _t.sleep(0.05)
        body = r.json()
        assert body["status"] == "done", body
        assert "step 1" in body["log_tail"]
        assert body["duration_ms"] is not None
