"""R8 contracts: actual exported runtime, private bounded storage, and inert Studio intake."""
from __future__ import annotations

import json
import sqlite3
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest

from finetune_studio.smart_brain import package_usage as usage
from finetune_studio.smart_brain.usage_report import suggestions
from tests import test_brain_server_package as packages
from tests.test_package_knowledge_api import CHAT, host, request


@pytest.fixture()
def corpus_dir(tmp_path):
    return packages.corpus_dir.__wrapped__(tmp_path)


def test_counts_query_opt_in_erasure_retention_and_feedback(tmp_path, monkeypatch):
    tracker = usage.Usage(tmp_path)
    secret = "private-question-f615938"
    owner_key = tracker.owner_token()
    assert len(owner_key) == 64 and tracker.owner_token() == owner_key
    assert (tracker.root / "owner-token").stat().st_mode & 0o777 == 0o600
    with tracker.measure("search", secret) as event:
        event["result"] = [{"filename": "a"}, {"filename": "a"}, {"filename": "b"}]
    report = tracker.report()
    assert report["settings"] == {"store_queries": False, "retention_days": 30}
    assert report["summary"]["questions"] == 1
    assert report["summary"]["sources_found"] == 2
    assert report["events"][0]["query"] is None
    dbpath = tracker.root / "usage.sqlite3"
    assert dbpath.stat().st_mode & 0o777 == 0o600
    assert tracker.root.stat().st_mode & 0o777 == 0o700
    assert secret.encode() not in dbpath.read_bytes()
    assert owner_key.encode() not in dbpath.read_bytes()
    tracker.feedback({"id": event["id"], "rating": "down"})
    tracker.feedback({"id": event["id"], "rating": "down"})
    assert tracker.report()["summary"]["thumbs_down"] == 1
    tracker.feedback({"id": event["id"], "rating": "up"})
    assert tracker.report()["summary"]["thumbs_down"] == 0
    tracker.settings({"store_queries": True, "retention_days": 1})
    with tracker.measure("answer", secret):
        pass
    assert tracker.report()["events"][0]["query"] == secret
    tracker.settings({"store_queries": False, "retention_days": 1})
    assert secret.encode() not in dbpath.read_bytes()
    with sqlite3.connect(dbpath) as conn:
        conn.execute("UPDATE events SET at=?", (time.time() - 86401,))
    assert tracker.report()["summary"]["questions"] == 0
    with pytest.raises(ValueError, match="expired"):
        tracker.feedback({"id": event["id"], "rating": "down"})
    monkeypatch.setattr(usage, "MAX_EVENTS", 3)
    for _ in range(5):
        with tracker.measure("search", "x"):
            pass
    assert tracker.report()["summary"]["questions"] == 3
    tracker.clear()
    assert tracker.report()["events"] == []


def test_nested_operations_errors_concurrency_and_no_raw_exceptions(tmp_path):
    tracker = usage.Usage(tmp_path)
    with tracker.measure("answer", "user question") as event, tracker.measure("search", "follow-up") as nested:
        assert nested is event
        event["result"] = {"sources": [{"filename": "a"}], "weak": True}
    with pytest.raises(RuntimeError), tracker.measure("answer", "private"):
        raise RuntimeError("PRIVATE_PROVIDER_KEY_AND_PASSAGE")
    assert tracker.report()["summary"]["questions"] == 2
    assert tracker.report()["summary"]["errors"] == 1
    assert tracker.report()["summary"]["low_confidence"] == 1
    assert "PRIVATE_PROVIDER" not in json.dumps(tracker.report())
    def record(_):
        with tracker.measure("search", "x"):
            pass
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(record, range(40)))
    assert tracker.report()["summary"]["questions"] == 42


def test_storage_failure_does_not_break_query(tmp_path, monkeypatch):
    tracker = usage.Usage(tmp_path)
    def broken():
        raise OSError("disk full")
    monkeypatch.setattr(tracker, "db", broken)
    with tracker.measure("search", "x") as event:
        event["result"] = []
    assert tracker.local.last_id is None


def test_hidden_answer_keeps_distinct_source_union_across_followups(tmp_path):
    class Corpus:
        def __init__(self):
            self.usage = usage.Usage(tmp_path)

        @usage.tracked("search")
        def search(self, query):
            return [{"document_id": query + "~p1"}, {"document_id": query + "~p2"}]

        @usage.tracked("answer")
        def answer(self, question):
            self.search("first-document")
            self.search("second-document")
            return {"sources": [], "weak": False, "answer": "A hidden-source answer."}

    corpus = Corpus()
    corpus.answer("question")
    report = corpus.usage.report()
    assert report["summary"]["questions"] == 1
    assert report["summary"]["sources_found"] == 2
    assert "first-document" not in json.dumps(report)


def test_answer_followups_and_mcp_share_persistent_single_operation_tracking(corpus_dir, tmp_path, monkeypatch):
    from finetune_studio.smart_brain import standalone_server as srv
    corpus = packages._in_process_corpus(corpus_dir, tmp_path)
    corpus.cfg.update(answer=True, follow_up=0, hide_file_names=True)
    monkeypatch.setattr(corpus, "_chat", lambda *_: "That detail is not stated in these documents.")
    answer = corpus.answer(question="Highmere Stair salt", include_sources=False)
    assert answer["sources"] == []
    report = corpus.usage.report()
    assert report["summary"]["questions"] == 1
    assert report["summary"]["sources_found"] > 0
    assert report["summary"]["not_found"] == 1
    responder = srv.mcp_responder(corpus)
    result = responder({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": "rag_search", "arguments": {"query": "Highmere salt"}}})
    assert len(result["result"]["_meta"]["smart-brain/usage-id"]) == 32
    reopened = usage.Usage(corpus.dir.parent)
    assert reopened.report()["summary"]["questions"] == 2
    assert ".usage" in srv.UPDATE_KEEP


@pytest.mark.parametrize("body", [{"store_queries": "false", "retention_days": 30},
                                  {"store_queries": False, "retention_days": True},
                                  {"store_queries": True, "retention_days": 0},
                                  {"store_queries": True, "retention_days": 366}, {"unknown": 2}])
def test_invalid_privacy_settings_rejected_without_mutation(tmp_path, body):
    tracker = usage.Usage(tmp_path)
    with pytest.raises(ValueError):
        tracker.settings(body)
    assert tracker.report()["settings"] == {"store_queries": False, "retention_days": 30}


def test_usage_owner_rejects_forwarded_remote_hosts_and_reports_storage_failure(corpus_dir, tmp_path, monkeypatch):
    import threading

    from finetune_studio.smart_brain import standalone_server as srv
    corpus = packages._in_process_corpus(corpus_dir, tmp_path)
    server = srv._Server(("127.0.0.1", 0), srv.make_handler(corpus, "api-test-token"))
    owner_headers = {"X-Usage-Owner": corpus.usage.owner_token()}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        for extra in ({"X-Forwarded-For": "198.51.100.5"}, {"Host": "public.example"},
                      {"Forwarded": "for=198.51.100.5"}, {"X-Real-IP": "198.51.100.5"}):
            req = urllib.request.Request(base + "/usage/report", headers={"Authorization": "Bearer api-test-token", **extra})
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(req, timeout=5)
            assert error.value.code == 403
        def broken():
            raise sqlite3.OperationalError("PRIVATE_FILESYSTEM_PATH")
        monkeypatch.setattr(corpus.usage, "db", broken)
        status, text, _ = request(base, "/usage/report", extra_headers=owner_headers)
        assert status == 503 and "PRIVATE_FILESYSTEM_PATH" not in text
        status, text, headers = request(base, "/search", {"query": "Highmere"})
        assert status == 200 and "hits" in text
        assert "X-Usage-ID" not in headers
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("encrypted", [False, True])
def test_actual_export_tracking_auth_feedback_errors_privacy_and_gap_export(corpus_dir, tmp_path, encrypted):
    with host(corpus_dir, tmp_path, encrypted=encrypted, hidden=True) as (base, root):
        assert (root / "usage.html").is_file()
        assert request(base, "/usage")[0] == 200
        assert request(base, "/usage/report", token="")[0] == 401
        assert request(base, "/usage/report", origin="https://other.example")[0] == 403
        assert request(base, "/usage/settings", {"store_queries": True, "retention_days": 2},
                       origin="https://other.example")[0] == 403
        assert request(base, "/usage/report")[0] == 403  # shared client token cannot read other users' history
        owner_key = (root / ".usage/owner-token").read_text().strip()
        def owner(path, body=None):
            return request(base, path, body, extra_headers={"X-Usage-Owner": owner_key})
        assert request(base, "/usage/report", extra_headers={"X-Usage-Owner": "wrong"})[0] == 403
        status, text, headers = request(base, "/v1/retrieve", {"query": "Highmere Stair salt"})
        assert status == 200
        assert "filename" not in text
        request_id = headers["X-Usage-ID"]
        assert request(base, "/usage/feedback", {"id": request_id, "rating": "down"})[0] == 200
        assert request(base, "/usage/feedback", {"id": request_id, "rating": "sideways"})[0] == 400
        assert request(base, "/v1/chat/completions", CHAT)[0] == 503
        assert request(base, "/v1/retrieve", {"query": "quuxxyyzz"})[0] == 200
        report = json.loads(owner("/usage/report")[1])
        assert report["summary"]["questions"] == 3
        assert report["summary"]["errors"] == 1
        assert report["summary"]["thumbs_down"] == 1
        assert report["summary"]["not_found"] == 1
        assert report["summary"]["sources_found"] == 1
        assert all(row["query"] is None for row in report["events"])
        assert report["summary"]["latency_mean_ms"] > 0
        assert "private-upstream-key" not in json.dumps(report)
        assert owner("/usage/settings", {"store_queries": True, "retention_days": 2})[0] == 200
        request(base, "/search", {"query": "Highmere salt quuxxyyzz unknown"})
        gaps = json.loads(owner("/usage/gaps")[1])
        assert "events" not in gaps
        assert any(row["query"] for row in gaps["gaps"])
        parsed = suggestions(json.dumps(gaps).encode())
        assert parsed["suggestions"]
        assert owner("/usage/settings", {"store_queries": False, "retention_days": 2})[0] == 200
        assert all(r["query"] is None for r in json.loads(owner("/usage/export")[1])["events"])
        assert owner("/usage/clear", {})[0] == 200
        assert json.loads(owner("/usage/report")[1])["summary"]["questions"] == 0


def gap_report():
    return {"schema": "smart-brain-usage-v1", "package": {"package_id": "p1", "name": "Example", "version": "1.0.0"},
            "gaps": [{"query": "<script>inert</script>", "not_found": 1, "weak": 0, "feedback": -1, "at": time.time(),
                      "filename": "/must/not/be/used", "url": "http://must-not-fetch.invalid"}]}


def test_studio_report_intake_is_project_scoped_inert_and_deletable(client, tmp_path, monkeypatch):
    import finetune_studio.webui.routes.brain_export as routes
    monkeypatch.setattr(routes, "project_dir", lambda pid: tmp_path / pid)
    pid = client.post("/api/projects", json={"name": "agent-test-usage"}).json()["id"]
    endpoint = f"/api/projects/{pid}/brain/usage-gaps"
    raw = json.dumps(gap_report()).encode()
    missing = client.post("/api/projects/missing/brain/usage-gaps", files={"file": ("report.json", raw)})
    assert missing.status_code == 404
    assert not (tmp_path / "missing").exists()
    assert client.get(endpoint).json()["suggestions"] == []
    response = client.post(endpoint, files={"file": ("report.json", raw)})
    assert response.status_code == 200, response.text
    assert response.json()["suggestions"][0]["question"] == "<script>inert</script>"
    stored = client.get(endpoint).json()
    assert "must-not-fetch" not in json.dumps(stored)
    assert "/must/not" not in json.dumps(stored)
    assert [p.name for p in (tmp_path / pid).iterdir()] == ["package-usage-gaps.json"]
    assert client.post(endpoint, files={"file": ("bad.json", b"{bad")}).status_code == 400
    assert client.get(endpoint).json() == stored  # rejected reports do not replace the last valid one
    assert client.delete(endpoint).status_code == 200
    assert client.get(endpoint).json()["suggestions"] == []


@pytest.mark.parametrize("change", [{"schema": "wrong"}, {"gaps": "bad"}, {"package": []},
                                  {"gaps": [{"query": 3}]}, {"gaps": [{"query": "x" * 2001}]}])
def test_malformed_report_rejected(change):
    with pytest.raises(ValueError):
        suggestions(json.dumps({**gap_report(), **change}).encode())
