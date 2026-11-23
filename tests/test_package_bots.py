"""Real loopback HTTP bridge -> exported scoped package -> synthetic operator LLM.

Vendor callback transport is redirected ONLY by the test to a local collector. No
vendor account, real credential, public listener or external message is used.
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from finetune_studio.smart_brain import package_bots as bots
from tests import test_brain_server_package as packages
from tests.test_package_knowledge_api import host, request, upstream  # noqa: F401

PRIVATE = Ed25519PrivateKey.generate()
PUBLIC = PRIVATE.public_key().public_bytes_raw().hex()
KEYS = {"slack": "synthetic-slack-secret", "teams": base64.b64encode(b"synthetic-teams-key").decode(),
        "discord": PUBLIC}


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    return packages.corpus_dir.__wrapped__(tmp_path)


def config(vendor: str, root: Path, base: str = "http://127.0.0.1:8899") -> dict:
    return {**bots.setup_plan(vendor, base)["config"], "team_id": "111", "channel_id": "222",
            "application_id": "333", "user_ids": ["444"], "replay_file": str(root / (vendor + ".sqlite3"))}


def payload(vendor: str, *, key: str = "one", question: str = "Highmere Stair warehousing cost?") -> dict:
    if vendor == "slack":
        return {"team_id": "111", "channel_id": "222", "user_id": "444", "api_app_id": "333",
                "trigger_id": key, "text": question, "command": "/brain",
                "response_url": "https://hooks.slack.com/commands/T/B/synthetic"}
    if vendor == "discord":
        return {"type": 2, "guild_id": "111", "channel_id": "222", "application_id": "333",
                "member": {"user": {"id": "444"}}, "id": key, "token": "synthetic-interaction-token",
                "data": {"type": 1, "name": "brain", "options": [
                    {"name": "question", "type": 3, "value": question}]}}
    return {"type": "message", "channelId": "msteams", "channelData": {
        "team": {"id": "111"}, "channel": {"id": "222"}, "tenant": {"id": "333"}},
        "from": {"id": "444"}, "timestamp": datetime.now(UTC).isoformat(), "id": key,
        "text": "<at>Smart Brain</at> " + question}


def post(base: str, vendor: str, data: dict, *, valid: bool = True, stale: bool = False,
         path: str | None = None, origin: str = "") -> tuple[int, dict]:
    raw = (urllib.parse.urlencode(data).encode() if vendor == "slack" else json.dumps(data).encode())
    stamp = str(int(time.time()) - (600 if stale else 0))
    headers = {"Content-Type": "application/x-www-form-urlencoded" if vendor == "slack" else "application/json"}
    if vendor == "slack":
        headers.update({"X-Slack-Request-Timestamp": stamp, "X-Slack-Signature": "v0=" +
                        hmac.new(KEYS[vendor].encode(), b"v0:" + stamp.encode() + b":" + raw, hashlib.sha256).hexdigest()})
    elif vendor == "teams":
        headers["Authorization"] = "HMAC " + base64.b64encode(hmac.digest(
            base64.b64decode(KEYS[vendor]), raw, "sha256")).decode()
    else:
        headers.update({"X-Signature-Timestamp": stamp,
                        "X-Signature-Ed25519": PRIVATE.sign(stamp.encode() + raw).hex()})
    if not valid:
        raw += b" "
    if origin:
        headers["Origin"] = origin
    req = urllib.request.Request(base + (path or "/" + vendor), data=raw, headers=headers)
    try:
        response = urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, json.loads(response.read())


@contextlib.contextmanager
def bridge(cfg: dict):
    server = bots.RelayServer(("127.0.0.1", 0), bots.make_handler(cfg, "api-test-token", KEYS[cfg["vendor"]]))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture()
def callbacks(monkeypatch):
    captured, done = [], threading.Event()

    class Collector(BaseHTTPRequestHandler):
        def receive(self):
            captured.append({"method": self.command, "path": self.path,
                             "authorization": self.headers.get("Authorization"),
                             "body": json.loads(self.rfile.read(int(self.headers["Content-Length"])))})
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")
            done.set()

        do_POST = receive
        do_PATCH = receive

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Collector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    original = urllib.request.build_opener
    monkeypatch.setattr(urllib.request, "_opener", original(urllib.request.ProxyHandler({})))

    class LocalCallbacks:
        def __init__(self, *args):
            self.inner = original(*args)

        def open(self, req, *args, **kwargs):
            if req.full_url.startswith(("https://hooks.slack.com/commands/", "https://discord.com/api/v10/webhooks/")):
                req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/callback",
                    data=req.data, headers=dict(req.headers), method=req.method)
            else:
                assert req.full_url.startswith("http://127.0.0.1:"), "External network forbidden"
            return self.inner.open(req, *args, **kwargs)

    monkeypatch.setattr(bots.urllib.request, "build_opener", LocalCallbacks)
    try:
        yield captured, done
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("vendor", bots.VENDORS)
@pytest.mark.parametrize("hidden", [False, True])
def test_actual_scoped_answer_and_private_vendor_reply(corpus_dir, tmp_path, upstream, callbacks, vendor, hidden):
    url, model_calls = upstream
    captured, done = callbacks
    with host(corpus_dir, tmp_path, upstream_url=url, encrypted=True, hidden=hidden) as (base, root):
        assert (root / "package_bots.py").read_bytes() == Path(bots.__file__).read_bytes()
        assert request(base, "/bots.js", token="")[0] == 200
        status, text, _ = request(base, "/connect/bots-plan?vendor=" + vendor, token="")
        assert status == 200 and json.loads(text)["config"]["vendor"] == vendor
        cfg = config(vendor, tmp_path, base)
        with bridge(cfg) as address:
            before = len(model_calls)
            assert post(address, vendor, payload(vendor), valid=False)[0] == 401
            assert len(model_calls) == before
            wrong = payload(vendor)
            if vendor == "slack":
                wrong["channel_id"] = "outside"
            elif vendor == "discord":
                wrong["channel_id"] = "outside"
            else:
                wrong["channelData"]["channel"]["id"] = "outside"
            assert post(address, vendor, wrong)[0] == 403
            assert len(model_calls) == before
            start = time.monotonic()
            status, answer = post(address, vendor, payload(vendor))
            assert status == 200
            if vendor != "teams":
                assert time.monotonic() - start < 2.5
                if vendor == "discord":
                    assert answer == {"type": 5, "data": {"flags": 64}}
                else:
                    assert answer["response_type"] == "ephemeral"
                assert done.wait(10), "Real local callback was not delivered"
                answer = captured[-1]["body"]
                assert captured[-1]["authorization"] is None
                assert captured[-1]["method"] == ("PATCH" if vendor == "discord" else "POST")
            output = answer.get("text", answer.get("content"))
            assert "Nine crowns" in output and "Sources:" in output
            assert ("names hidden by package" in output) is hidden
            assert ("highmere.txt" in output) is not hidden
            assert "ledger" not in output and "emberfall" not in output
            if vendor == "discord":
                assert answer["allowed_mentions"] == {"parse": []}
            if vendor == "teams":
                assert answer["textFormat"] == "plain"
            assert len(model_calls) == before + 1
            prompt = json.dumps(model_calls[-1]["body"])
            assert "nine crowns" in prompt and "ledger" not in prompt
            assert post(address, vendor, payload(vendor))[0] == 409
            done.clear()
            status, miss = post(address, vendor, payload(vendor, key="miss", question="zzzzzz"))
            assert status == 200
            if vendor != "teams":
                assert done.wait(10)
                miss = captured[-1]["body"]
            assert "no matching evidence" in json.dumps(miss)
            assert len(model_calls) == before + 1, "No-hit must not fabricate an LLM call"
        # A fresh bridge process/handler still denies the previously claimed event.
        with bridge(cfg) as address:
            assert post(address, vendor, payload(vendor))[0] == 409
        replay = Path(cfg["replay_file"])
        assert replay.stat().st_mode & 0o777 == 0o600
        assert b"Highmere" not in replay.read_bytes() and b"synthetic-interaction-token" not in replay.read_bytes()


@pytest.mark.parametrize("vendor", bots.VENDORS)
def test_unconfigured_llm_errors_and_miss_are_honest(corpus_dir, tmp_path, callbacks, vendor):
    captured, done = callbacks
    with host(corpus_dir, tmp_path) as (base, _), bridge(config(vendor, tmp_path, base)) as address:
        status, result = post(address, vendor, payload(vendor))
        assert status == 200
        if vendor != "teams":
            assert done.wait(10)
            result = captured[-1]["body"]
        assert bots.ERROR in json.dumps(result)
        assert "private-upstream-key" not in json.dumps(result)


@pytest.mark.parametrize("vendor", bots.VENDORS)
def test_scope_command_signature_freshness_paths_and_rate(tmp_path, callbacks, vendor):
    with bridge(config(vendor, tmp_path)) as base:
        if vendor == "discord":
            assert post(base, vendor, {"type": 1}) == (200, {"type": 1})
            assert post(base, vendor, {"type": 1}, valid=False)[0] == 401
        data = payload(vendor)
        if vendor == "teams":
            data["timestamp"] = "2001-01-01T00:00:00Z"
            assert post(base, vendor, data)[0] == 403
        else:
            assert post(base, vendor, data, stale=True)[0] == 401
        for path in ("/", "/qc/run", "/usage/report", "/v1/chat/completions", "/" + vendor + "?scope=all"):
            assert post(base, vendor, payload(vendor), path=path)[0] == 404
        assert post(base, vendor, payload(vendor), origin="https://foreign.example.com")[0] == 403
        # Invalid questions fail without consuming stable event ID.
        assert post(base, vendor, payload(vendor, question=" "))[0] == 400
        for field in ("team", "user", "application"):
            data = payload(vendor)
            if vendor == "slack":
                data[{"team": "team_id", "user": "user_id", "application": "api_app_id"}[field]] = "outside"
            elif vendor == "discord":
                if field == "user":
                    data["member"]["user"]["id"] = "outside"
                else:
                    data[{"team": "guild_id", "application": "application_id"}[field]] = "outside"
            else:
                if field == "user":
                    data["from"]["id"] = "outside"
                else:
                    data["channelData"][{"team": "team", "application": "tenant"}[field]]["id"] = "outside"
            assert post(base, vendor, data)[0] == 403
        # Rate limit authenticated processing, including duplicate events. No external callback.
        if vendor == "teams":
            data = payload(vendor)
        else:
            data = payload(vendor)
        assert post(base, vendor, data)[0] == 200
        if vendor != "teams":
            assert callbacks[1].wait(10)
        for _ in range(29):
            assert post(base, vendor, data)[0] == 409
        assert post(base, vendor, data)[0] == 429


def test_config_templates_ui_and_safety(client, tmp_path, monkeypatch):
    monkeypatch.setattr(bots.urllib.request, "urlopen", lambda *a, **k: pytest.fail("No network in setup"))
    for vendor in bots.VENDORS:
        plan = bots.setup_plan(vendor)
        assert not plan["network_checked"] and not plan["public_exposure_started"]
        if vendor == "discord":
            assert "contexts" not in plan["registration"] and "integration_types" not in plan["registration"]
            assert "/guilds/REPLACE_TEAM_ID/commands" in plan["registration_endpoint"]
        with pytest.raises(ValueError):
            bots.validate_config(plan["config"])
        response = client.get("/connect/bots-plan", params={"vendor": vendor})
        assert response.status_code == 200 and response.json() == plan
    assert client.get("/connect/bots-plan", params={"vendor": "other"}).status_code == 400
    assert client.get("/connect/bots-plan", params={"vendor": "slack", "package_url": "https://evil.example.com"}).status_code == 400
    for key, value in (("user_ids", []), ("channel_id", "*"), ("application_id", "../escape"), ("port", 8899)):
        bad = config("slack", tmp_path)
        bad[key] = value
        with pytest.raises(ValueError):
            bots.validate_config(bad)
    target = tmp_path / "link"
    target.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="symlink"):
        bots.ReplayStore(str(target))
    page = client.get("/connect").text
    assert 'id="bots-generate"' in page and 'data-endpoint="/connect/bots-plan"' in page
    assert '/static/js/bots.js?v=1' in page
    from finetune_studio.guide.registry import CONTROLS
    from finetune_studio.guide.settings_help import SETTING_HELP
    assert "connect.bots_download" in CONTROLS and SETTING_HELP["bots_vendor"]["kb"] == "package-bots"


def test_bad_callback_and_injected_options_cannot_widen_scope(tmp_path):
    cfg = config("slack", tmp_path)
    for url in ("http://127.0.0.1:1", "https://hooks.slack.com.evil.example.com/commands/T/B/s",
                "https://hooks.slack.com/commands/T/B/s?token=secret", "https://u@hooks.slack.com/commands/T"):
        data = {**payload("slack"), "response_url": url}
        with pytest.raises(ValueError):
            bots.event(cfg, urllib.parse.urlencode(data).encode())
    data = payload("discord")
    data["data"]["options"].append({"name": "scope", "type": 3, "value": "all"})
    with pytest.raises(ValueError):
        bots.event(config("discord", tmp_path), json.dumps(data).encode())
    text = bots.answer_text({"choices": [{"message": {"content": "@everyone <https://evil.example.com> test"}}],
                             "knowledge": {"sources_hidden": True, "sources": [{"filename": "PRIVATE"}]}})
    assert "@everyone" not in text and "https://" not in text and "PRIVATE" not in text


@pytest.mark.parametrize("mode", ["redirect", "malformed", "failure", "slow"])
def test_real_http_upstream_failures_are_sanitized_and_bounded(tmp_path, mode):
    calls = []

    class Backend(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            calls.append(self.path)
            self.rfile.read(int(self.headers["Content-Length"]))
            if mode == "slow":
                time.sleep(4)
            data = b'{"PRIVATE": "private-upstream-key private-document"}'
            self.send_response(307 if mode == "redirect" else 503 if mode == "failure" else 200)
            if mode == "redirect":
                self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/trap")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            with contextlib.suppress(OSError):
                self.wfile.write(data)

        def log_message(self, *args):
            pass

    backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    try:
        cfg = config("teams", tmp_path, f"http://127.0.0.1:{backend.server_port}")
        with bridge(cfg) as base:
            started = time.monotonic()
            status, result = post(base, "teams", payload("teams"))
            assert status == 200 and result["text"] == bots.ERROR
            assert time.monotonic() - started < 4.5
            assert calls == ["/v1/chat/completions"], "A redirect must not reach its destination"
            assert "PRIVATE" not in json.dumps(result) and "private-upstream-key" not in json.dumps(result)
    finally:
        backend.shutdown()
        backend.server_close()
        thread.join(timeout=5)


def test_replay_storage_failure_fails_closed(tmp_path, monkeypatch):
    import sqlite3

    with bridge(config("teams", tmp_path)) as base:
        def fail(self, key):
            raise sqlite3.OperationalError("private path")
        monkeypatch.setattr(bots.ReplayStore, "claim", fail)
        status, result = post(base, "teams", payload("teams"))
        assert status == 503 and "private path" not in json.dumps(result)


def test_output_budget_and_markup_are_bounded():
    result = {"choices": [{"message": {"content": "💡" * 4000 + "@everyone"}}],
              "knowledge": {"sources": [{"filename": "*" * 120} for _ in range(5)]}}
    out = bots.answer_text(result)
    assert len(out.encode("utf-16-le")) <= 3800 and "@everyone" not in out


def test_teams_vendor_zulu_timestamp_with_python310_parser(tmp_path, monkeypatch):
    # Export UI promises Python3.10+. That parser accepts +00:00, not a trailing Z
    # (native Z support was added in3.11). Exercise real signed HTTP at this boundary.
    real_datetime = bots.datetime

    class Python310Datetime:
        @staticmethod
        def fromisoformat(value):
            if value.endswith("Z"):
                raise ValueError("Invalid isoformat string on Python3.10")
            return real_datetime.fromisoformat(value)

    monkeypatch.setattr(bots, "datetime", Python310Datetime)
    data = payload("teams")
    data["timestamp"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    with bridge(config("teams", tmp_path)) as base:
        status, result = post(base, "teams", data)
        assert status == 200 and result["type"] == "message"
