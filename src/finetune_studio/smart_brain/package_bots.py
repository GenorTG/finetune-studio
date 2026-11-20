"""Opt-in Linux bot bridges. One fixed package/team/channel/user audience per process.

Slack slash commands, Teams outgoing webhooks, Discord HTTP interactions. No account
setup or publication. Tokens live only in environment; replay storage holds hashes only.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import importlib.util
import json
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

try:
    from .package_actions import RelayServer, _NoRedirect, package_address
except ImportError:
    _spec = importlib.util.spec_from_file_location("_bot_actions", Path(__file__).with_name("package_actions.py"))
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    RelayServer, _NoRedirect, package_address = _module.RelayServer, _module._NoRedirect, _module.package_address

VENDORS = ("slack", "teams", "discord")
SAFE_ID = re.compile(r"[A-Za-z0-9:_@.=-]{1,200}")
ERROR = "Answer unavailable. Ask the package operator to check its LLM/configuration; no answer was generated."


def setup_plan(vendor: str, package_url: str = "http://127.0.0.1:8899", port: int = 8901) -> dict[str, Any]:
    """Credential-free templates; no network, persistence, or execution."""
    if vendor not in VENDORS:
        raise ValueError("Choose slack, teams or discord.")
    upstream = package_address(package_url)
    if type(port) is not int or not 1024 <= port <= 65535 or port == urlsplit(upstream).port:
        raise ValueError("Choose a separate bridge port from 1024 to 65535.")
    config = {"vendor": vendor, "package_url": upstream, "port": port,
              "team_id": "REPLACE_TEAM_ID", "channel_id": "REPLACE_CHANNEL_ID",
              "user_ids": ["REPLACE_ALLOWED_USER_ID"], "application_id": "REPLACE_APPLICATION_ID",
              "replay_file": "./bot-replay.sqlite3"}
    registration = {
        "slack": {"display_information": {"name": "Smart Brain"},
                  "features": {"slash_commands": [{"command": "/brain",
                    "description": "Ask scoped knowledge", "url": "https://REPLACE_HOST/slack",
                    "usage_hint": "question", "should_escape": False}]},
                  "oauth_config": {"scopes": {"bot": ["commands"]}}},
        "discord": {"name": "brain", "description": "Ask scoped knowledge", "type": 1,
                    "contexts": [0], "integration_types": [0],
                    "options": [{"name": "question", "description": "Question", "type": 3,
                                 "required": True, "min_length": 1, "max_length": 4000}]},
        "teams": {"name": "Smart Brain", "callback_url": "https://REPLACE_HOST/teams",
                  "interface": "Team-scoped outgoing webhook; public channels only; replies visible to channel"},
    }[vendor]
    secret = {"slack": "SLACK_SIGNING_SECRET", "teams": "TEAMS_SIGNING_SECRET",
              "discord": "DISCORD_PUBLIC_KEY"}[vendor]
    return {"status": "configuration_only", "network_checked": False, "public_exposure_started": False,
            "config": config, "registration": registration, "secret_environment": ["RAG_AUTH_TOKEN", secret],
            "command": ".venv/bin/python package_bots.py --config bot-config.json",
            "listener": f"http://127.0.0.1:{port}/{vendor}"}


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    expected = set(setup_plan("slack")["config"])
    if set(config) != expected:
        raise ValueError("Use the generated config keys without additional overrides.")
    setup_plan(config["vendor"], config["package_url"], config["port"])
    for field in ("team_id", "channel_id", "application_id"):
        if not isinstance(config[field], str) or not SAFE_ID.fullmatch(config[field]) or "REPLACE" in config[field]:
            raise ValueError(f"Set a concrete {field}; no wildcard audiences.")
    users = config["user_ids"]
    if (not isinstance(users, list) or not 1 <= len(users) <= 100
            or any(not isinstance(u, str) or not SAFE_ID.fullmatch(u) or "REPLACE" in u for u in users)):
        raise ValueError("Set 1–100 explicit allowed user IDs; no wildcard audience.")
    if not isinstance(config["replay_file"], str) or not config["replay_file"]:
        raise ValueError("Set a private persistent replay_file path.")
    return config


class ReplayStore:
    """Persist only SHA256 digests and expiration, bounded and fail-closed across restarts."""

    def __init__(self, path: str):
        target = Path(path)
        if target.is_symlink():
            raise ValueError("Replay file must not be a symlink.")
        fd = os.open(target, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        target.chmod(0o600)
        self.path = str(target)
        self.lock = threading.Lock()
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS seen (digest TEXT PRIMARY KEY, expires REAL NOT NULL)")

    def claim(self, key: str) -> bool:
        digest = hashlib.sha256(key.encode()).hexdigest()
        with self.lock, sqlite3.connect(self.path, timeout=0.1) as db:
            db.execute("DELETE FROM seen WHERE expires < ?", (time.time(),))
            if db.execute("SELECT count(*) FROM seen").fetchone()[0] >= 10000:
                return False
            try:
                db.execute("INSERT INTO seen VALUES (?, ?)", (digest, time.time() + 86400))
            except sqlite3.IntegrityError:
                return False
        return True


def fresh(timestamp: str) -> bool:
    try:
        value = float(timestamp)
    except (ValueError, TypeError):
        return False
    return abs(time.time() - value) <= 300


def signed(vendor: str, raw: bytes, headers: Any, secret: str) -> bool:
    """Authenticate raw bytes, never deserialized/re-encoded payloads."""
    try:
        if vendor == "slack":
            stamp = headers.get("X-Slack-Request-Timestamp", "")
            expected = "v0=" + hmac.new(secret.encode(), b"v0:" + stamp.encode() + b":" + raw, hashlib.sha256).hexdigest()
            return fresh(stamp) and hmac.compare_digest(expected, headers.get("X-Slack-Signature", ""))
        if vendor == "teams":
            key = base64.b64decode(secret, validate=True)
            expected = "HMAC " + base64.b64encode(hmac.digest(key, raw, "sha256")).decode()
            return hmac.compare_digest(expected, headers.get("Authorization", ""))
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        stamp = headers.get("X-Signature-Timestamp", "")
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(secret)).verify(
            bytes.fromhex(headers.get("X-Signature-Ed25519", "")), stamp.encode() + raw)
        return fresh(stamp)
    except Exception:  # noqa: BLE001 - untrusted crypto inputs fail closed; never log keys
        return False


def event(config: dict, raw: bytes) -> tuple[str, str, str, str]:
    """Return question, deduplication key, callback URL and callback method; deny scope overrides."""
    vendor = config["vendor"]
    if vendor == "slack":
        data = parse_qs(raw.decode(), keep_blank_values=True, strict_parsing=True)
        if any(len(v) != 1 for v in data.values()):
            raise ValueError("Duplicate fields.")
        data = {k: v[0] for k, v in data.items()}
        team, channel, user = data.get("team_id"), data.get("channel_id"), data.get("user_id")
        if data.get("command") != "/brain" or data.get("api_app_id") != config["application_id"]:
            raise PermissionError
        question = data.get("text", "")
        key = data.get("trigger_id", "")
        callback = data.get("response_url", "")
        url = urlsplit(callback)
        if (url.scheme != "https" or url.netloc != "hooks.slack.com"
                or not re.fullmatch(r"/commands/[A-Za-z0-9/_-]+", url.path)
                or url.query or url.fragment):
            raise ValueError("Invalid Slack response URL.")
        method = "POST"
    else:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Object required.")
        if vendor == "discord":
            team, channel = data.get("guild_id"), data.get("channel_id")
            user = data.get("member", {}).get("user", {}).get("id")
            command = data.get("data", {})
            if (data.get("type") != 2 or command.get("name") != "brain"
                    or command.get("type") != 1 or data.get("application_id") != config["application_id"]):
                raise PermissionError
            options = command.get("options", [])
            if (len(options) != 1 or options[0].get("name") != "question"
                    or options[0].get("type") != 3):
                raise ValueError("Only question is supported.")
            question, key = options[0].get("value"), data.get("id", "")
            token = data.get("token", "")
            if not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,512}", token):
                raise ValueError("Invalid interaction token.")
            callback = f"https://discord.com/api/v10/webhooks/{config['application_id']}/{token}/messages/@original"
            method = "PATCH"
        else:
            channel_data = data.get("channelData", {})
            team = channel_data.get("team", {}).get("id")
            channel = channel_data.get("channel", {}).get("id")
            if (data.get("type") != "message" or data.get("channelId") != "msteams"
                    or channel_data.get("tenant", {}).get("id") != config["application_id"]):
                raise PermissionError
            user = data.get("from", {}).get("id")
            stamp = datetime.fromisoformat(data.get("timestamp", ""))
            if stamp.tzinfo is None or not fresh(str(stamp.timestamp())):
                raise PermissionError
            question = re.sub(r"<at>[^<]{1,200}</at>", "", data.get("text", "")).strip()
            key = data.get("id", "")
            callback, method = "", ""
    if team != config["team_id"] or channel != config["channel_id"] or user not in config["user_ids"]:
        raise PermissionError
    if not isinstance(question, str) or not question.strip() or len(question) > 4000:
        raise ValueError("Question must contain 1–4000 characters.")
    if not isinstance(key, str) or not SAFE_ID.fullmatch(key):
        raise ValueError("Missing stable event ID.")
    return question.strip(), vendor + ":" + key, callback, method


def answer_text(result: dict) -> str:
    """Append package-projected citations only; no links, original paths, images or mentions."""
    content = result["choices"][0]["message"]["content"]
    knowledge = result["knowledge"]
    if not isinstance(content, str) or not isinstance(knowledge, dict):
        raise TypeError("Invalid answer.")
    sources = knowledge.get("sources", [])
    if not isinstance(sources, list):
        raise TypeError("Invalid citations.")
    hidden = knowledge.get("sources_hidden", False) is True
    names = [] if hidden else [s.get("filename", "")[:120] for s in sources[:5] if isinstance(s, dict)]
    suffix = "\nSources: " + (", ".join(n for n in names if n) or ("names hidden by package" if hidden else "no matching evidence"))
    # Plain text plus escaping prevents chat markup, mentions, links and model-produced citation URLs.
    text = re.sub(r"https?://\S+", "[link omitted]", content[:1400] + suffix)
    plain = text.replace("@", "＠").replace("<", "‹").replace(">", "›").replace("&", "＆")
    plain = "".join("\\" + char if char in "\\*_`~|[]" else char for char in plain)
    # Discord's 2000-character limit: also bound UTF-16 units for astral text.
    return plain.encode("utf-16-le")[:3800].decode("utf-16-le", errors="ignore")


def make_handler(config: dict, package_token: str, secret: str) -> type[BaseHTTPRequestHandler]:
    config = validate_config(config.copy())
    vendor = config["vendor"]
    if (not 1 <= len(package_token) <= 512 or any(ord(c) < 33 or ord(c) > 126 for c in package_token)
            or not secret or package_token == secret):
        raise ValueError("Set separate package and vendor credentials privately.")
    try:
        if vendor == "discord":
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PublicKey,
            )
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(secret))
        elif vendor == "teams":
            if len(base64.b64decode(secret, validate=True)) < 16:
                raise ValueError
        elif not 16 <= len(secret) <= 512:
            raise ValueError
    except ValueError:
        raise ValueError("Invalid vendor verification key.") from None
    replay = ReplayStore(config["replay_file"])
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    slots = threading.BoundedSemaphore(2)
    lock = threading.Lock()
    window: list[float] = []

    def call(url: str, body: dict, method: str = "POST", timeout: float = 120,
             authorization: str = "", decode: bool = True) -> dict:
        headers = {"Content-Type": "application/json"}
        if authorization:
            headers["Authorization"] = authorization
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method=method)
        with opener.open(req, timeout=timeout) as response:
            raw = response.read(200001)
            if len(raw) > 200000:
                raise ValueError("Response too large.")
            return json.loads(raw) if raw and decode else {}

    def answer(question: str) -> str:
        try:
            result = call(config["package_url"] + "/v1/chat/completions",
                          {"model": "smart-brain", "messages": [{"role": "user", "content": question}],
                           "stream": False, "max_tokens": 400},
                          timeout=3.5 if vendor == "teams" else 120,
                          authorization="Bearer " + package_token)
            return answer_text(result)
        except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
            return ERROR

    def output(text: str) -> dict:
        if vendor == "slack":
            return {"response_type": "ephemeral", "text": text, "mrkdwn": False,
                    "unfurl_links": False, "unfurl_media": False}
        if vendor == "discord":
            return {"content": text, "allowed_mentions": {"parse": []}}
        return {"type": "message", "text": text, "textFormat": "plain"}

    def worker(question: str, callback: str, method: str) -> None:
        try:
            call(callback, output(answer(question)), method, timeout=10, decode=False)
        except (OSError, ValueError, TypeError):
            print("Bot reply delivery failed; not retried. No question or credential logged.", flush=True)
        finally:
            slots.release()

    class Handler(BaseHTTPRequestHandler):
        server_version = "SmartBrainBots/1"
        sys_version = ""

        def log_message(self, fmt: str, *args: Any) -> None:
            pass

        def reply(self, value: dict, code: int = 200) -> None:
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if code == 429:
                self.send_header("Retry-After", "60")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self.reply({"error": "not found"}, 404)

        def do_POST(self) -> None:
            if self.path != "/" + vendor:
                self.reply({"error": "not found"}, 404)
                return
            if self.headers.get("Origin"):
                self.reply({"error": "Browser requests are not supported."}, 403)
                return
            try:
                lengths = self.headers.get_all("Content-Length", [])
                if len(lengths) != 1 or self.headers.get("Transfer-Encoding"):
                    raise ValueError
                length = int(lengths[0])
                if not 1 <= length <= 24000:
                    raise ValueError
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError
            except (ValueError, OSError):
                self.reply({"error": "Invalid body framing or size."}, 400)
                return
            if not signed(vendor, raw, self.headers, secret):
                self.reply({"error": "Invalid or stale vendor signature."}, 401)
                return
            try:
                if vendor == "discord" and json.loads(raw).get("type") == 1:
                    self.reply({"type": 1})
                    return
                question, key, callback, method = event(config, raw)
            except PermissionError:
                self.reply({"error": "Event outside configured audience or command."}, 403)
                return
            except (ValueError, TypeError, AttributeError, KeyError, IndexError):
                self.reply({"error": "Invalid command payload."}, 400)
                return
            with lock:
                now = time.monotonic()
                window[:] = [at for at in window if at > now - 60]
                limited = len(window) >= 30
                if not limited:
                    window.append(now)
            if limited or not slots.acquire(blocking=False):
                self.reply({"error": "Bridge request/concurrency limit."}, 429)
                return
            handed_off = False
            try:
                if not replay.claim(key):
                    self.reply({"error": "Duplicate event or replay capacity reached; not reprocessed."}, 409)
                    return
                if vendor == "teams":
                    self.reply(output(answer(question)))
                else:
                    ack = ({"response_type": "ephemeral", "text": "Searching scoped knowledge…"}
                           if vendor == "slack" else {"type": 5, "data": {"flags": 64}})
                    self.reply(ack)
                    threading.Thread(target=worker, args=(question, callback, method), daemon=True).start()
                    handed_off = True
                    return
            except (sqlite3.Error, OSError):
                self.reply({"error": "Replay storage unavailable; event not processed."}, 503)
            finally:
                # Async worker owns its slot after a successful acknowledgment.
                if not handed_off:
                    slots.release()

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Loopback bot bridge; starts no tunnel or vendor registration.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        config = validate_config(json.loads(Path(args.config).read_text()))
        secret_name = {"slack": "SLACK_SIGNING_SECRET", "teams": "TEAMS_SIGNING_SECRET",
                       "discord": "DISCORD_PUBLIC_KEY"}[config["vendor"]]
        handler = make_handler(config, os.environ.get("RAG_AUTH_TOKEN", ""), os.environ.get(secret_name, ""))
        server = RelayServer(("127.0.0.1", config["port"]), handler)
    except (ValueError, OSError, TypeError, KeyError) as error:
        parser.error(str(error))
    print("Bot bridge listening on loopback only; no account or public endpoint created. Ctrl+C stops it.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
