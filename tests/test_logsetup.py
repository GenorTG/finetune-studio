"""Persistent, redacted logs (2026-10-15 review: no logging was configured; INFO vanished, the journal kept ~100 lines)."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from finetune_studio import logsetup

SECRETS = [
    "hunter2-hunter2", "sk-abcdefghijklmnop1234", "hf_" + "A" * 34, "ghp_" + "b" * 36,
    "fts_" + "c" * 43, "s3cr3t-token-value", "my-passphrase-123", "cookie-value-xyz",
]


@pytest.mark.parametrize("line", [
    'login body {"password": "hunter2-hunter2", "next": "/"}',
    "provider cfg {'api_key': 'sk-abcdefghijklmnop1234', 'base_url': 'https://x'}",
    "GET /api/x?token=s3cr3t-token-value&page=2",
    "export started passphrase=my-passphrase-123 out=/tmp/x",
    "headers Authorization: Bearer fts_" + "c" * 43,
    "Cookie: fts_session=cookie-value-xyz",
    "the provider said: invalid key hf_" + "A" * 34,
    "push failed for ghp_" + "b" * 36,
])
def test_redact_removes_the_secret_and_keeps_the_rest(line: str) -> None:
    out = logsetup.redact(line)
    assert not any(s in out for s in SECRETS), out
    assert "***" in out


def test_redact_leaves_ordinary_text_alone() -> None:
    line = "trained 804 steps, loss 0.18, tokens=2048 per batch, session 3 of 5"
    assert logsetup.redact("trained 804 steps, loss 0.18") == "trained 804 steps, loss 0.18"
    assert "804 steps" in logsetup.redact(line)


@pytest.fixture
def file_logging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("FTS_ROOT", str(tmp_path))
    monkeypatch.delenv("FTS_LOG_TO_FILES", raising=False)
    root = logging.getLogger()
    before, level = list(root.handlers), root.level
    yield tmp_path / "logs"
    for h in list(root.handlers):
        if h not in before:
            root.removeHandler(h)
            h.close()
    root.setLevel(level)
    for h in list(logging.getLogger("fts.access").handlers):
        logging.getLogger("fts.access").removeHandler(h)
        h.close()


def test_setup_writes_info_to_the_component_file_with_secrets_redacted(file_logging: Path) -> None:
    path = logsetup.setup("web")
    assert path == file_logging / "web.log"
    log = logging.getLogger("finetune_studio.test")
    token = logsetup.request_id.set("abc123")
    try:
        log.info("helper call with api_key=sk-abcdefghijklmnop1234 done")
        try:
            raise ValueError('upstream refused {"password": "hunter2-hunter2"}')
        except ValueError:
            log.exception("call failed")
    finally:
        logsetup.request_id.reset(token)
    for h in logging.getLogger().handlers:
        h.flush()
    text = path.read_text()
    assert "helper call with api_key=*** done" in text and " web abc123 finetune_studio.test:" in text
    assert "ValueError" in text and "hunter2" not in text and "sk-abcdef" not in text
    assert logsetup.setup("web") == path
    assert sum(getattr(h, "_fts_managed", False) for h in logging.getLogger().handlers) == 2   # idempotent


def test_tests_never_write_into_a_live_log_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FTS_ROOT", str(tmp_path))
    monkeypatch.setenv("FTS_LOG_TO_FILES", "0")
    logsetup.setup("web")
    assert not (tmp_path / "logs").exists()


def test_every_request_gets_an_access_line_and_a_request_id(file_logging: Path) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from finetune_studio.webui.access_log import AccessLogMiddleware

    app = FastAPI()

    @app.get("/x")
    def x() -> dict:
        return {"ok": 1}

    app.add_middleware(AccessLogMiddleware)
    r = TestClient(app).get("/x?token=s3cr3t-token-value&q=1")
    assert r.status_code == 200 and len(r.headers["x-request-id"]) == 12
    for h in logging.getLogger("fts.access").handlers:
        h.flush()
    line = (file_logging / "web-access.log").read_text().strip().splitlines()[-1]
    assert "GET /x?token=***&q=1 200" in line and r.headers["x-request-id"] in line
