"""Walkthrough 2026-11-13: the package chat page works without an answer model and sends on Enter (m6), cites only
the files the answer names (p4); a busy port is one line (m4); a declined studio answer lists no "sources" (m13);
connection errors read as sentences (m24); the public-bot hint says what the server still shows (m2)."""
from __future__ import annotations

import errno
from pathlib import Path

import requests

from finetune_studio.models.providers import plain_network_error
from finetune_studio.smart_brain import standalone_server as ss

TEMPLATES = Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates"


def test_chat_page_search_only_enter_and_cited_sources() -> None:
    js = ss._CHAT_JS
    assert "search: { label: 'No answer model: show the matching passages'" in js
    assert "if (k === 'search') {" in js and "fetch('/search', { method: 'POST'" in js
    assert "e.key === 'Enter' && !e.shiftKey" in js and "$('ask').requestSubmit()" in js
    assert "const cited = answer ? list.filter((s) => s.filename && answer.includes(s.filename))" in js
    assert "hideNames ? `Passage ${i + 1}`" in js


def test_a_busy_port_is_one_line(monkeypatch, capsys, tmp_path: Path) -> None:
    def busy(corpus, cfg):
        raise OSError(errno.EADDRINUSE, "Address already in use")

    class FakeCorpus:
        def __init__(self, *a, **k) -> None:
            pass

    monkeypatch.setattr(ss, "serve_http", busy)
    monkeypatch.setattr(ss, "Corpus", FakeCorpus)
    (tmp_path / "manifest.json").write_text("{}")
    code = ss.main(["--http", "8899", "--corpus", str(tmp_path)])
    err = capsys.readouterr().err
    assert code == 2 and "port 8899 is in use" in err and "--port 8900" in err and "Traceback" not in err


def test_network_errors_read_as_sentences() -> None:
    url = "http://127.0.0.1:9/v1/chat/completions"
    refused = requests.ConnectionError("HTTPConnectionPool: Max retries exceeded ([Errno 111] Connection refused)")
    assert plain_network_error(refused, url).startswith(f"could not connect to {url}: connection refused")
    assert "in time" in plain_network_error(requests.Timeout("read timed out"), url)
    dns = requests.ConnectionError("Failed to resolve 'nope.invalid' (Name or service not known)")
    assert plain_network_error(dns, url).startswith("could not find the host")


def test_studio_chat_and_export_copy() -> None:
    brain = (TEMPLATES / "brain.html").read_text(encoding="utf-8")
    assert "const DECLINE_RE =" in brain and "none answers this</summary>" in brain
    export = (TEMPLATES / "export_models.html").read_text(encoding="utf-8")
    assert "The server itself still lists every file name in the package" in export
    settings = (TEMPLATES / "settings.html").read_text(encoding="utf-8")
    assert "/static/js/settings.js?v=26" in settings
