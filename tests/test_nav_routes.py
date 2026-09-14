"""Tests for top-nav inference → chat remap (QABUG-004 regression).

QABUG-004 was: under a project context the tools "inference" nav item
could resolve to /projects/{pid}/inference (404). The real chat-against-
model surface is /projects/{pid}/chat.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jinja2 import Environment, FileSystemLoader, select_autoescape

from finetune_studio import db
from finetune_studio.config import settings
from finetune_studio.webui.app import app

_TEMPLATES = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "templates"
)


@pytest.fixture
def client_and_db(tmp_path, monkeypatch):
    db_path = tmp_path / "fts_test.db"
    monkeypatch.setattr(settings, "db_path", str(db_path))
    db.init_db()
    return TestClient(app), db_path


def test_inference_nav_link_points_to_chat() -> None:
    """With a stub pid, tools→inference must href to /projects/<pid>/chat."""
    pid = "deadbeefdeadbeefdeadbeefdeadbeef"
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    child = env.from_string(
        '{% extends "base.html" %}'
        "{% block content %}nav-probe{% endblock %}"
    )
    html = child.render(
        pid=pid,
        project={"name": "nav-probe", "id": pid},
        app_version="0.0-test",
        request=None,
    )
    expected = f"/projects/{pid}/chat"
    assert expected in html
    assert f"/projects/{pid}/inference" not in html
    # The tools inference tab specifically (data-tab="inference").
    assert 'data-tab="inference"' in html
    # Extract the inference anchor href.
    marker = 'data-tab="inference"'
    idx = html.find(marker)
    assert idx != -1
    # Walk backwards to the opening <a ... href="...">
    open_a = html.rfind("<a ", 0, idx)
    assert open_a != -1
    snippet = html[open_a:idx]
    assert f'href="{expected}"' in snippet


def test_projects_pid_chat_route_returns_200(client_and_db) -> None:
    client, _db_path = client_and_db
    r = client.post(
        "/api/projects",
        json={"name": f"nav-{uuid.uuid4().hex[:6]}", "base_model": "x/test"},
    )
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    page = client.get(f"/projects/{pid}/chat")
    assert page.status_code == 200, page.text
