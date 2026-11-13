"""A project always has a name (walkthrough 2026-11-13 M9: Project settings saved a blank name, no message)."""
from __future__ import annotations

import pytest


def test_blank_or_bad_names_are_refused_with_a_reason(client) -> None:
    pid = client.post("/api/projects", json={"name": "agent-test-name-a"}).json()["id"]
    for bad, why in (("", "cannot be blank"), ("   ", "cannot be blank"), ("x" * 121, "at most 120"),
                     (None, "give the project a name"), (12, "give the project a name"),
                     ("a\x07b", "control characters")):
        r = client.patch(f"/api/projects/{pid}", json={"name": bad})
        assert r.status_code == 422 and why in r.json()["detail"], (bad, r.text)
    assert client.get(f"/api/projects/{pid}").json()["name"] == "agent-test-name-a"
    ok = client.patch(f"/api/projects/{pid}", json={"name": "  agent-test-name   renamed  "})
    assert ok.status_code == 200 and ok.json()["name"] == "agent-test-name renamed"
    client.post("/api/projects", json={"name": "agent-test-name-b"})
    taken = client.patch(f"/api/projects/{pid}", json={"name": "AGENT-TEST-NAME-B"})
    assert taken.status_code == 409 and taken.json()["code"] == "duplicate_name"
    assert client.patch(f"/api/projects/{pid}", json={"description": "only the description"}).status_code == 200
    assert client.post("/api/projects", json={"name": "  "}).status_code == 422
    page = client.get(f"/projects/{pid}/settings").text
    assert 'name="name"' in page and 'required maxlength="120"' in page


def test_cli_refuses_blank_names(client, capsys) -> None:
    from argparse import Namespace

    from finetune_studio.cli.commands import projects as cli

    pid = client.post("/api/projects", json={"name": "agent-test-cli-blank"}).json()["id"]
    for ns in (Namespace(projects_action="rename", project=pid, name="  ", json=False),
               Namespace(projects_action="create", name="   ", description="", base_model="", system_prompt="",
                         allow_duplicate=False, json=False)):
        with pytest.raises(SystemExit) as stop:
            cli.cmd_projects(ns)
        assert stop.value.code == 1
        assert "cannot be blank" in capsys.readouterr().err


def test_cli_rename(client, monkeypatch) -> None:
    from argparse import Namespace

    from finetune_studio.cli.commands import projects as cli

    pid = client.post("/api/projects", json={"name": "agent-test-cli-rename"}).json()["id"]
    cli.cmd_projects(Namespace(projects_action="rename", project=pid, name="agent-test-cli-renamed", json=False))
    from finetune_studio import db
    assert db.get_project(pid)["name"] == "agent-test-cli-renamed"
