"""The wizard's Step 6 drives every project-version endpoint: pin (with "builds on"), list, detail, lineage, delete."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db

TEMPLATE = Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates/project_wizard.html"


@pytest.fixture()
def client(temp_db):
    from finetune_studio.webui.app import app
    return TestClient(app)


def test_wizard_template_calls_every_version_endpoint() -> None:
    html = TEMPLATE.read_text(encoding="utf-8")
    for needle in (
        'id="wiz-version-parent"', 'id="flow-version-parent"', 'id="wiz-versions-table"', 'id="wiz-version-detail"',
        "body.parent_version_id = parentSel.value",          # pin sends the picked parent
        '"/api/projects/" + PID + "/versions"',               # list + pin
        'base + "/lineage"',                                  # lineage walk
        'window.api.del("/api/projects/" + PID + "/versions/"',  # delete
        "sel.value = picked ? keep : (vs.length ? vs[0].id : \"\")",  # default: the newest version
        "delete s.dataset.userPicked",                        # a pin resets the pick, so the next pin continues
    ):
        assert needle in html, needle


def test_pin_with_parent_then_lineage_detail_and_delete(client) -> None:
    pid = db.create_project("Versions UI")["id"]
    v1 = client.post(f"/api/projects/{pid}/versions", json={"label": "first"}).json()
    v2 = client.post(f"/api/projects/{pid}/versions",
                     json={"label": "second", "parent_version_id": v1["id"]}).json()
    assert v2["parent_version_id"] == v1["id"]

    listed = client.get(f"/api/projects/{pid}/versions").json()["versions"]
    assert [v["version_number"] for v in listed] == [2, 1]          # newest first: the pickers default to v2

    chain = client.get(f"/api/projects/{pid}/versions/{v2['id']}/lineage").json()["lineage"]
    assert [c["label"] for c in chain] == ["first", "second"]

    detail = client.get(f"/api/projects/{pid}/versions/{v2['id']}").json()
    assert isinstance(detail["manifest"], dict)
    assert {"datasets", "training_runs"} <= set(detail["manifest"])  # what the Contents column and Lineage panel read

    assert client.delete(f"/api/projects/{pid}/versions/{v2['id']}").json() == {"ok": True}
    assert client.get(f"/api/projects/{pid}/versions/{v2['id']}").status_code == 404


def test_pin_refuses_a_parent_from_another_project(client) -> None:
    a = db.create_project("A")["id"]
    b = db.create_project("B")["id"]
    foreign = client.post(f"/api/projects/{a}/versions", json={"label": "a1"}).json()
    r = client.post(f"/api/projects/{b}/versions", json={"label": "b1", "parent_version_id": foreign["id"]})
    assert r.status_code == 404
