"""Training page "Start from a use case" card (#uc-card, static/js/usecases.js) against the use-case API.

Pins the contract between the page and the server: the card renders with every control the script and the Guide
use, every endpoint the script calls exists with that method, and the plan's overrides map onto real form fields.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from finetune_studio.usecases.templates import TEMPLATES, recommend_settings

ROOT = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui"
JS = (ROOT / "static" / "js" / "usecases.js").read_text(encoding="utf-8")
PAGE = (ROOT / "templates" / "project_training.html").read_text(encoding="utf-8")
CARD_IDS = ("uc-card", "uc-list", "uc-new", "uc-template", "uc-name", "uc-template-help", "uc-file",
            "uc-instruction", "uc-labels", "uc-schema", "uc-style-guide", "uc-teacher", "uc-teacher-name",
            "uc-build-btn", "uc-build-status", "uc-build-result", "uc-report")


@pytest.fixture
def project(client: TestClient) -> str:
    return client.post("/api/projects", json={"name": "uc-ui", "base_model": "x/t"}).json()["id"]


def test_training_page_renders_the_card_and_mounts_the_script(client: TestClient, project: str) -> None:
    html = client.get(f"/projects/{project}/training").text
    for el_id in CARD_IDS:
        assert f'id="{el_id}"' in html, el_id
    assert f'data-pid="{project}"' in html
    assert '<script src="/static/js/usecases.js?v=' in html
    # SPA re-entry does not re-run a src script, so the page must call mount() inline too.
    assert 'window.ftsUsecases && window.ftsUsecases.mount(document.getElementById("uc-card"))' in html
    assert client.get("/static/js/usecases.js").status_code == 200


def test_script_reads_every_control_the_template_has() -> None:
    for el_id in CARD_IDS[1:]:
        assert f"'{el_id}'" in JS or f"#{el_id}" in JS, el_id
    for opt in ("instruction", "labels", "schema", "style_guide", "teacher_provider_id", "teacher"):
        assert f'data-uc-opt="{opt}"' in PAGE, opt


def _paths(app) -> dict[str, set[str]]:
    return {p: {m.upper() for m in ops} for p, ops in app.openapi()["paths"].items()}


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/usecases/templates"),
    ("GET", "/api/projects/{pid}/usecases"),
    ("POST", "/api/projects/{pid}/usecases/upload"),
    ("GET", "/api/projects/{pid}/usecases/teacher/progress"),
    ("DELETE", "/api/projects/{pid}/usecases/{uid}"),
    ("POST", "/api/projects/{pid}/usecases/{uid}/evaluate"),
    ("GET", "/api/projects/{pid}/usecases/{uid}/evaluations/{gid}"),
    ("POST", "/api/projects/{pid}/usecases/{uid}/evaluations/{gid}/ab"),
    ("GET", "/api/projects/{pid}/runs"),
    ("GET", "/api/providers"),
])
def test_every_endpoint_the_script_calls_exists(method: str, path: str) -> None:
    from finetune_studio.webui.app import app

    assert method in _paths(app).get(path, set()), f"{method} {path} is not served"
    tail = re.split(r"\{[^}]+\}", path)[-1]
    # the script builds project URLs from `${base}` (= /api/projects/<pid>/usecases) plus a suffix
    probe = tail.split("/usecases", 1)[-1] if "/api/projects/" in path and "/usecases" in tail else tail
    assert probe.strip("/") == "" or probe in JS, f"{path} is not called by usecases.js"


def test_plan_overrides_fill_real_form_fields() -> None:
    form_fields = set(re.findall(r"FORM_FIELDS = \[([^\]]+)\]", JS)[0].replace("'", "").replace("\n", "").split(","))
    form_fields = {f.strip() for f in form_fields if f.strip()}
    plan = recommend_settings(TEMPLATES["classification"], [[{"role": "user", "content": "x"},
                                                             {"role": "assistant", "content": "y"}]] * 80)
    assert set(plan.overrides) == form_fields, "usecases.js must apply every override the plan proposes"
    for name in form_fields:
        assert f'name="{name}"' in PAGE, f"the training form has no {name} field"
    assert plan.system_prompt_mode in re.findall(r'<option value="(\w+)"', PAGE)
    assert f'name="training_mode" value="{plan.training_mode}"' in PAGE


def test_form_hook_routes_the_pick_through_the_normal_dataset_path() -> None:
    hook = PAGE[PAGE.index("window.ftsTraining = {"):PAGE.index("// Deep-link from Data Prep")]
    assert "ensureDatasetSelect()" in hook and "onDatasetPicked()" in hook
    assert "window.ftsTraining.selectDataset(" in JS


def test_build_response_links_the_usecase_deep_link(client: TestClient, project: str) -> None:
    rows = [{"input": f"text number {i} about {'billing' if i % 2 else 'shipping'}",
             "label": "billing" if i % 2 else "shipping"} for i in range(90)]
    r = client.post(f"/api/projects/{project}/usecases", json={"template": "classification", "rows": rows})
    assert r.status_code == 201, r.text
    m = r.json()
    assert m["train_url"] == f"/projects/{project}/training?usecase={m['id']}"
    assert "get('usecase')" in JS  # the card reads that parameter and fills the form


def test_guide_knows_the_card() -> None:
    from finetune_studio.guide import registry
    from finetune_studio.guide.kb import load_kb

    entry = load_kb()["use-cases"]
    assert entry.page == "training"
    assert {"#uc-card", "#uc-build-btn", "#uc-report"} <= set(entry.controls)
    assert registry.CONTROLS["training.usecases"].selector == "#uc-card"


def test_usecase_dataset_check_does_not_flag_short_labels(client: TestClient, project: str) -> None:
    """A classification row's answer is one label by design; the Training page's dataset check must not call that
    a problem for a use-case dataset, and still does for an ordinary one."""
    from finetune_studio.data.dataset_health import check_dataset

    rows = [{"input": f"ticket {i} about {'refunds' if i % 2 else 'parcels'}", "label": "finance" if i % 2 else "logistics"}
            for i in range(90)]
    m = client.post(f"/api/projects/{project}/usecases", json={"template": "classification", "rows": rows}).json()
    r = client.get(f"/api/projects/{project}/datasets/{m['dataset']['id']}/health")
    assert r.status_code == 200, r.text
    assert "short_answers" not in {i["code"] for i in r.json()["issues"]}
    plain = check_dataset(m["dataset"]["path"])
    assert "short_answers" in {i["code"] for i in plain["issues"]}
