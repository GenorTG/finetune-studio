"""The Training page says plainly that eval loss is not a recall signal on fact Q&A data.

The trainer's held-out rows often hold facts no training row teaches, so eval loss rises while recall keeps
improving, and early stopping on the best eval loss can keep a worse model. The note shows for SFT routes (form and
run detail) and stays away from the preference routes, which report their own held-out accuracy and margin.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_TEMPLATE = Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates/project_training.html"
_PHRASE = "a rising eval loss is not a sign that recall is getting worse"


def _project(client) -> str:
    r = client.post("/api/projects", json={"name": "eval loss note"})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def _run(pid: str, mode: str | None, metrics: dict) -> str:
    from finetune_studio import db

    settings = {"training_mode": mode} if mode else {}
    run = db.create_run(project_id=pid, name=f"run-{mode}", base_model="/m", data_path="/d", settings_obj=settings)
    db.update_run(run["id"], status="done", metrics=metrics, final_loss=0.5)
    return run["id"]


def _note(html: str) -> bool:
    return 'id="run-eval-loss-note"' in html and _PHRASE in html


def test_form_carries_the_note_beside_eval_steps(client) -> None:
    pid = _project(client)
    body = client.get(f"/projects/{pid}/training").text
    after = body.split('id="eval-steps"', 1)[1]
    assert 'id="eval-loss-note"' in after.split("</details>", 1)[0]
    assert _PHRASE in after and "Judge recall with a Testing run instead." in after


@pytest.mark.parametrize("mode", ["sft", None])
def test_sft_run_detail_with_eval_loss_shows_the_note(client, mode: str | None) -> None:
    pid = _project(client)
    rid = _run(pid, mode, {"eval_loss": 1.9, "best_eval_loss": 1.2})
    assert _note(client.get(f"/projects/{pid}/training?run={rid}").text)


def test_preference_run_detail_has_no_note(client) -> None:
    pid = _project(client)
    rid = _run(pid, "dpo", {"eval_loss": 0.6, "eval_rewards/accuracies": 0.8})
    assert not _note(client.get(f"/projects/{pid}/training?run={rid}").text)


def test_run_without_eval_loss_has_no_note(client) -> None:
    pid = _project(client)
    rid = _run(pid, "sft", {"final_loss": 0.5})
    assert not _note(client.get(f"/projects/{pid}/training?run={rid}").text)


@pytest.mark.parametrize(("mode", "hidden"), [("dpo", True), ("kto", True), ("sft", False), ("tool_sft", False)])
def test_switching_route_hides_the_note_only_for_preference_routes(mode: str, hidden: bool) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the template's route switch")
    html = _TEMPLATE.read_text()
    start = html.index('  let previousTrainingMode = document.querySelector')
    end = html.index("  });\n  });\n", start) + len("  });\n  });\n")
    source = html[start:end]
    harness = r"""
const listeners = [];
const el = () => ({hidden: false, textContent: '', value: ''});
const els = {'eval-loss-note': el(), 'preference-sft-weight-row': el()};
const radio = {value: MODE, checked: true, addEventListener: (_e, f) => listeners.push(f)};
const context = {
  document: {
    querySelector: q => q.includes(':checked') ? {value: 'sft'} : null,
    querySelectorAll: () => [radio],
    getElementById: id => els[id] || el(),
  },
  PREF_MODES: ['dpo', 'kto'], PREF_DEFAULTS: {dpo: {}, kto: {}}, routeHelp: {sft: ''},
  modeHelp: el(), dpoHelp: el(), ktoHelp: el(), prefOptions: el(), advancedHelp: el(),
  loadStartPoints: () => {}, loadHealth: () => {}, datasetIdInput: null,
};
require('node:vm').createContext(context);
require('node:vm').runInContext(SOURCE, context);
listeners.forEach(f => f());
console.log(JSON.stringify({hidden: els['eval-loss-note'].hidden}));
"""
    script = f"const MODE = {json.dumps(mode)}; const SOURCE = {json.dumps(source)};\n{harness}"
    r = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1])["hidden"] is hidden


def test_guide_help_says_it_too() -> None:
    from finetune_studio.guide import settings_help

    help_ = settings_help.SETTING_HELP
    assert "not a recall signal" in help_["eval_steps"]["pitfall"]
    assert "worse model" in help_["early_stopping"]["pitfall"]
    kb = (Path(__file__).resolve().parents[1] / "src/finetune_studio/guide/kb/06-training.md").read_text()
    assert _PHRASE in kb
