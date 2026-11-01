"""The Q&A export registers ONE dataset and hands its id to the Training deep link.

B6-01: the post-export CTA linked ``/training?dataset=latest``, which picks the
newest-*created* row; re-exporting an older dataset updates that row in place,
so "latest" selected an unrelated dataset. B6-02: a failure to register the
export was logged and swallowed, and the page still said "exported into project
datasets".
"""
from __future__ import annotations

import json
import secrets
import shutil
import subprocess
from pathlib import Path

import pytest

from finetune_studio import db
from finetune_studio.data.fs import qa as qa_fs

_TEMPLATE = Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates/data_prep.html"


def _project_with_pair(client) -> str:
    pid = client.post("/api/projects", json={"name": "agent-test-ds-identity"}).json()["id"]
    qa_fs.write_qa_pair(pid, {
        "id": f"ds-{secrets.token_hex(3)}", "source_id": "synthetic-source",
        "question": "What sealed the archive?", "answer": "A vow on green wax.", "status": "approved",
    })
    return pid


def test_re_export_returns_its_own_dataset_id_not_the_newest(client, tmp_path: Path) -> None:
    pid = _project_with_pair(client)
    first = client.get(f"/api/projects/{pid}/data-prep/export?force=true")
    assert first.status_code == 200, first.text
    a_id = first.headers["X-Dataset-Id"]
    assert db.get_dataset(a_id)["project_id"] == pid

    # Dataset B arrives later (an upload, another build): it is now the newest row.
    b_file = tmp_path / "b.jsonl"
    b_file.write_text('{"conversations": []}\n', encoding="utf-8")
    b = db.create_dataset(project_id=pid, name="B", data_path=str(b_file), qa_count=1)
    assert db.list_datasets(pid)[0]["id"] == b["id"]

    again = client.get(f"/api/projects/{pid}/data-prep/export?force=true")
    assert again.status_code == 200, again.text
    assert again.headers["X-Dataset-Id"] == a_id  # the row it updated, not B
    assert again.text.strip()  # the JSONL body is unchanged for existing clients


def test_a_registry_failure_is_an_error_not_an_export(client, monkeypatch: pytest.MonkeyPatch) -> None:
    from finetune_studio.data.prep import dataset_build

    pid = _project_with_pair(client)

    def broken(*_a, **_kw):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(dataset_build, "persist_export", broken)
    r = client.get(f"/api/projects/{pid}/data-prep/export?force=true")
    assert r.status_code == 500, r.text
    assert "could not be saved to the project" in r.json()["detail"]
    assert "No space left" in r.json()["detail"]
    assert "X-Dataset-Id" not in r.headers
    assert db.list_datasets(pid) == []


# ── the page: the CTA links the exact id, and only after it was registered ──

_HARNESS = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
let statusMsg = null;
const panel = {hidden: false, children: [], appendChild(el) { this.children.push(el); }};
const els = {'dp-exported-datasets': panel};
const context = {
  _rq: {counts: {approved: 3}}, _prepPid: 'p1', _dpExportBlobUrl: null,
  dpGroundedShare: () => '0',
  dpBridgeStatus: (msg, state) => { statusMsg = {msg, state}; },
  dpRefreshExportedDatasets: async () => {},
  escapeHtml: s => s, encodeURIComponent,
  URL: {createObjectURL: () => 'blob:x', revokeObjectURL: () => {}},
  fts: {confirm: async () => true},
  window: {},
  document: {
    getElementById: id => els[id] || null,
    createElement: () => { const el = {}; return el; },
  },
  fetch: async () => RESPONSE_OF(),
};
context.window.fts = context.fts;
function RESPONSE_OF() {
  const h = new Map(Object.entries(RESP.headers || {}));
  return {
    ok: RESP.status < 300, status: RESP.status,
    headers: {get: k => (h.has(k) ? h.get(k) : null)},
    json: async () => RESP.json || {}, blob: async () => ({size: 1}),
  };
}
vm.createContext(context);
vm.runInContext(SOURCE, context);
(async () => {
  await context.dpExportApproved();
  const cta = panel.children.find(c => c.id === 'dp-post-export-cta');
  process.stdout.write(JSON.stringify({status: statusMsg, cta: cta ? cta.innerHTML : null}));
})().catch(e => { console.error(e); process.exit(1); });
"""


def _run_export_js(resp: dict) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the template's export handler")
    html = _TEMPLATE.read_text(encoding="utf-8")
    start = html.index("async function dpExportApproved()")
    source = html[start:html.index("\n}\n", start) + 3]
    script = (f"const SOURCE = {json.dumps(source)};\nconst RESP = {json.dumps(resp)};\n" + _HARNESS)
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30, check=True)
    return json.loads(out.stdout)


def test_cta_deep_links_the_registered_dataset_id() -> None:
    got = _run_export_js({"status": 200, "headers": {"X-Rows": "3", "X-Grounded-Rows": "0", "X-Dataset-Id": "ds42"}})
    assert got["status"]["state"] == "success"
    assert "/projects/p1/training?dataset=ds42" in got["cta"]
    assert "dataset=latest" not in got["cta"]


def test_no_training_cta_without_a_registered_dataset() -> None:
    got = _run_export_js({"status": 200, "headers": {"X-Rows": "3"}})
    assert got["cta"] is None


def test_a_failed_registration_shows_the_server_reason() -> None:
    got = _run_export_js({"status": 500, "json": {"detail": "The dataset was built but could not be saved"}})
    assert got["status"]["state"] == "error"
    assert "could not be saved" in got["status"]["msg"]
    assert got["cta"] is None
