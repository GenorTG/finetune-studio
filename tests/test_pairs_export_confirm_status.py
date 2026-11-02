"""Pairs → Export approved: the status line must not say "Exporting…" while the coverage confirm waits.

The export first asks the server; a 409 lists files whose data points are not all approved and the page asks
"Export anyway?". The status line already read "Exporting 8 approved pair(s)…" behind that dialog, although
nothing had been exported yet (found live 2026-11-02, B3).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_TEMPLATE = Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates/data_prep.html"


def _run(answer: bool) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the template's export handler")
    html = _TEMPLATE.read_text()
    start = html.index("async function dpExportApproved()")
    end = html.index("\n}\n", start) + 3
    source = html[start:end]
    harness = r"""
const statuses = [];
let atConfirm = null;
let calls = 0;
const context = {
  _rq: {counts: {approved: 8}}, _prepPid: 'p1', encodeURIComponent, URL: {createObjectURL: () => 'blob:x', revokeObjectURL: () => {}},
  _dpExportBlobUrl: '', escapeHtml: s => s, dpGroundedShare: () => 0,
  dpBridgeStatus: (t, kind) => statuses.push(t),
  dpRefreshExportedDatasets: async () => {},
  document: {getElementById: () => null},
  window: {},
  fts: {confirm: async () => { atConfirm = statuses[statuses.length - 1]; return ANSWER; }},
  fetch: async url => {
    calls += 1;
    if (!url.includes('force=true')) return {status: 409, ok: false,
      json: async () => ({incomplete_files: [{filename: 'a.txt', status: 'gaps', counts: {open: 1}}]})};
    return {status: 200, ok: true, headers: {get: k => ({'X-Rows': '8', 'X-Dataset-Id': 'd1'})[k] || null},
            blob: async () => ({})};
  },
};
require('node:vm').createContext(context);
require('node:vm').runInContext(SOURCE + '\nthis.run = dpExportApproved;', context);
context.run().then(() => console.log(JSON.stringify({statuses, atConfirm, calls})))
  .catch(e => { console.error(e); process.exitCode = 1; });
"""
    script = f"const ANSWER = {json.dumps(answer)}; const SOURCE = {json.dumps(source)};\n{harness}"
    r = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_status_says_not_exported_while_the_confirm_waits() -> None:
    out = _run(answer=False)
    assert "Exporting" not in out["atConfirm"]
    assert "Not exported yet" in out["atConfirm"] and "1 file(s) not complete" in out["atConfirm"]
    assert out["statuses"][-1].startswith("Export cancelled")
    assert out["calls"] == 1


def test_export_anyway_reports_the_export() -> None:
    out = _run(answer=True)
    assert out["calls"] == 2
    assert out["statuses"][-1].startswith("Exported 8 pair(s)")
