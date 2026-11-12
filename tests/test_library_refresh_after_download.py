"""Model Library: a finished download shows up in "Downloaded to this computer" without pressing Refresh."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

WEBUI = Path(__file__).resolve().parents[1] / "src/finetune_studio/webui"


def test_library_page_reloads_its_list_when_a_download_finishes() -> None:
    html = (WEBUI / "templates/hf_models.html").read_text(encoding="utf-8")
    assert "document.addEventListener('fts:task-finished'" in html
    assert "ev.detail.kind === 'download'" in html and "loadLocalModels();" in html
    assert "if (!window._hfDownloadListener)" in html            # one listener across SPA visits


def test_activity_feed_announces_a_task_once_when_it_leaves_running() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to run the activity helper")
    js = (WEBUI / "static/js/activity.js").read_text(encoding="utf-8")
    start = js.index("const _wasActive")
    src = js[start:js.index("function applySnapshot", start)]
    script = src + "\n" + r"""
const fired = [];
globalThis.document = {dispatchEvent: e => fired.push(e.detail.id + ':' + e.detail.status)};
globalThis.CustomEvent = class { constructor(type, init) { this.type = type; this.detail = init.detail; } };
announceFinished([{kind: 'download', id: 'a', status: 'done'}]);         // never seen running: no event
announceFinished([{kind: 'download', id: 'b', status: 'running'}]);
announceFinished([{kind: 'download', id: 'b', status: 'running'}]);
announceFinished([{kind: 'download', id: 'b', status: 'done'}]);
announceFinished([{kind: 'download', id: 'b', status: 'done'}]);         // already announced
process.stdout.write(JSON.stringify(fired));
"""
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10, check=True).stdout
    assert json.loads(out) == ["b:done"]
