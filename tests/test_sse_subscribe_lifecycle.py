"""A page's live stream closes when the SPA leaves the page (found live 2026-11-03).

``fts.subscribe`` opened an EventSource that only the caller could close. The Testing page never closed its
``/api/testing/events`` stream, so every in-app visit to Testing left one more connection open; with the activity
feed's own stream the browser's six connections to the app filled after three visits and every later request of the
tab (a run delete, the page's own API calls) waited until some stream ended. Now every stream is page-scoped by
default and closes on ``fts:beforeNavigate``; only ``persistent: true`` (the activity drawer) outlives the page.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_STATIC = Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/static/js"
_TEMPLATES = _STATIC.parent.parent / "templates"


def _subscribe_source() -> str:
    src = (_STATIC / "app.js").read_text(encoding="utf-8")
    start = src.index("  function subscribe(url, onEvent, opts) {")
    end = src.index("\n  }\n", start) + len("\n  }\n")
    return src[start:end]


def _run(calls: str) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute fts.subscribe")
    harness = r"""
const open = [];
class EventSource { constructor(url) { this.url = url; open.push(this); } close() { open.splice(open.indexOf(this), 1); } }
const listeners = {};
const document = {
  addEventListener(type, fn, o) { (listeners[type] = listeners[type] || []).push({fn, once: !!(o && o.once)}); },
  dispatch(type) { const ls = listeners[type] || []; listeners[type] = ls.filter(l => !l.once); ls.forEach(l => l.fn()); },
};
const context = {EventSource, document, setInterval, clearInterval, fetch: () => Promise.reject(new Error('no'))};
const vm = require('node:vm');
vm.createContext(context);
vm.runInContext(SOURCE + '\nthis.subscribe = subscribe;', context);
const subscribe = context.subscribe;
const out = {};
CALLS
console.log(JSON.stringify(out));
"""
    script = f"const SOURCE = {json.dumps(_subscribe_source())};\n" + harness.replace("CALLS", calls)
    r = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_page_streams_close_on_spa_navigation_and_the_persistent_one_stays() -> None:
    out = _run("""
subscribe('/api/activity/events', () => {}, {persistent: true});
for (let visit = 0; visit < 4; visit++) {          // four in-app visits to the Testing page
  subscribe('/api/testing/events', () => {}, {});
  document.dispatch('fts:beforeNavigate');
}
out.afterVisits = open.map(e => e.url);
const stop = subscribe('/api/x/events', () => {});
stop();                                            // the caller closing it first is still fine
document.dispatch('fts:beforeNavigate');
out.afterExplicit = open.map(e => e.url);
""")
    assert out["afterVisits"] == ["/api/activity/events"]
    assert out["afterExplicit"] == ["/api/activity/events"]


def test_only_the_activity_drawer_keeps_its_stream_across_pages() -> None:
    assert "persistent: true" in (_STATIC / "activity.js").read_text(encoding="utf-8")
    users = [p for p in [*_STATIC.glob("*.js"), *_TEMPLATES.glob("*.html")]
             if p.name != "app.js" and "persistent: true" in p.read_text(encoding="utf-8")]  # app.js defines it
    assert [p.name for p in users] == ["activity.js"]
