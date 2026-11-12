"""Settings → API tokens: a blank name is refused in plain words, and Revoke asks before cutting scripts off.

A blank name came back as the validator's "name: String should have at least 1 character", and Revoke deleted the
token on one click although every script using it stops working (found live 2026-11-02, B3). "Sign out other
browsers" also acted on one click and signed every other device out (found by the control sweep 2026-11-04).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_JS = Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/static/js/security.js"


def _node(script: str) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute security.js")
    r = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


_CREATE = r"""
const els = {}, calls = [];
const mk = id => (els[id] = els[id] || {id, value: '', hidden: true, checked: false, disabled: false, textContent: '',
  listeners: {}, addEventListener(ev, f) { this.listeners[ev] = f; }, replaceChildren() {}, append() {},
  classList: {toggle() {}}});
const request = async (method, url, body) => {
  calls.push([method, url, body || null]);
  if (url.endsWith('/tokens')) return {name: body.name, token: 'secret'};
  return new Promise(() => {});   // page data: never answers, so it cannot overwrite the status
};
const context = {document: {getElementById: mk, createElement: () => mk('x' + Math.random())},
                 window: {fts: {confirm: async () => true}, location: {}, api: {request}},
                 console, setTimeout, Date, JSON, Promise, Object, Error, encodeURIComponent};
require('node:vm').createContext(context);
require('node:vm').runInContext(SOURCE, context);
(async () => {
  mk('access-token-name').value = NAME;
  await mk('access-token-form').listeners.submit({preventDefault() {}});
  await new Promise(r => setTimeout(r, 30));
  console.log(JSON.stringify({posts: calls.filter(c => c[0] === 'POST').map(c => c[2]),
                              created: mk('access-token-new').textContent, status: mk('access-status').textContent}));
})();
"""


def _create(name: str) -> dict:
    return _node(f"const NAME = {json.dumps(name)}; const SOURCE = {json.dumps(_JS.read_text())};\n{_CREATE}")


def test_blank_name_is_refused_in_plain_words_before_any_request() -> None:
    out = _create("   ")
    assert out["posts"] == [] and out["created"] == ""
    assert "Give the token a name first" in out["status"] and "String should" not in out["status"]


def test_named_token_is_created_with_the_trimmed_name() -> None:
    out = _create("  backup script ")
    assert out["posts"] == [{"name": "backup script"}] and "shown once" in out["created"]


_REVOKE = r"""
const calls = [];
let handler = null;
const b = {addEventListener: (_ev, f) => { handler = f; }};
const t = {id: 't1', name: 'backup'};
const context = {b, t, encodeURIComponent,
  window: {fts: {confirm: async () => CONFIRM}},
  call: async (m, u) => { calls.push([m, u]); return {}; },
  run: async fn => fn()};
require('node:vm').createContext(context);
require('node:vm').runInContext(BLOCK, context);
handler().then(() => console.log(JSON.stringify({deletes: calls.filter(c => c[0] === 'DELETE').length})));
"""


@pytest.mark.parametrize(("confirm", "deletes"), [(False, 0), (True, 1)])
def test_revoke_asks_first(confirm: bool, deletes: int) -> None:
    js = _JS.read_text()
    start = js.index("      b.addEventListener('click', async () => {")
    end = js.index("      li.append(b);", start)
    block = js[start:end]
    out = _node(f"const CONFIRM = {json.dumps(confirm)}; const BLOCK = {json.dumps(block)};\n{_REVOKE}")
    assert out["deletes"] == deletes


@pytest.mark.parametrize(("confirm", "deletes"), [(False, 0), (True, 1)])
def test_sign_out_other_browsers_asks_first(confirm: bool, deletes: int) -> None:
    js = _JS.read_text()
    start = js.index("  $('access-signout-others').addEventListener('click', async () => {")
    end = js.index("  $('access-signout').addEventListener", start)
    block = js[start:end].replace("$('access-signout-others')", "b", 1)
    out = _node(f"const CONFIRM = {json.dumps(confirm)}; const BLOCK = {json.dumps(block)};\n{_REVOKE}")
    assert out["deletes"] == deletes
