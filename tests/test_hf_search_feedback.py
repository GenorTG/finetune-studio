"""Model Library must keep the completed search result, not a typing timeout."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_TEMPLATE = Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates/hf_models.html"


@pytest.mark.parametrize("outcome", ["empty", "results", "error"])
def test_typing_timeout_cannot_replace_search_feedback(outcome: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the template's search handlers")
    source = _TEMPLATE.read_text().split("<script>", 1)[1].split("// ── Model favorites", 1)[0]
    harness = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const timers = new Map();
let timerId = 0;
let statusText = '';
const status = {
  get textContent() { return statusText; },
  set textContent(value) { statusText = value; },
  get innerHTML() { return statusText; },
  set innerHTML(value) { statusText = value; }
};
const elements = {
  'hf-q': {value: 'missing-model'},
  'hf-task': {value: 'text-generation'},
  'hf-sort': {value: 'downloads'},
  'hf-status': status,
  'hf-grid': {innerHTML: ''}
};
const context = {
  document: {getElementById: id => elements[id]},
  setTimeout: callback => { timers.set(++timerId, callback); return timerId; },
  clearTimeout: id => timers.delete(id),
  encodeURIComponent,
  esc: value => value,
  console: {error: () => {}},
  renderResults: () => {},
  fts: {api: {get: async () => {
    if (OUTCOME === 'error') throw new Error('Hub unavailable');
    return {results: OUTCOME === 'empty' ? [] : [{repo_id: 'test/model'}], took_ms: 12};
  }}}
};
vm.createContext(context);
vm.runInContext(SOURCE, context);
(async () => {
  context.onHfQueryInput();
  await context.doSearch();
  const expected = OUTCOME === 'error' ? '⚠ Search failed: Hub unavailable'
    : (OUTCOME === 'empty' ? '0 results · 12ms' : '1 result · 12ms');
  assert.equal(status.textContent, expected);
  for (const callback of timers.values()) callback();
  assert.equal(status.textContent, expected, 'typing timeout replaced the completed search feedback');
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    script = f"const OUTCOME = {json.dumps(outcome)}; const SOURCE = {json.dumps(source)};\n{harness}"
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_files_dialog_does_not_promise_a_remote_file_picker() -> None:
    source = _TEMPLATE.read_text()
    assert "Pick files…" not in source
    assert ">Downloaded files</button>" in source
    assert "Downloaded files in ${repo}" in source


def test_an_older_search_answering_late_cannot_replace_the_newer_one() -> None:
    """The page-load search (all models) answered after the user's query and drew the unfiltered list over it."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the template's search handlers")
    source = _TEMPLATE.read_text().split("<script>", 1)[1].split("// ── Model favorites", 1)[0]
    harness = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
let statusText = '';
const status = {
  get textContent() { return statusText; }, set textContent(v) { statusText = v; },
  get innerHTML() { return statusText; }, set innerHTML(v) { statusText = v; }
};
const elements = {
  'hf-q': {value: ''}, 'hf-task': {value: 'text-generation'}, 'hf-sort': {value: 'downloads'},
  'hf-status': status, 'hf-grid': {innerHTML: ''}
};
const pending = [];
let rendered = null;
const context = {
  document: {getElementById: id => elements[id]},
  setTimeout: () => 0, clearTimeout: () => {}, encodeURIComponent, esc: v => v, console: {error: () => {}},
  renderResults: rows => { rendered = rows.map(r => r.repo_id); },
  fts: {api: {get: url => new Promise(resolve => pending.push({url, resolve}))}}
};
vm.createContext(context);
vm.runInContext(SOURCE, context);
(async () => {
  const first = context.doSearch();                 // page load: q=''
  elements['hf-q'].value = 'SmolLM2';
  const second = context.doSearch();                // the user's search
  assert.equal(pending.length, 2);
  pending[1].resolve({results: [{repo_id: 'HuggingFaceTB/SmolLM2-135M-Instruct'}], took_ms: 310});
  await second;
  pending[0].resolve({results: [{repo_id: 'openai-community/gpt2'}, {repo_id: 'Qwen/Qwen3-0.6B'}], took_ms: 4200});
  await first;
  assert.deepEqual(rendered, ['HuggingFaceTB/SmolLM2-135M-Instruct']);
  assert.equal(status.textContent, '1 result · 310ms');
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    script = f"const SOURCE = {json.dumps(source)};\n{harness}"
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_search_route_reports_its_time(client, monkeypatch) -> None:
    """The status line read `· 0ms` on every search: the route never sent took_ms."""
    from finetune_studio.webui.routes import hf_models

    monkeypatch.setattr(hf_models, "_search_hf", lambda req: [{"repo_id": "test/model"}])
    r = client.get("/api/hf/search?q=test")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["results"] == [{"repo_id": "test/model"}]
    assert isinstance(body["took_ms"], int) and body["took_ms"] >= 0
