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
