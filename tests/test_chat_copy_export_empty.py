"""Chat: Copy all / Export on an empty conversation say there is nothing yet; Copy all uses the shared fallback.

On an empty conversation Copy all reported "Conversation copied" with nothing copied, and Export downloaded a file
holding only the title and the date (found live 2026-11-02, B3). Copy all also called navigator.clipboard directly,
which is missing over plain HTTP from another machine.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_TEMPLATE = Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates/chat_v2.html"


def _source() -> str:
    html = _TEMPLATE.read_text()
    start = html.index("function _chatCopyAll() {")
    end = html.index("\n}\n", html.index("function _chatExport() {")) + 3
    return html[start:end]


def _run(history: list[dict]) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the template's chat handlers")
    harness = r"""
const notes = [], copies = [];
let downloads = 0;
const context = {
  _chatHistory: HISTORY,
  window: {fts: {notify: (m, k) => notes.push([m, k])}, ftsCopyText: async (t, ok) => copies.push([t, ok])},
  navigator: {},
  Blob: function () {}, URL: {createObjectURL: () => 'blob:x', revokeObjectURL: () => {}},
  document: {createElement: () => ({click: () => { downloads += 1; }})},
  Date,
};
require('node:vm').createContext(context);
require('node:vm').runInContext(SOURCE + '\nthis.copy = _chatCopyAll; this.exp = _chatExport;', context);
Promise.resolve(context.copy()).then(() => { context.exp(); console.log(JSON.stringify({notes, copies, downloads})); });
"""
    script = f"const HISTORY = {json.dumps(history)}; const SOURCE = {json.dumps(_source())};\n{harness}"
    r = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_empty_conversation_copies_and_exports_nothing_and_says_so() -> None:
    out = _run([])
    assert out["copies"] == [] and out["downloads"] == 0
    assert [n[0] for n in out["notes"]] == ["Nothing to copy yet: the conversation is empty.",
                                           "Nothing to export yet: the conversation is empty."]


def test_a_conversation_is_copied_through_the_shared_helper_and_exported() -> None:
    out = _run([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
    assert out["copies"] == [["**You**: hi\n\n---\n\n**Assistant**: hello", "Conversation copied to clipboard"]]
    assert out["downloads"] == 1
    assert out["notes"] == [["Chat exported as Markdown", "success"]]
