"""Copy path works when the Clipboard API is missing or refuses the write.

Over plain HTTP from another machine the page is not a secure context, so ``navigator.clipboard`` does not exist;
the project overview called it directly and threw. A refused write (permissions) said "Copy failed" without
trying the textarea fallback the shared helper already had for the missing case (found live 2026-11-02, B3).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_TEMPLATES = Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates"


def _helper_source() -> str:
    html = (_TEMPLATES / "base.html").read_text()
    start = html.index("    window.ftsCopyText = async function (text) {")
    end = html.index("\n    };\n", start) + len("\n    };\n")
    return html[start:end]


def _run(clipboard: str, exec_ok: bool = True) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the template's copy helper")
    harness = r"""
const notes = [];
let execCalls = 0;
const textarea = {style: {}, setAttribute() {}, select() {}, remove() {}};
const navigator = CLIPBOARD === 'missing' ? {} : {clipboard: {writeText: async () => {
  if (CLIPBOARD === 'refused') throw new Error('NotAllowedError');
}}};
const window = {fts: {notify: (m, k) => notes.push([m, k])}};
const document = {createElement: () => textarea, body: {appendChild() {}},
                  execCommand: () => { execCalls += 1; return EXEC_OK; }};
const context = {window, navigator, document};
require('node:vm').createContext(context);
require('node:vm').runInContext(SOURCE, context);
window.ftsCopyText('/x/y').then(() => console.log(JSON.stringify({notes, execCalls})));
"""
    script = (f"const CLIPBOARD = {json.dumps(clipboard)}; const EXEC_OK = {json.dumps(exec_ok)}; "
              f"const SOURCE = {json.dumps(_helper_source())};\n{harness}")
    r = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(("clipboard", "exec_calls"), [("ok", 0), ("missing", 1), ("refused", 1)])
def test_copy_succeeds_through_the_api_or_the_fallback(clipboard: str, exec_calls: int) -> None:
    out = _run(clipboard)
    assert out["notes"] == [["Path copied", "success"]]
    assert out["execCalls"] == exec_calls


def test_copy_that_cannot_work_says_so_in_plain_words() -> None:
    out = _run("refused", exec_ok=False)
    assert out["notes"][0][1] == "warn" and "copy it by hand" in out["notes"][0][0]


def test_project_overview_uses_the_shared_helper() -> None:
    html = (_TEMPLATES / "project.html").read_text()
    body = html.split("window.dashCopyPath = function (path) {", 1)[1].split("};", 1)[0]
    assert "window.ftsCopyText(path)" in body
    assert "navigator.clipboard" not in body
