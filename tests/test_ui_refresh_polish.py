"""UI refresh follow-ups (#54 #45 #44): field sizing, danger hover, light contrast, re-runnable page scripts, wizard.

Live 2026-11-01 (light theme, 1440 px): the four-:not() input rule (specificity 0,4,1) forced the Files list's
compact search (.fl-search, 11 px / 4 px padding) to 40 px / 15 px, and number fields (content-box) to 58 px; a
hovered red "Remove" turned teal because the generic .btn hover out-ranked .btn.danger.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

W = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui"
TEMPLATES = W / "templates"


def _r(rel: str) -> str:
    return (W / rel).read_text(encoding="utf-8")


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the template scripts")
    return node


def _run_node(script: str) -> subprocess.CompletedProcess:
    return subprocess.run([_node(), "-e", script], capture_output=True, text=True, timeout=20, check=False)


# ── CSS: cascade and contrast ─────────────────────────────────────────


def _rule(css: str, selector: str) -> str:
    start = css.index(selector)
    return css[start: css.index("}", start)]


def test_bare_input_refresh_rule_has_zero_specificity_and_border_box() -> None:
    css = _r("static/css/app.css")
    rule = _rule(css, ':where(input:not([type="checkbox"])')
    assert "select, textarea, .input {" in rule and "box-sizing: border-box;" in rule
    assert not re.search(r'(?m)^input:not\(\[type="checkbox"\]\)', css)   # the 0,4,1 form is gone


def test_compact_fields_keep_their_size_under_the_refresh_rule() -> None:
    css = _r("static/css/app.css")
    for selector in (".input.fl-search {", ".input.text-xs {"):   # 0,2,0 beats the refresh's .input (0,1,0)
        assert selector in css, selector
    assert "min-height: 0" in _rule(css, ".input.fl-search {")


def test_danger_hover_outranks_the_generic_button_hover() -> None:
    css = _r("static/css/app.css")
    generic = css.index(".btn:hover:not(:disabled) { border-color: var(--accent)")
    danger = css.index(".btn.danger:hover:not(:disabled) {")
    assert danger > generic            # same-or-higher specificity (0,4,0 vs 0,3,0) and later: red wins
    assert "border-color: var(--err)" in _rule(css, ".btn.danger:hover:not(:disabled) {")


def test_next_step_why_is_declared_once() -> None:
    css = _r("static/css/app.css")
    assert len(re.findall(r"(?m)^\.workspace-subnav > \.next-step \.next-step-why \{", css)) == 1


def _lum(hex_color: str) -> float:
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_lum(a), _lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _mix(a: str, b: str, pa: float) -> str:
    """CSS color-mix(in srgb, a pa, b)."""
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x * pa + y * (1 - pa)):02x}" for x, y in zip(ca, cb, strict=True))


def _light_tokens() -> dict[str, str]:
    block = _r("static/css/tokens.css").split(':root[data-theme="light"]', 1)[1].split("}", 1)[0]
    found = dict(re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3})\b", block))
    return {k: v if len(v) == 7 else "#" + "".join(c * 2 for c in v[1:]) for k, v in found.items()}


def test_light_theme_text_meets_aa_contrast() -> None:
    t = _light_tokens()
    for bg in (t["--bg-app"], t["--bg-panel"], t["--bg-panel-2"]):
        assert _contrast(t["--text-dim"], bg) >= 4.5, bg
    css = _r("static/css/app.css")
    assert "color-mix(in srgb, var(--accent) 86%, var(--text)); color: var(--on-accent)" in css
    hover = _mix(t["--accent"], t["--text"], 0.86)
    assert _contrast(t["--on-accent"], hover) >= 4.5                    # was white-mixed: 4.0:1
    assert "color: var(--accent-text, var(--accent));" in _rule(css, ".sb-tab.active {\n  border-color")
    nav_bg = _mix(t["--accent"], t["--bg-panel"], 0.12)
    assert _contrast(t["--accent-text"], nav_bg) >= 4.5                 # selected nav was 4.36:1


def test_token_and_app_css_cache_busts_were_bumped() -> None:
    base = _r("templates/base.html")
    assert "tokens.css?v=2" in base and "app.css?v=141" in base


# ── page scripts can run twice (full load, then an SPA visit) ─────────


def _scripts_block(template: Path) -> str | None:
    text = template.read_text(encoding="utf-8")
    if "{% block scripts %}" not in text:
        return None
    block = text.split("{% block scripts %}", 1)[1].split("{% endblock %}", 1)[0]
    return block


def _inline_js(block: str) -> list[str]:
    bodies = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", block, flags=re.S)
    out = []
    for body in bodies:
        body = re.sub(r"\{%.*?%\}", "", body, flags=re.S)
        body = re.sub(r"\{\{.*?\}\}", "null", body, flags=re.S)
        out.append(body)
    return out


ALL_PAGE_SCRIPTS = sorted(p for p in TEMPLATES.rglob("*.html") if _scripts_block(p))


@pytest.mark.parametrize("template", ALL_PAGE_SCRIPTS, ids=lambda p: p.name)
def test_page_script_has_no_column0_const_or_let(template: Path) -> None:
    """spa.js rewrites column-0 const/let to var on a re-visit; a const created by the first full load cannot be
    redeclared, so the page threw a SyntaxError on its next SPA visit (#45 fixed this for projects.html only)."""
    block = _scripts_block(template) or ""
    hits = [m.group(0) for js in _inline_js(block) for m in re.finditer(r"(?m)^(?:const|let) \S+", js)]
    assert not hits, f"{template.name}: declare page globals with var: {hits}"


@pytest.mark.parametrize("name", ["hf_models.html", "inference.html"])
def test_page_script_runs_again_after_a_full_load(name: str) -> None:
    """Execute the real script twice in one realm: as served (full load), then through spa.js's rewrite."""
    spa = _r("static/js/spa.js")
    fn = spa[spa.index("function rewriteTopLevelDecls"):]
    fn = fn[: fn.index("\n  }\n") + 4]
    sources = _inline_js(_scripts_block(TEMPLATES / name) or "")
    harness = r"""
const vm = require('node:vm');
eval(REWRITE);
const stub = () => new Proxy(function () {}, {get: (t, k) => k === Symbol.toPrimitive ? () => '' : stub(),
                                              apply: () => stub(), construct: () => stub()});
const ctx = vm.createContext({document: stub(), window: stub(), fts: stub(), localStorage: stub(),
                              setTimeout: () => 0, setInterval: () => 0, clearTimeout: () => {},
                              clearInterval: () => {}, fetch: async () => stub(), console: {error() {}, log() {}}});
function run(code) {
  try { new vm.Script(code).runInContext(ctx); }
  catch (e) { if (e instanceof SyntaxError || /already been declared/.test(String(e))) throw e; }
}
for (const s of SOURCES) run(s);                       // full page load
for (const s of SOURCES) run(rewriteTopLevelDecls(s)); // SPA re-visit
"""
    script = f"const REWRITE = {json.dumps(fn)}; const SOURCES = {json.dumps(sources)};\n{harness}"
    result = _run_node(script)
    assert result.returncode == 0, result.stdout + result.stderr


# ── Quick start: a stopped test run is not a finished test ────────────


def _wiz_test_source() -> str:
    page = (TEMPLATES / "project_wizard.html").read_text(encoding="utf-8")
    start = page.index("  window.wizTest = async function () {")
    return page[start: page.index("  window.wizRunAll", start)]


_WIZ_HARNESS = r"""
const assert = require('node:assert/strict');
const lines = [];
const status = {};
let busy = false;
function setBusy(b) { busy = b; }
function line(s) { lines.push(s); }
function set(id, sub) { status[id] = sub; }
function pill() {}
function sleep() { return Promise.resolve(); }
async function wizRefresh() {}
async function wizEnsureMerged() { return '/merged'; }
var wizLastRunId = '';
const PID = 'p1';
const document = {getElementById: () => null};
const window = {};
async function api(u) {
  if (u.startsWith('/api/training/runs/')) return [{id: 'r1', status: 'done'}];
  if (u.endsWith('/auto-suites')) return [{suite_path: '/s.yaml', case_count: 3}];
  if (u === '/api/settings/testing') return {auto_judge: AUTO_JUDGE};
  if (u.endsWith('/cases')) return [];
  return RUN_ROW;
}
async function post(u) { return u.endsWith('/run-suite') ? {benchmark_id: 'b1'} : {}; }
eval(WIZ);
(async () => {
  const ok = await window.wizTest();
  console.log(JSON.stringify({ok, lines, status: status['wiz-test']}));
})().catch(e => { console.error(e); process.exitCode = 1; });
"""


def _wiz(run_row: dict, auto_judge: bool = False) -> dict:
    script = (f"const RUN_ROW = {json.dumps(run_row)}; const AUTO_JUDGE = {json.dumps(auto_judge)}; "
              f"const WIZ = {json.dumps(_wiz_test_source())};\n{_WIZ_HARNESS}")
    result = _run_node(script)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_a_cancelled_test_run_fails_the_step() -> None:
    out = _wiz({"status": "cancelled", "progress_done": 1, "progress_total": 3})
    assert out["ok"] is False                          # Run all stops instead of announcing completion
    assert any("stopped before it finished" in s for s in out["lines"])
    assert not any("answers are saved" in s for s in out["lines"])


def test_a_finished_test_run_still_succeeds() -> None:
    out = _wiz({"status": "done", "progress_done": 3, "progress_total": 3})
    assert out["ok"] is True and out["status"] == "answers saved — awaiting judgement"


def test_stopped_judging_is_reported_as_stopped_not_as_a_score() -> None:
    out = _wiz({"status": "done", "progress_done": 3, "judge_status": "cancelled", "scores": {}}, auto_judge=True)
    assert out["status"] == "answers saved — judging stopped"
    assert not any(s.startswith("Test result") for s in out["lines"])
