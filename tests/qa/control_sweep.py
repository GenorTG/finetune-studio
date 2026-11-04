"""Click every visible, enabled control on every registered page and report what each one did (B3).

    FTS_SESSION=<session token> .venv/bin/python tests/qa/control_sweep.py <project id> [--pages files,pairs]
        [--base http://127.0.0.1:7860] [--gpu] [--per-label N] [--json out.json]

Per page, per control (the n-th visible enabled button / summary / checkbox / select / tab link inside the page
content, re-found after a fresh load): reload the page, click it, answer any in-app confirm with Cancel and any native
dialog with dismiss, wait, then record
  - JS errors (uncaught exceptions, console errors) the click caused,
  - server errors (5xx) and client errors (4xx: listed as REFUSED, to review: an honest refusal the page explains is
    fine, a control that can only be refused should be disabled with its reason),
  - feedback: a DOM change outside the session bar (page content, toast, overlay, panel), a scroll, a changed
    form value, a request, a dialog, or a navigation. Radios and checkboxes are clicked through their label.
A control with no feedback and no error is listed as SILENT (a dead button, or one that needs a state the page is not
in). Controls whose label starts GPU / helper work (train, build, run, export, load, mine, judge, ...) are skipped unless
--gpu: run that pass only when no GPU job is running. Use a disposable project: Cancel keeps deletes from happening, but
a control without a confirm acts for real. Controls that end sessions or change the password (Sign out, Sign out other
browsers, Change / Turn password off) are never clicked: they sign the sweep and the owner's other browsers out.
Exit code 1 when any control raised a JS error or a 5xx.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from finetune_studio.guide.registry import PAGES

GPU_WORDS = re.compile(
    r"\b(train|training|build|rebuild|index|re-?embed|embed|run|re-?run|export|merge|load|download|install|update|repair|restart|"
    r"apply|mine|mining|generate|judge|study set|preference|send|ask|test connection|test local|test reading|"
    r"benchmark|evaluate|quantize|abliterat|heretic|start|quick)\b", re.IGNORECASE)
NEVER = re.compile(r"^(sign out|change password|set password|turn password off)\b", re.IGNORECASE)
CONTROLS = ("#content button, #content summary, #content input[type=checkbox], #content input[type=radio], "
            "#content [role=tab], #content a.btn")
FEEDBACK_JS = """() => new Promise(resolve => {
  // Feedback = anything the user can see change: the page content, a toast, an overlay or panel outside
  // #content (tutorial, modal, Guide), a scroll (jump links), or a form field's value (presets write .value,
  // which is a property, not DOM). The session bar ticks on its own (clock, status), so it never counts.
  const content = document.getElementById('content') || document.body;
  const bar = document.getElementById('session-bar');
  let changed = false;
  const mo = new MutationObserver(recs => {
    for (const r of recs) { if (!(bar && bar.contains(r.target))) { changed = true; return; } }
  });
  mo.observe(document.body, {subtree: true, childList: true, attributes: true, characterData: true});
  const fields = () => Array.from(content.querySelectorAll('input, textarea, select'))
    .map(e => e.type === 'checkbox' || e.type === 'radio' ? String(e.checked) : String(e.value)).join('\\u0001');
  const before = fields(), y = window.scrollY;
  window.__ftsSweepStop = () => { mo.disconnect(); return changed || window.scrollY !== y || fields() !== before; };
  resolve(true);
})"""


def label(el) -> str:
    try:
        t = el.evaluate("e => (e.getAttribute('aria-label') || (e.labels && e.labels[0] && e.labels[0].innerText) || "
                        "(e.closest('label') && e.closest('label').innerText) || e.innerText || e.title || e.id || "
                        "e.value || '').trim().replace(/\\s+/g, ' ').slice(0, 60)")
    except PlaywrightError:
        return "?"
    return t or "?"


def candidates(page):
    out = []
    for el in page.query_selector_all(CONTROLS):
        try:
            if el.is_visible() and el.is_enabled():
                out.append(el)
        except PlaywrightError:
            continue
    return out


class _Recorder:
    """What one click caused: JS errors, responses, native dialogs."""

    def __init__(self, base: str) -> None:
        self.base = base
        self.errors: list[str] = []
        self.responses: list[str] = []
        self.dialogs: list[str] = []

    def on_error(self, e) -> None:
        self.errors.append(str(e)[:160])

    def on_console(self, m) -> None:
        # The browser logs every 4xx as "Failed to load resource"; the response itself is recorded (and reviewed
        # as REFUSED), so only a 5xx one, or a real script error, is an error here.
        if m.type == "error" and not re.match(r"Failed to load resource: the server responded with a status of 4",
                                              m.text):
            self.errors.append("console: " + m.text[:160])

    def on_response(self, r) -> None:
        self.responses.append(f"{r.status} {r.request.method} {r.url.replace(self.base, '')[:90]}")

    def on_dialog(self, d) -> None:
        self.dialogs.append(d.type)
        d.dismiss()

    def _pairs(self):
        return (("pageerror", self.on_error), ("console", self.on_console), ("response", self.on_response),
                ("dialog", self.on_dialog))

    def attach(self, page) -> None:
        for ev, fn in self._pairs():
            page.on(ev, fn)

    def detach(self, page) -> None:
        for ev, fn in self._pairs():
            page.remove_listener(ev, fn)


def load(page, url: str) -> None:
    """Open ``url``; a page script that re-navigates during the load (SPA redirect) gets one retry."""
    for attempt in range(3):
        try:
            page.goto(url, wait_until="networkidle", timeout=60000)
            return
        except PlaywrightError as e:
            if "interrupted by another navigation" not in str(e) or attempt == 2:
                raise
            page.wait_for_timeout(1000)


def _row_key(text: str) -> str:
    """A row control's label without its row-specific numbers (``Approve #12`` and ``Approve #13`` are one control)."""
    return re.sub(r"\d+", "#", text).strip()


def sweep_page(page, name: str, url: str, gpu: bool, base: str, per_label: int = 0) -> list[dict]:
    load(page, url)
    first = candidates(page)
    total = len(first)
    keep = set(range(total))
    if per_label:   # list pages repeat one control per row: sample the first N of each kind
        seen: dict[str, int] = {}
        keep = set()
        for i, el in enumerate(first):
            k = _row_key(label(el))
            seen[k] = seen.get(k, 0) + 1
            if seen[k] <= per_label:
                keep.add(i)
    results = []
    for i in range(total):
        if i not in keep:
            continue
        load(page, url)
        page.wait_for_timeout(300)
        els = candidates(page)
        if i >= len(els):
            break
        el = els[i]
        text = label(el)
        row = {"page": name, "i": i, "label": text}
        if NEVER.search(text):
            results.append({**row, "result": "skipped (session/password)"})
            continue
        if not gpu and GPU_WORDS.search(text):
            results.append({**row, "result": "skipped (gpu/helper)"})
            continue
        rec = _Recorder(base)
        rec.attach(page)
        before_url = page.url
        dialogs: list[str] = []
        try:
            page.evaluate(FEEDBACK_JS)
            target = el
            if el.evaluate("e => e.matches('input[type=radio], input[type=checkbox]') && !!e.closest('label')"):
                target = el.evaluate_handle("e => e.closest('label')").as_element() or el   # what a user clicks
            target.click(timeout=3000)
            page.wait_for_timeout(1200)
            modal = page.query_selector("[data-fts-modal=cancel]")
            if modal and modal.is_visible():
                dialogs.append("fts.confirm")
                modal.click()
                page.wait_for_timeout(300)
            changed = page.evaluate("() => (window.__ftsSweepStop ? window.__ftsSweepStop() : true)")
        except PlaywrightError as e:
            changed = page.url != before_url
            if "Execution context was destroyed" not in str(e) and page.url == before_url:
                why = [ln.strip() for ln in str(e).splitlines() if "intercepts pointer events" in ln or "not visible" in ln]
                rec.errors.append(f"click: {str(e).splitlines()[0][:120]}" + (f" ({why[-1][:160]})" if why else ""))
        finally:
            rec.detach(page)
        errors, responses, dialogs = rec.errors, rec.responses, rec.dialogs + dialogs
        api = [r for r in responses if " /api/" in r or "/api/" in r.split(" ", 2)[-1]]
        bad5 = [r for r in responses if r[:1] == "5"]
        bad4 = [r for r in api if r[:1] == "4"]
        navigated = page.url != before_url
        feedback = changed or bool(api) or bool(dialogs) or navigated
        result = ("ERROR" if errors or bad5 else "REFUSED" if bad4 else "ok" if feedback else "SILENT")
        results.append({**row, "result": result, "errors": errors, "5xx": bad5, "4xx": bad4,
                        "requests": len(api), "dialogs": dialogs, "navigated": navigated})
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pid")
    ap.add_argument("--base", default="http://127.0.0.1:7860")
    ap.add_argument("--pages", default="")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--per-label", type=int, default=0,
                    help="click only the first N controls of each label (row lists repeat one control per row)")
    a = ap.parse_args()
    token = os.environ.get("FTS_SESSION", "")
    if not token:
        print("set FTS_SESSION to a session token", file=sys.stderr)
        return 2
    only = {p for p in a.pages.split(",") if p}
    all_rows: list[dict] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--disable-gpu"])
        ctx = browser.new_context(viewport={"width": 1440, "height": 1000})
        ctx.add_cookies([{"name": "fts_session", "value": token, "url": a.base}])
        ctx.add_init_script("try { localStorage.setItem('fts.tutorial.seen', '1'); } catch (e) {}")
        page = ctx.new_page()
        for name, reg in PAGES.items():
            if only and name not in only:
                continue
            rows = sweep_page(page, name, a.base + reg.url.replace("{pid}", a.pid), a.gpu, a.base, a.per_label)
            all_rows += rows
            bad = [r for r in rows if r["result"] in ("ERROR", "SILENT", "REFUSED")]
            print(f"{name:18s} controls {len(rows):3d}  ok {sum(r['result'] == 'ok' for r in rows):3d}  "
                  f"skipped {sum(r['result'].startswith('skipped') for r in rows):3d}  silent "
                  f"{sum(r['result'] == 'SILENT' for r in rows):3d}  refused {sum(r['result'] == 'REFUSED' for r in rows):3d}  "
                  f"error {sum(r['result'] == 'ERROR' for r in rows):3d}",
                  flush=True)
            for r in bad:
                print(f"    {r['result']:6s} #{r['i']} {r['label']!r} {r.get('errors') or ''} {r.get('5xx') or ''} "
                      f"{r.get('4xx') or ''}",
                      flush=True)
            if a.json:   # written after every page, so a crash keeps what was swept
                with open(a.json, "w", encoding="utf-8") as f:
                    json.dump(all_rows, f, indent=1)
        if a.gpu:   # a Load / Run click loads a model and the load finishes after the click: say what is left behind
            st = page.request.get(a.base + "/api/inference/status")
            if st.ok and st.json().get("loaded"):
                print(f"NOTE  a model is loaded after the sweep: {st.json().get('model')} (unload it if the sweep "
                      "loaded it: POST /api/models/unload)", flush=True)
        browser.close()
    return 1 if any(r["result"] == "ERROR" for r in all_rows) else 0


if __name__ == "__main__":
    sys.exit(main())
