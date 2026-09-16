#!/usr/bin/env python3
"""Multi-viewport breakpoint test with per-viewport screenshots.

For each viewport in BREAKPOINTS:
  1. Set viewport, navigate to training page
  2. Wait for CSS to settle
  3. Verify expected @media rules are active via matchMedia
  4. Take full-page screenshot
  5. Capture computed styles on a few key elements to spot-check layout
     (horiz scroll, main width, header height, step indicator visibility)
  6. Save screenshot to OUT_DIR/<width>.png

Also runs the same checks across multiple pages (overview, training, data,
benchmarks) at 1200px to spot visual regressions anywhere.

Usage: python3 tests/test_breakpoints_visual.py
"""
import json
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parent.parent
OUT = Path("/home/genorbox1/.openclaw/media/outbound")
OUT.mkdir(parents=True, exist_ok=True)

# All @media rules DEFINED in app.css (keep in sync)
DEFINED_RULES = [
    "(max-width: 480px)",
    "(max-width: 700px)",
    "(max-width: 780px)",
    "(max-width: 900px)",
    "(max-width: 1100px)",
    "(max-width: 1280px)",
]

BREAKPOINTS = [320, 375, 414, 480, 600, 768, 780, 900, 1024, 1100, 1200, 1280, 1440, 1920]

# Pages to screenshot at 1200px (single canonical viewport)
PAGES_1200 = [
    ("overview", "/projects/264f8765"),
    ("training", "/projects/264f8765/training"),
    ("data", "/projects/264f8765/data"),
    ("data-prep", "/projects/264f8765/data-prep"),
    ("rag", "/projects/264f8765/rag"),
    ("testing", "/projects/264f8765/testing"),
    ("chat", "/projects/264f8765/chat"),
    ("benchmarks", "/projects/264f8765/benchmarks"),
    ("dashboard", "/"),
]


def parse_width(rule: str) -> int:
    import re
    m = re.search(r"(\d+)\s*px", rule)
    return int(m.group(1)) if m else 0


def inspect_layout(page) -> dict:
    """Return layout metrics we want to spot-check per viewport."""
    return page.evaluate("""
        (function(){
            var r = {};
            r.viewport = window.innerWidth;
            r.docWidth = document.documentElement.scrollWidth;
            r.hasHorizScroll = document.documentElement.scrollWidth > window.innerWidth;
            r.bodyWidth = document.body.scrollWidth;
            // App + grid layout
            var app = document.querySelector('.app');
            var main = document.querySelector('.main');
            var content = document.querySelector('.content');
            var header = document.querySelector('.session-bar');
            var tabs = document.querySelector('.sb-tabs');
            r.appW = app ? Math.round(app.getBoundingClientRect().width) : null;
            r.mainW = main ? Math.round(main.getBoundingClientRect().width) : null;
            r.contentW = content ? Math.round(content.getBoundingClientRect().width) : null;
            r.headerH = header ? Math.round(header.getBoundingClientRect().height) : null;
            // Tools row visibility at this width
            r.toolsTop = tabs ? Math.round(tabs.getBoundingClientRect().height) : null;
            // Step indicator visible?
            var step4 = document.querySelectorAll('.step')[3];
            r.step4Text = step4 ? step4.textContent.trim() : null;
            r.step4BoxRight = step4 ? Math.round(step4.getBoundingClientRect().right) : null;
            // RAM/VRAM labels (glued-to-value bug check)
            var ram = document.getElementById('ram-text');
            if (ram) {
                var ramBox = ram.getBoundingClientRect();
                var label = ram.previousElementSibling;
                var labelBox = label ? label.getBoundingClientRect() : null;
                r.ramLabelGap = labelBox ? Math.round(ramBox.left - labelBox.right) : null;
            } else {
                r.ramLabelGap = "no_ram_element";
            }
            return r;
        })()
    """)


def check_rules(page, actual_w: int):
    """Verify each DEFINED_RULE matches iff actual_w <= threshold."""
    fails = []
    matched = []
    for rule in DEFINED_RULES:
        threshold = parse_width(rule)
        should = actual_w <= threshold
        is_match = page.evaluate(f"window.matchMedia({rule!r}).matches")
        if should and not is_match:
            fails.append(f"  {rule}: SHOULD match (<={threshold}) but DOES NOT")
        elif not should and is_match:
            fails.append(f"  {rule}: should NOT match (>{threshold}) but DOES")
        if is_match:
            matched.append(rule)
    return fails, matched


def multi_viewport_suite(page) -> tuple[bool, list]:
    """Run all viewport checks on /training. Return (all_ok, results)."""
    results = []
    all_ok = True
    for w in BREAKPOINTS:
        page.set_viewport_size({"width": w, "height": 800})
        page.goto(
            f"http://fan-dragon:7860/projects/264f8765/training?bust=bp-{w}",
            wait_until="networkidle",
            timeout=15000,
        )
        actual_w = page.evaluate("window.innerWidth")
        fails, matched = check_rules(page, actual_w)
        layout = inspect_layout(page)
        # Screenshot
        png = page.screenshot(full_page=True)
        out_path = OUT / f"bp-training-{w}.png"
        out_path.write_bytes(png)
        ok = not fails
        all_ok = all_ok and ok
        results.append({
            "viewport": w,
            "actual": actual_w,
            "ok": ok,
            "matched": matched,
            "fails": fails,
            "layout": layout,
            "screenshot": str(out_path),
            "screenshot_bytes": len(png),
        })
    return all_ok, results


def multi_page_suite(page) -> tuple[bool, list]:
    """Snapshot every project page at canonical 1200px viewport."""
    page.set_viewport_size({"width": 1200, "height": 800})
    results = []
    all_ok = True
    for name, path in PAGES_1200:
        url = f"http://fan-dragon:7860{path}?bust=phase-c-{name}"
        resp = page.goto(url, wait_until="networkidle", timeout=15000)
        status = resp.status if resp else None
        actual_w = page.evaluate("window.innerWidth")
        layout = inspect_layout(page)
        png = page.screenshot(full_page=True)
        out_path = OUT / f"phase-c-{name}.png"
        out_path.write_bytes(png)
        ok = status == 200
        all_ok = all_ok and ok
        results.append({
            "page": name,
            "url": path,
            "status": status,
            "viewport": actual_w,
            "ok": ok,
            "layout": layout,
            "screenshot": str(out_path),
        })
    return all_ok, results


def main() -> int:
    print(f"== breakpoints test (visual) ==")
    print(f"  output dir: {OUT}")
    print(f"  viewports:  {BREAKPOINTS}")
    print(f"  pages @1200:{len(PAGES_1200)}")
    summary = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            ctx = browser.new_context(device_scale_factor=1)
            page = ctx.new_page()
            print("\n-- Phase A: multi-viewport breakpoint tests --")
            a_ok, a_results = multi_viewport_suite(page)
            for r in a_results:
                tag = "OK  " if r["ok"] else "FAIL"
                mt = ",".join(str(parse_width(x)) for x in r["matched"])
                print(f"  [{tag}] {r['viewport']:>4}px -> {len(r['matched'])} rules [{mt}]"
                      f" horiz={r['layout']['hasHorizScroll']}"
                      f" mainW={r['layout']['mainW']}"
                      f" headerH={r['layout']['headerH']}"
                      f" ramGap={r['layout']['ramLabelGap']}"
                      f" -> {Path(r['screenshot']).name}")
                if not r["ok"]:
                    for fail in r["fails"]:
                        print(f"          {fail}")
            summary["phase_a"] = {
                "ok": a_ok,
                "count": len(a_results),
                "results": a_results,
            }

            print("\n-- Phase C: multi-page screenshots @ 1200px --")
            c_ok, c_results = multi_page_suite(page)
            for r in c_results:
                tag = "OK  " if r["ok"] else "FAIL"
                print(f"  [{tag}] {r['page']:12} status={r['status']}"
                      f" horiz={r['layout']['hasHorizScroll']}"
                      f" mainW={r['layout']['mainW']}"
                      f" ramGap={r['layout']['ramLabelGap']}"
                      f" -> {Path(r['screenshot']).name}")
            summary["phase_c"] = {
                "ok": c_ok,
                "count": len(c_results),
                "results": c_results,
            }

            ctx.close()
        finally:
            browser.close()

    # Write JSON summary
    summary_path = OUT / "phase-ac-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nJSON summary: {summary_path}")
    overall = summary["phase_a"]["ok"] and summary["phase_c"]["ok"]
    print(f"\nRESULT: {'ALL PASS' if overall else 'FAILURES'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
