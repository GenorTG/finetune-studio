"""Capture the README screenshots from a running WebUI (FTS_BASE, default http://127.0.0.1:7860).

Point it at a throwaway instance holding demo data only (temp HOME / FTS_ROOT, another port), never a real install:
    FTS_BASE=http://127.0.0.1:7871 FTS_DEMO_PID=<project id> .venv/bin/python scripts/capture_shots.py
Writes docs/screenshots/NN_*.png (1440px wide, full-page) and hero_NN_*.png (one 1440x1000 viewport).
"""
from __future__ import annotations

import os
import pathlib
import sys

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

BASE = os.environ.get("FTS_BASE", "http://127.0.0.1:7860").rstrip("/")
PID = os.environ.get("FTS_DEMO_PID", "")
OUT = pathlib.Path("docs/screenshots")

PAGES: list[tuple[str, str]] = [
    ("01_dashboard", f"{BASE}/"),
    ("02_project", f"{BASE}/projects/{PID}"),
    ("03_quick_work", f"{BASE}/projects/{PID}/work"),
    ("04_files", f"{BASE}/projects/{PID}/data"),
    ("05_pairs", f"{BASE}/projects/{PID}/data-prep"),
    ("06_brain", f"{BASE}/projects/{PID}/brain"),
    ("07_training", f"{BASE}/projects/{PID}/training"),
    ("08_models", f"{BASE}/projects/{PID}/models"),
    ("09_export", f"{BASE}/projects/{PID}/export"),
    ("10_testing", f"{BASE}/projects/{PID}/testing"),
    ("11_benchmarks", f"{BASE}/projects/{PID}/benchmarks"),
    ("12_settings", f"{BASE}/settings"),
    ("13_chat", f"{BASE}/projects/{PID}/chat"),
]


# README top table: one viewport each (the full-page shots are thousands of pixels tall).
HERO = {"04_files", "05_pairs", "06_brain", "07_training"}


def main() -> int:
    if not PID:
        print("set FTS_DEMO_PID to the demo project's id", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        # Mark the onboarding tour as seen so it doesn't cover page content.
        page.add_init_script(
            "try { localStorage.setItem('fts.tutorial.seen', '1'); } catch (e) {}"
        )
        for name, url in PAGES:
            page.goto(url, wait_until="networkidle", timeout=60000)
            # Skip the onboarding tour / dismiss popups so the UI itself is visible.
            for sel in ("text=Skip", "text=SKIP"):
                try:
                    btn = page.locator(sel).first
                    if btn.is_visible(timeout=800):
                        btn.click(timeout=2000)
                        page.wait_for_timeout(400)
                        break
                except PlaywrightError as e:
                    print(f"{name}: skip-button probe {sel!r} failed: {e}", file=sys.stderr)
                    continue
            page.wait_for_timeout(1500)
            # A sticky header is painted again mid-page in a full-page capture: pin everything to the flow.
            page.evaluate("""() => { for (const el of document.querySelectorAll('body *')) {
                const pos = getComputedStyle(el).position;
                if (pos === 'sticky' || pos === 'fixed') el.style.position = 'static'; } }""")
            page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
            if name in HERO:
                page.evaluate("window.scrollTo(0, 0)")
                page.screenshot(path=str(OUT / f"hero_{name}.png"), clip={"x": 0, "y": 0, "width": 1440, "height": 1000})
            print(name, "ok")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
