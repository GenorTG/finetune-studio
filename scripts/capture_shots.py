"""Capture fresh full-page screenshots of the running WebUI (fan-dragon :7860).

Usage: .venv/bin/python scripts/capture_shots.py
Writes docs/screenshots/NN_*.png at 1440px viewport, full-page.
"""
from __future__ import annotations

import pathlib
import sys

from playwright.sync_api import sync_playwright

BASE = "http://fan-dragon:7860"
PID = "58d4e331"
OUT = pathlib.Path("docs/screenshots")

PAGES: list[tuple[str, str]] = [
    ("01_dashboard", f"{BASE}/"),
    ("02_project", f"{BASE}/projects/{PID}"),
    ("03_data_prep", f"{BASE}/projects/{PID}/data-prep"),
    ("04_rag", f"{BASE}/projects/{PID}/rag"),
    ("05_models", f"{BASE}/projects/{PID}/models"),
    ("06_training", f"{BASE}/projects/{PID}/training"),
    ("07_benchmarks", f"{BASE}/projects/{PID}/benchmarks"),
    ("08_export", f"{BASE}/projects/{PID}/export"),
    ("09_testing", f"{BASE}/projects/{PID}/testing"),
    ("10_settings", f"{BASE}/projects/{PID}/settings"),
    ("11_chat", f"{BASE}/projects/{PID}/chat"),
]


def main() -> int:
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
                except Exception:
                    continue
            page.wait_for_timeout(1500)
            page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
            print(name, "ok")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
