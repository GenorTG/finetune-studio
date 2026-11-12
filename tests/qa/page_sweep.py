"""Load every registered page of the live app, then walk them all inside one tab through the SPA, twice.

    FTS_SESSION=<session token> .venv/bin/python tests/qa/page_sweep.py <project id> [--base http://127.0.0.1:7860]

Read-only (nothing is clicked except navigation). Reports per page: HTTP status, the page heading, JS errors
(uncaught exceptions and console errors) and failed requests (4xx/5xx). The SPA laps catch page scripts that only
break when a page is entered a second time without a full load (the Projects page's duplicate-const bug).
Exit code 1 when any page has a problem. Get a session with `.tmp/agent/session.py create` (and revoke it after).
"""
from __future__ import annotations

import argparse
import os
import sys

from playwright.sync_api import sync_playwright

from finetune_studio.guide.registry import PAGES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pid")
    ap.add_argument("--base", default="http://127.0.0.1:7860")
    a = ap.parse_args()
    token = os.environ.get("FTS_SESSION", "")
    if not token:
        print("set FTS_SESSION to a session token", file=sys.stderr)
        return 2
    urls = [(name, page.url.replace("{pid}", a.pid)) for name, page in PAGES.items()]
    problems = 0
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1366, "height": 900})
        ctx.add_cookies([{"name": "fts_session", "value": token, "url": a.base}])

        def watch(page, errors: list[str], failed: list[str]) -> None:
            page.on("pageerror", lambda e: errors.append(str(e)[:200]))
            page.on("console", lambda m: errors.append("console: " + m.text[:200]) if m.type == "error" else None)
            page.on("response", lambda r: failed.append(f"{r.status} {r.url.replace(a.base, '')}")
                    if r.status >= 400 else None)

        def heading(page) -> str:
            return page.locator("h1").first.inner_text(timeout=3000).strip()[:60] if page.locator("h1").count() else ""

        print("== full loads")
        for name, url in urls:
            page = ctx.new_page()
            errors: list[str] = []
            failed: list[str] = []
            watch(page, errors, failed)
            resp = page.goto(a.base + url, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(1500)
            ok = bool(resp) and resp.status == 200 and not errors and not failed
            problems += not ok
            print("OK " if ok else "!! ", f"{name:16s} {resp.status if resp else '-'} {url:34s} {heading(page)!r}")
            for line in errors[:4] + failed[:4]:
                print("      ", line)
            page.close()

        print("== SPA navigation, two laps in one tab")
        page = ctx.new_page()
        errors, failed = [], []
        watch(page, errors, failed)
        page.goto(a.base + "/", wait_until="networkidle")
        for lap in (1, 2):
            for name, url in urls:
                errors.clear()
                failed.clear()
                page.evaluate("u => window.ftsSPA.navigate(u)", url)
                page.wait_for_timeout(1800)
                ok = not errors and not failed and page.url.endswith(url)
                problems += not ok
                if not ok:
                    print("!! ", f"lap {lap} {name:16s} {url}")
                    for line in errors[:4] + failed[:4]:
                        print("      ", line)
        print(f"   {2 * len(urls)} SPA hops done")
        browser.close()
    print("pages with problems:", problems)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
