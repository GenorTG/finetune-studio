"""Browser E2E: connect an API provider as the helper on Settings, like a user would.

    FTS_ALLOW_LIVE_E2E=1 FTS_E2E_API_KEY=... [FTS_E2E_API_URL=...] \\
        .venv/bin/python tests/e2e_helper_settings.py --model deepseek-v4-flash [--preset opencode-go] [--local]

Drives /settings: seat "API provider", pick the preset, type the key, Fetch model list, choose the model,
Test connection, Save, reload and re-read the card (the key must show only as "(set)"). ``--local`` seats the
local GGUF again instead. It changes the LIVE helper seat and stores the key in the local provider DB (never
in the repo); the key comes from the environment and is never printed. Shots: ``.tmp/qa-shots/helper/``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

REPO = Path(__file__).resolve().parents[1]
BASE = os.environ.get("FTS_BASE", "http://127.0.0.1:7860").rstrip("/")
SHOTS = Path(os.environ.get("FTS_QA_SHOTS", REPO / ".tmp" / "qa-shots")) / "helper"
CHECKS: list[tuple[bool, str]] = []


def check(ok: bool, what: str) -> None:
    CHECKS.append((ok, what))
    print(("PASS " if ok else "FAIL ") + what, flush=True)


def status(page: Page) -> str:
    return page.locator("#helper-status").inner_text()


def wait_status(page: Page, needle: str, timeout_ms: int = 90_000) -> str:
    page.wait_for_function("n => document.getElementById('helper-status').textContent.includes(n)", arg=needle,
                           timeout=timeout_ms)
    return status(page)


def run(args: argparse.Namespace) -> int:
    key = os.environ.get("FTS_E2E_API_KEY", "")
    if not args.local and not key:
        print("set FTS_E2E_API_KEY (the key is read from the environment and never printed)", file=sys.stderr)
        return 2
    SHOTS.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 1500})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(f"{BASE}/settings")
        page.wait_for_selector("#helper-card")
        page.wait_for_function("document.getElementById('helper-api-preset').options.length > 0")
        if args.local:
            page.check("#helper-seat-local")
            page.click("#btn-helper-save")
            wait_status(page, "Saved")
            page.screenshot(path=str(SHOTS / "settings-local.png"))
            check(page.is_checked("#helper-seat-local"), "local GGUF seated again")
        else:
            page.check("#helper-seat-api")
            check(page.is_visible("#helper-api-fields"), "API fields appear when the API seat is chosen")
            page.select_option("#helper-api-preset", args.preset)
            url = os.environ.get("FTS_E2E_API_URL") or page.input_value("#helper-api-url")
            page.fill("#helper-api-url", url)
            check(bool(url), f"preset filled the base URL ({url})")
            page.fill("#helper-api-key", key)
            page.click("#btn-helper-models")
            msg = wait_status(page, "models available")
            check(True, f"model list fetched: {msg}")
            offered = page.eval_on_selector_all("#helper-api-model-list option", "els => els.map(e => e.value)")
            check(args.model in offered, f"{args.model} is offered by the provider")
            page.fill("#helper-api-model", args.model)
            if args.effort:
                page.select_option("#helper-api-effort", args.effort)
            page.click("#btn-helper-test")
            msg = wait_status(page, "Works")
            check("Works" in msg, f"Test connection: {msg}")
            page.click("#btn-helper-save")
            wait_status(page, "Saved")
            page.screenshot(path=str(SHOTS / "settings-api-saved.png"))
            page.reload()
            page.wait_for_function("document.getElementById('helper-key-state').textContent !== '(not set)'")
            check(page.is_checked("#helper-seat-api"), "API seat survives a reload")
            check(page.input_value("#helper-api-model") == args.model, "model survives a reload")
            check(page.input_value("#helper-api-key") == "", "key field is empty after reload (write-only)")
            check(page.inner_text("#helper-key-state") == "(set)", "key shows only as (set)")
            check(key not in page.content(), "the key appears nowhere in the page HTML")
        check(not errors, f"no JS/console errors {errors[:3]}")
        browser.close()
    failed = [w for ok, w in CHECKS if not ok]
    print(f"RESULT: {len(CHECKS) - len(failed)}/{len(CHECKS)} passed" + (f"; FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--preset", default="opencode-go")
    ap.add_argument("--effort", default="")
    ap.add_argument("--local", action="store_true", help="seat the local GGUF instead")
    args = ap.parse_args()
    if os.environ.get("FTS_ALLOW_LIVE_E2E") != "1":
        print("Refusing: set FTS_ALLOW_LIVE_E2E=1 (this changes the live helper seat).", file=sys.stderr)
        return 2
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
