#!/usr/bin/env python3
"""CSS-level breakpoint tests via Playwright (independent of the OpenClaw browser tool).

Verifies that at each documented viewport width, the @media rules defined
in app.css behave correctly: a `(max-width: Xpx)` rule MUST match when
viewport <= X, MUST NOT match when viewport > X. This catches regressions
where someone removes a @media rule or breaks a selector without noticing.

Run after `systemctl restart finetune-studio` to ensure the latest CSS
is loaded.

Usage:
  python3 tests/test_breakpoints.py
"""
import re
import sys
from pathlib import Path

# Add repo root to path so we can import finetune_studio if needed
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from playwright.sync_api import sync_playwright

BASE = "http://fan-dragon:7860/projects/264f8765/training?bust=bp"

# @media rules ACTUALLY DEFINED in app.css.
# Keep this list in sync with the @media blocks in app.css.
# (max-width: X) matches when viewport <= X.
DEFINED_RULES = [
    "(max-width: 480px)",
    "(max-width: 700px)",
    "(max-width: 900px)",
    "(max-width: 1100px)",
    "(max-width: 1280px)",
]


def parse_width(rule: str) -> int:
    m = re.search(r"(\d+)\s*px", rule)
    return int(m.group(1)) if m else 0


def check_at_width(page, width: int):
    """Set viewport and verify each DEFINED_RULE matches iff width <= its threshold."""
    page.set_viewport_size({"width": width, "height": 800})
    page.wait_for_load_state("networkidle", timeout=5000)
    actual_w = page.evaluate("window.innerWidth")
    failures = []
    matched = []
    for rule in DEFINED_RULES:
        threshold = parse_width(rule)
        should_match = actual_w <= threshold
        actually_matches = page.evaluate(f"window.matchMedia({rule!r}).matches")
        if should_match and not actually_matches:
            failures.append(f"  {rule}: viewport={actual_w} SHOULD match (<= {threshold}) but DOES NOT")
        elif not should_match and actually_matches:
            failures.append(f"  {rule}: viewport={actual_w} should NOT match (> {threshold}) but DOES")
        if actually_matches:
            matched.append(rule)
    return failures, actual_w, matched


def main() -> int:
    print(f"== CSS breakpoint tests against {BASE} ==")
    test_widths = [320, 375, 414, 480, 600, 768, 1024, 1100, 1200, 1280, 1440, 1920]
    all_ok = True
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            ctx = browser.new_context()
            page = ctx.new_page()
            page.goto(BASE, wait_until="domcontentloaded")
            for width in test_widths:
                failures, actual_w, matched = check_at_width(page, width)
                if failures:
                    all_ok = False
                    print(f"\n[FAIL] width={width} (actual viewport={actual_w})")
                    for f in failures:
                        print(f)
                else:
                    summary = ", ".join(str(parse_width(r)) for r in matched)
                    print(f"[OK]   width={width:>4} -> {len(matched)} rules active: {summary}")
            ctx.close()
        finally:
            browser.close()
    print()
    print("RESULT:", "ALL PASS" if all_ok else "FAILURES")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
