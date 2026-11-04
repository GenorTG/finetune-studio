"""Capture the README screenshots from a running WebUI (FTS_BASE, default http://127.0.0.1:7860).

Use a project that holds synthetic demo data only. A signed-in app needs a session cookie:
    FTS_SESSION=<session token> FTS_DEMO_PID=<project id> .venv/bin/python scripts/capture_shots.py
FTS_SHOTS=04_files,06_brain limits the run to those names (the dashboard and Settings show every project name and
the host's addresses: retake them only from an instance that holds nothing else; this machine's addresses are scrubbed,
and FTS_SHOT_HOME names that instance's HOME). Pages render in the light theme;
the browser runs without a GPU so it never touches a card another process owns.
Writes docs/screenshots/NN_*.png (1440px wide, full-page) and hero_NN_*.png (one 1440x1000 viewport).
"""
from __future__ import annotations

import json
import os
import pathlib
import socket
import subprocess
import sys

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

BASE = os.environ.get("FTS_BASE", "http://127.0.0.1:7860").rstrip("/")
PID = os.environ.get("FTS_DEMO_PID", "")
SESSION = os.environ.get("FTS_SESSION", "")
ONLY = {n.strip() for n in os.environ.get("FTS_SHOTS", "").split(",") if n.strip()}
OUT = pathlib.Path("docs/screenshots")
# Full-page shots of a big project (hundreds of files) run to ~10,000 px and megabytes; the top tells the story.
MAX_HEIGHT = int(os.environ.get("FTS_SHOT_MAX_HEIGHT", "3200"))


def _machine_addresses() -> list[tuple[str, str]]:
    """This machine's LAN and Tailscale addresses and Tailscale name -> documentation placeholders (RFC 5737 /
    CGNAT example / example tailnet). Settings → Hosting lists them; a public image must not."""
    out: list[tuple[str, str]] = []
    try:
        ip = subprocess.run(["ip", "-o", "-4", "addr", "show"], capture_output=True, text=True, timeout=5,
                            check=False).stdout
        for n, addr in enumerate(sorted({line.split()[3].split("/")[0] for line in ip.splitlines()} - {"127.0.0.1"})):
            if not addr.startswith("100."):
                out.append((addr, f"192.0.2.{10 + n}"))
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    try:
        ts = json.loads(subprocess.run(["tailscale", "status", "--json"], capture_output=True, text=True,
                                       timeout=5, check=False).stdout or "{}").get("Self") or {}
        dns = str(ts.get("DNSName") or "").rstrip(".")
        if dns:
            out.append((dns, "host.example-tailnet.ts.net"))
        for n, addr in enumerate(a for a in ts.get("TailscaleIPs") or [] if "." in a):
            out.append((addr, f"100.64.0.{10 + n}"))
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return out


# What must not reach a public image: the app's and this machine's home directory, the machine's addresses and
# hostname, and a dev build's branch label. FTS_SHOT_HOME = the HOME the app instance runs with (a demo instance
# keeps its own), so its paths read "~/.finetune-studio/..." instead of the demo folder.
SCRUB = [(h, "~") for h in dict.fromkeys(filter(None, [os.environ.get("FTS_SHOT_HOME", ""), str(pathlib.Path.home())]))]
SCRUB += sorted(_machine_addresses(), key=lambda kv: -len(kv[0])) + [(socket.gethostname(), "host")]
VERSION = pathlib.Path("VERSION").read_text().strip() if pathlib.Path("VERSION").exists() else ""

PAGES: list[tuple[str, str]] = [
    ("01_dashboard", f"{BASE}/"),
    ("02_project", f"{BASE}/projects/{PID}"),
    ("03_quick_work", f"{BASE}/projects/{PID}/wizard"),
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
        browser = p.chromium.launch(args=["--disable-gpu"])
        context = browser.new_context(viewport={"width": 1440, "height": 1000}, color_scheme="light")
        if SESSION:
            context.add_cookies([{"name": "fts_session", "value": SESSION, "url": BASE}])
        page = context.new_page()
        # Mark the onboarding tour as seen so it doesn't cover page content; README images use the light theme.
        page.add_init_script(
            "try { localStorage.setItem('fts.tutorial.seen', '1'); localStorage.setItem('fts-theme', 'light'); }"
            " catch (e) {}"
        )
        for name, url in PAGES:
            if ONLY and name not in ONLY:
                continue
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
            page.evaluate(r"""([scrub, version]) => {
                const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
                for (let n = walk.nextNode(); n; n = walk.nextNode()) {
                    let t = n.nodeValue;
                    for (const [from, to] of scrub) if (from) t = t.split(from).join(to);
                    if (version) t = t.replace(/v\d+\.\d+\.\d+\+[\w.\/-]+/g, 'v' + version);
                    if (t !== n.nodeValue) n.nodeValue = t;
                }
                // Form fields hold paths and addresses too (Settings): scrub their values and placeholders.
                for (const el of document.querySelectorAll('input, textarea')) {
                    for (const prop of ['value', 'placeholder']) {
                        let t = el[prop] || '';
                        for (const [from, to] of scrub) if (from) t = t.split(from).join(to);
                        if (t !== (el[prop] || '')) el[prop] = t;
                    }
                } }""", [SCRUB, VERSION])
            height = min(page.evaluate("document.documentElement.scrollHeight"), MAX_HEIGHT)
            page.screenshot(path=str(OUT / f"{name}.png"), full_page=True,
                            clip={"x": 0, "y": 0, "width": 1440, "height": height})
            if name in HERO:
                page.evaluate("window.scrollTo(0, 0)")
                page.screenshot(path=str(OUT / f"hero_{name}.png"), clip={"x": 0, "y": 0, "width": 1440, "height": 1000})
            print(name, "ok")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
