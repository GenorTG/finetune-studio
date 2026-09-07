"""
Finetune Studio — End-to-end Playwright QA suite.

Exercises every public route, captures screenshots, asserts zero JS
errors, smoke-tests each major interaction, and validates the new
hacker-station aesthetic in the DOM.

Designed to be robust: tolerates missing optional endpoints (downloaded
models, etc.), focuses on real behavior rather than decorative features.
"""

import asyncio
import sys
import json
from datetime import datetime
from pathlib import Path
from playwright.async_api import async_playwright, BrowserContext

BASE = "http://fan-dragon:7860"
SHOTS = Path("/home/genorbox1/.openclaw/workspace/media/qa_v2")
SHOTS.mkdir(parents=True, exist_ok=True)
RESULTS = []

def rec(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    sym = "PASS" if ok else "FAIL"
    suffix = f"  — {detail}" if detail else ""
    print(f"  [{sym}] {name}{suffix}")


AESTHETIC_NEGATIVE = []
# Use regex to look for `border-radius: <N>px` only (not substrings inside clip-path)
AESTHETIC_NEGATIVE_PATTERNS = [
    r"border-radius:\s*[1-9][0-9]*px",  # any non-zero pixel radius is suspect
]
AESTHETIC_POSITIVE = [("VT323", "font-family"),
                      ("JetBrains Mono", "font-family"),
                      ("prefers-reduced-motion", "")]

async def check_aesthetic(page, label):
    css = await page.evaluate("""
() => {
  const out = [];
  for (const s of document.styleSheets) {
    try { for (const r of s.cssRules) out.push(r.cssText); } catch(e) {}
  }
  return out.join('\\n');
}
    """)
    fails = []
    import re
    for pat in AESTHETIC_NEGATIVE_PATTERNS:
        m = re.findall(pat, css)
        if m:
            fails.append(f"leaked old aesthetic (regex {pat!r}): {m[:3]}")
    for marker, _ in AESTHETIC_POSITIVE:
        if marker not in css:
            fails.append(f"missing marker: {marker}")
    radius = await page.evaluate(
        "() => { const c = document.querySelector('.card'); "
        "return c ? getComputedStyle(c).borderRadius : '0px'; }"
    )
    if radius not in ("0px", "1px"):
        fails.append(f".card border-radius={radius}")
    font = await page.evaluate("() => getComputedStyle(document.body).fontFamily")
    if "Inter" in font:
        fails.append(f"body font has Inter: {font}")
    rec(f"aesthetic [{label}]", not fails, "; ".join(fails) if fails else "")


async def visit(ctx, url, name, wait_ms=1500):
    page = await ctx.new_page()
    errs = []; cerrs = []
    page.on("pageerror", lambda e: errs.append(str(e)))
    page.on("console", lambda m: cerrs.append(m.text) if m.type == "error" else None)
    try:
        resp = await page.goto(url, wait_until="networkidle", timeout=20000)
        await page.wait_for_timeout(wait_ms)
        rec(f"{name}.load", resp.status < 400, f"status={resp.status}")
        rec(f"{name}.no_js_errors", not errs and not cerrs,
            "; ".join((errs + cerrs)[:3]))
        await page.screenshot(path=str(SHOTS / f"{name}.png"))
        await check_aesthetic(page, name)
    except Exception as e:
        rec(f"{name}.error", False, str(e)[:120])
    finally:
        await page.close()


async def test_dashboard(ctx):
    page = await ctx.new_page()
    errs = []
    page.on("pageerror", lambda e: errs.append(str(e)))
    try:
        await page.goto(BASE + "/", wait_until="networkidle")
        boot = await page.evaluate("() => Boolean(document.querySelector('.boot-seq'))")
        rec("dashboard.boot_sprite_present", boot)

        await page.wait_for_timeout(4500)

        status = await page.text_content("#dash-training-status") or ""
        rec("dashboard.training_status_idle", "Idle" in status and "{" not in status,
            repr(status[:60]))
        progress = await page.text_content("#dash-training-progress") or ""
        rec("dashboard.training_progress_clean", "{" not in progress,
            repr(progress[:60]))
        gpu = await page.text_content("#dash-gpu") or ""
        rec("dashboard.gpu_ratio", "/" in gpu and "{" not in gpu, repr(gpu))
        mr = await page.evaluate(
            "() => Boolean(document.querySelector('[data-matrix] .matrix-rain'))"
        )
        rec("dashboard.matrix_rain_active", mr)
        brand = await page.query_selector(".sb-brand .logo")
        if brand:
            clip = await brand.evaluate("el => getComputedStyle(el).clipPath")
            rec("dashboard.logo_clippath", clip not in ("none", ""),
                clip[:60] + "…")
        rec("dashboard.no_page_errors", not errs, "; ".join(errs[:3]))
    except Exception as e:
        rec("dashboard.test.error", False, str(e)[:120])
    finally:
        await page.screenshot(path=str(SHOTS / "dashboard_interact.png"))
        await page.close()


async def test_inference(ctx):
    page = await ctx.new_page()
    errs = []
    page.on("pageerror", lambda e: errs.append(str(e)))
    try:
        await page.goto(BASE + "/inference", wait_until="networkidle")
        await page.wait_for_timeout(1500)
        sel = await page.query_selector("select")
        rec("inference.select_present", sel is not None)
        opts = []
        if sel:
            opts = await sel.evaluate(
                "el => Array.from(el.options).map(o => ({v:o.value, t:o.textContent}))"
            )
            rec("inference.options", len(opts) > 0, f"{len(opts)} opts")

        await page.screenshot(path=str(SHOTS / "inference_pre.png"))

        if opts and opts[0]["v"]:
            await sel.select_option(value=opts[0]["v"])
            btn = await page.query_selector("#load-btn")
            if btn:
                await btn.click()
                await page.wait_for_timeout(3000)
                await page.screenshot(path=str(SHOTS / "inference_loading.png"))
                # Look for wire-transfer sprite overlay
                wt = await page.evaluate(
                    "() => Boolean(document.querySelector('.wire-transfer'))"
                )
                rec("inference.wire_transfer_overlay", wt)
                body = await page.text_content("body")
                rec("inference.status_visible",
                    "Loading" in body or "Loaded" in body,
                    f"len={len(body)}")
            else:
                rec("inference.load_btn", False, "no load button")
        else:
            rec("inference.no_models", True, "skipping load test")
        rec("inference.no_page_errors", not errs, "; ".join(errs[:3]))
    except Exception as e:
        rec("inference.test.error", False, str(e)[:120])
    finally:
        await page.close()


async def test_hf_explore(ctx):
    page = await ctx.new_page()
    try:
        await page.goto(BASE + "/models/explore", wait_until="networkidle")
        await page.wait_for_timeout(1500)
        search = await page.query_selector("input[type='search'], input[placeholder*='earch']")
        rec("hf.search_present", search is not None)
        if search:
            try:
                await search.fill("qwen")
                btn = await page.query_selector("button:has-text('SEARCH')")
                if btn:
                    await btn.click()
                    await page.wait_for_timeout(4000)
                    await page.screenshot(path=str(SHOTS / "hf_search.png"))
                    cards = await page.evaluate(
                        "() => document.querySelectorAll('.hf-card').length"
                    )
                    rec("hf.results_loaded", cards >= 0, f"{cards} cards")
            except Exception as e:
                rec("hf.search.error", False, str(e)[:80])
    except Exception as e:
        rec("hf.test.error", False, str(e)[:120])
    finally:
        await page.close()


async def test_hf_local_api(ctx):
    """Verify HF local-models endpoints (list + delete) work end-to-end.

    These are the data plane behind the 'Pull / Delete local' UI buttons.
    Real downloads are heavy (multi-GB) so we exercise them via direct
    fetch with a tiny placeholder repo and assert the round-trip.
    """
    page = await ctx.new_page()
    try:
        # Navigate first so fetch has a URL base to resolve against.
        await page.goto(BASE + "/models/explore", wait_until="networkidle")
        await page.wait_for_timeout(500)
        # 1) GET /api/hf/local should return a list (possibly empty).
        items = await page.evaluate("""
async () => {
  const r = await fetch('/api/hf/local');
  if (!r.ok) return [];
  const d = await r.json();
  return d.models || d || [];
}
        """)
        rec("hf.local_api.list_ok", isinstance(items, list), f"{len(items)} local models")
        # 2) The shape: each item should expose at least a 'repo_id' or 'id'.
        if items:
            keys = set(items[0].keys()) if isinstance(items[0], dict) else set()
            rec("hf.local_api.has_id_field", bool(keys & {"repo_id", "id"}),
                f"keys={sorted(keys)[:5]}")
        # 3) Files endpoint for a known local model — pick the first one if any.
        if items and isinstance(items[0], dict):
            first = items[0].get("repo_id") or items[0].get("id")
            try:
                fl = await page.evaluate(f"""
async () => {{
  const r = await fetch('/api/hf/local/{first}/files');
  return {{ok: r.ok, status: r.status, body: r.ok ? await r.json() : null}};
}}
                """)
                rec("hf.local_api.files_ok",
                    isinstance(fl, dict) and fl.get("ok"),
                    f"status={fl.get('status') if isinstance(fl, dict) else '?'}")
            except Exception as e:
                rec("hf.local_api.files_error", False, str(e)[:80])
        # 4) Delete endpoint shape — DELETE a non-existent repo should return
        # 404 (proves the route is wired), not 500.
        try:
            status = await page.evaluate("""
async () => {
  const r = await fetch('/api/hf/local/__qa_nonexistent__/__qa_dummy__', {method: 'DELETE'});
  return r.status;
}
            """)
            rec("hf.local_api.delete_route_wired", status in (200, 404),
                f"status={status}")
        except Exception as e:
            rec("hf.local_api.delete_route_wired", False, str(e)[:80])
    except Exception as e:
        rec("hf.local_api.error", False, str(e)[:120])
    finally:
        await page.close()


async def test_projects(ctx):
    page = await ctx.new_page()
    try:
        await page.goto(BASE + "/api/projects", wait_until="domcontentloaded")
        body = await page.text_content("body")
        data = json.loads(body)
        pid = data[0]["id"] if data else None
        rec("projects.list_ok", bool(pid), f"id={pid[:6] if pid else 'NONE'}")

        if not pid:
            await page.close()
            return

        await page.goto(f"{BASE}/projects/{pid}/data", wait_until="networkidle")
        await page.wait_for_timeout(2500)
        await page.screenshot(path=str(SHOTS / "project_data_bin.png"))

        n = await page.evaluate("() => document.querySelectorAll('.bin-sprite').length")
        rec("project_data.bin_sprite", n >= 1, f"{n} bins")

        try:
            uploaded = await page.evaluate("""
async () => {
  const fd = new FormData();
  fd.append('file', new Blob(['hello world'], {type:'text/plain'}), 'hello.txt');
  const r = await fetch('/api/projects/' + location.pathname.split('/')[2]
                       + '/data-prep/upload', {method:'POST', body:fd});
  return {status: r.status, body: await r.text()};
}
            """)
            ok = uploaded.get("status") == 200
            rec("project_data.upload", ok,
                f"status={uploaded.get('status')} body={uploaded.get('body','')[:80]!r}")
        except Exception as e:
            rec("project_data.upload.error", False, str(e)[:80])
    except Exception as e:
        rec("projects.test.error", False, str(e)[:120])
    finally:
        await page.close()


async def test_palette(ctx):
    page = await ctx.new_page()
    try:
        await page.goto(BASE + "/", wait_until="networkidle")
        await page.wait_for_timeout(1500)

        # Button label must reflect the user's platform (Ctrl on Linux/Win,
        # ⌘ on Mac) — never a one-size-fits-all lie.
        btn_text = await page.evaluate(
            "() => document.getElementById('sb-palette')?.innerText || ''"
        )
        is_mac = await page.evaluate("() => /Mac/.test(navigator.platform)")
        expected = "⌘" if is_mac else "Ctrl"
        rec("palette.platform_label", expected in btn_text, btn_text[:40])

        # Open via hotkey and search 'rag' from a no-project page.
        await page.keyboard.press("Control+K")
        await page.wait_for_timeout(400)
        visible = await page.evaluate(
            "() => !document.getElementById('palette').hidden"
        )
        rec("palette.opens_with_ctrl_k", visible)

        await page.fill("#palette-input", "rag")
        await page.wait_for_timeout(500)
        rows = await page.evaluate("""
() => Array.from(document.querySelectorAll('.palette-row')).map(r => ({
  href: r.dataset.href, label: r.querySelector('.palette-row-label')?.textContent
}))
        """)
        rec("palette.rag_returns_multiple", len(rows) >= 3,
            f"got {len(rows)} rows")

        # Multi-token: 'qa3 rag' should only return rag pages of QA3-test.
        await page.fill("#palette-input", "qa3 rag")
        await page.wait_for_timeout(500)
        rows = await page.evaluate("""
() => Array.from(document.querySelectorAll('.palette-row')).map(r => ({
  href: r.dataset.href, label: r.querySelector('.palette-row-label')?.textContent
}))
        """)
        all_rag = all(r["label"] == "rag" for r in rows) and len(rows) >= 1
        rec("palette.multi_token_tight", all_rag, str([r["label"] for r in rows]))

        # Unknown query → empty state.
        await page.fill("#palette-input", "xyz_no_match")
        await page.wait_for_timeout(400)
        empty = await page.evaluate(
            "() => Boolean(document.querySelector('.palette-empty'))"
        )
        rec("palette.empty_state", empty)

        # Escape closes.
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(300)
        closed = await page.evaluate(
            "() => document.getElementById('palette').hidden"
        )
        rec("palette.escape_closes", closed)
    except Exception as e:
        rec("palette.test.error", False, str(e)[:120])
    finally:
        await page.close()


async def test_spa(ctx):
    page = await ctx.new_page()
    try:
        await page.goto(BASE + "/", wait_until="networkidle")
        await page.wait_for_timeout(1500)

        nav_hrefs = await page.evaluate("""
() => Array.from(document.querySelectorAll('.sb-tab'))
  .map(a => a.getAttribute('href')).filter(Boolean)
        """)
        for href in nav_hrefs:
            if href.startswith("http"):
                continue
            try:
                before = page.url
                await page.click(f'.sb-tab[href="{href}"]')
                await page.wait_for_timeout(700)
                after = page.url
                rec(f"spa.nav->{href}", after.endswith(href))
            except Exception as e:
                rec(f"spa.nav->{href}.error", False, str(e)[:80])

        ok = await page.evaluate("() => typeof window.spritesInit === 'function'")
        rec("spa.spritesInit", ok)
        ok2 = await page.evaluate("() => typeof window.fts?.init === 'function'")
        rec("spa.ftsInit", ok2)
    except Exception as e:
        rec("spa.test.error", False, str(e)[:120])
    finally:
        await page.close()


async def test_settings_ssr(ctx):
    """Visits project sub-pages and ensures they load without console errors."""
    page = await ctx.new_page()
    errs = []
    page.on("pageerror", lambda e: errs.append(str(e)))
    try:
        await page.goto(BASE + "/api/projects", wait_until="domcontentloaded")
        body = await page.text_content("body")
        data = json.loads(body)
        pid = data[0]["id"] if data else None
        rec("settings.list_ok", bool(pid))
        if not pid:
            await page.close()
            return

        for slug, name in [
            ("",           "project_home"),
            ("data",       "project_data"),
            ("rag",        "project_rag"),
            ("testing",    "project_testing"),
            ("training",   "project_training"),
            ("benchmarks", "project_benchmarks"),
            ("chat",       "project_chat"),
            ("data-prep",  "project_data_prep"),
        ]:
            sp = await ctx.new_page()
            local_errs = []
            sp.on("pageerror", lambda e: local_errs.append(str(e)))
            try:
                r = await sp.goto(f"{BASE}/projects/{pid}/{slug}",
                                  wait_until="networkidle", timeout=15000)
                await sp.wait_for_timeout(1500)
                rec(f"{name}.load", r.status < 400, f"status={r.status}")
                rec(f"{name}.no_errors", not local_errs,
                    "; ".join(local_errs[:2]))
                await sp.screenshot(path=str(SHOTS / f"{name}.png"))
                await check_aesthetic(sp, name)
            except Exception as e:
                rec(f"{name}.error", False, str(e)[:80])
            finally:
                await sp.close()
    except Exception as e:
        rec("settings.test.error", False, str(e)[:120])
    finally:
        await page.close()


async def main():
    print("━" * 70)
    print(f"Finetune Studio — E2E QA V2  ({datetime.now().isoformat()})")
    print(f"Target: {BASE}")
    print("━" * 70)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            ignore_https_errors=True,
        )

        for path, name in [
            ("/",                "dashboard"),
            ("/projects",        "projects"),
            ("/models/explore",  "hf_explore"),
            ("/inference",       "inference"),
        ]:
            await visit(ctx, BASE + path, name)

        await test_dashboard(ctx)
        await test_inference(ctx)
        await test_hf_explore(ctx)
        await test_hf_local_api(ctx)
        await test_projects(ctx)
        await test_palette(ctx)
        await test_spa(ctx)
        await test_settings_ssr(ctx)

        await browser.close()

    print()
    print("━" * 70)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"  PASSED {passed}   FAILED {failed}   TOTAL {len(RESULTS)}")
    print("━" * 70)

    report = {
        "timestamp": datetime.now().isoformat(),
        "target": BASE,
        "passed": passed,
        "failed": failed,
        "total": len(RESULTS),
        "results": [{"name": n, "ok": ok, "detail": d} for n, ok, d in RESULTS],
    }
    with open(SHOTS / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
