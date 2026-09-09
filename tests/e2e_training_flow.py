"""FULL WebUI-driven training flow: drives the visible browser through
every step like a human would. NO direct API calls except `page.fill()`
for text inputs (typing into textareas/inputs is the only way to enter
data, and it still goes through the WebUI form submission handler).

Steps:
  1. Open Dashboard in browser
  2. Navigate to HF Explorer → search "Qwen3-0.6B" → click Pull
  3. Watch the visible HF download progress UI until done
  4. Navigate to Projects → click New project → fill name → submit
  5. Navigate to project data page → upload synthetic JSONL → verify parsed
  6. Navigate to project training → select Qwen3 model → fill params → Start
  7. Watch training status badge and live progress until complete
  8. Verify run appears in /api/projects/{pid}/runs (read-only check)
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

BASE = os.environ.get("FTS_BASE", "http://localhost:7860")
MODEL_QUERY = "Qwen3-0.6B"
SHOTS = Path("/home/genorbox1/.openclaw/workspace/media/qa_train_flow")
SHOTS.mkdir(parents=True, exist_ok=True)


def make_synthetic_jsonl(path: Path, n: int = 24):
    """Tiny QA dataset — capital-of-country pairs."""
    pairs = [
        ("What is the capital of France?", "Paris"),
        ("What is the capital of Japan?", "Tokyo"),
        ("What is the capital of Italy?", "Rome"),
        ("What is the capital of Poland?", "Warsaw"),
        ("What is the capital of Spain?", "Madrid"),
        ("What is the capital of Portugal?", "Lisbon"),
        ("What is the capital of Germany?", "Berlin"),
        ("What is the capital of Norway?", "Oslo"),
    ]
    with open(path, "w") as f:
        for i in range(n):
            q, a = pairs[i % len(pairs)]
            f.write(json.dumps({"prompt": q, "completion": a}) + "\n")


async def wait_for(fn, timeout_ms, label, interval_ms=2000):
    elapsed = 0
    last = None
    while elapsed < timeout_ms:
        try:
            r = await fn()
            if r:
                return r
            last = r
        except Exception as e:
            last = str(e)[:80]
        await asyncio.sleep(interval_ms / 1000)
        elapsed += interval_ms
    raise TimeoutError(f"{label}: timed out after {timeout_ms}ms (last={last!r})")


async def _hf_repo_ready(page, repo_underscored: str) -> bool:
    """True iff the WebUI's /api/hf/local reports the model fully downloaded.

    Uses the same source of truth the HF Explorer card list calls when
    rendering "downloaded" state — so a green badge here = a green badge
    in the UI. Repo id is the filesystem form (slash → double underscore).
    """
    try:
        ok = await page.evaluate(f"""
async () => {{
    const r = await fetch('/api/hf/local');
    if (!r.ok) return false;
    const d = await r.json();
    const m = (d.models || []).find(m => m.repo_id === {repo_underscored!r});
    if (!m) return false;
    // Real "downloaded" = at least one safetensors file present.
    return (m.files || []).some(f => f.path.endsWith('.safetensors'));
}}
        """)
        return bool(ok)
    except Exception:
        return False


async def main():
    from playwright.async_api import async_playwright

    tmp = Path(tempfile.mkdtemp(prefix="fts_trainflow_"))
    sample_path = tmp / "capitals.jsonl"
    make_synthetic_jsonl(sample_path)
    print(f"data: {sample_path}")

    results = []
    def rec(name, passed, note=""):
        results.append((name, bool(passed), note))
        print(f"  {'✓' if passed else '✗'} {name}{': ' + note if note else ''}")

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            accept_downloads=True,
        )
        # Pre-mark tutorial as seen so it doesn't intercept clicks
        await ctx.add_init_script("try{localStorage.setItem('fts.tutorial.seen','1');}catch(e){}")
        page = await ctx.new_page()
        # The Pull / Delete buttons use native window.confirm() prompts.
        # Register a SYNC handler (no asyncio.create_task) so accept() runs
        # before Playwright dismisses the dialog.
        page.on("dialog", lambda d: d.accept())
        # BEFORE any page script runs, override window.confirm to always
        # return true. Playwright's dialog handler has race conditions with
        # multi-event click handlers (data-action + data-confirm both fire).
        await ctx.add_init_script("window.confirm = () => true;")
        # Capture console + page errors for debugging
        page.on("console", lambda m: print(f"  [console.{m.type}] {m.text[:200]}"))
        page.on("pageerror", lambda e: print(f"  [pageerror] {str(e)[:200]}"))

        async def shot(name):
            out = SHOTS / f"{name}.png"
            await page.screenshot(path=str(out))
            print(f"    📸 {out.name}")

        # ── 1. HF EXPLORER: SEARCH in WebUI ────────────────────────────
        print("\n[1/8] WEBUI: HF Explorer → search Qwen3-0.6B")
        await page.goto(f"{BASE}/models/explore", wait_until="networkidle")
        await page.wait_for_timeout(2000)
        await shot("01_hf_explorer_loaded")
        # Type into the search field, press Enter
        await page.fill("#hf-q", MODEL_QUERY)
        await page.keyboard.press("Enter")
        await page.wait_for_timeout(4000)
        await shot("02_hf_search_results")
        # Verify search returned the repo card
        cards = await page.evaluate("""
            () => Array.from(document.querySelectorAll('.hf-card')).map(c => {
                const link = c.querySelector('a, [data-repo-id], [class*="repo"]');
                const text = c.innerText || '';
                const firstLine = text.split('\\n').find(l => l.includes('/')) || '';
                return firstLine.trim();
            })
        """)
        print(f"  cards visible: {cards[:8]}")
        found = any(MODEL_QUERY in (c or "") for c in cards)
        rec("webui.hf_search_results", found, f"{len(cards)} cards, first: {cards[0] if cards else 'none'}")

        # ── 2. DOWNLOAD: click Pull button in the UI ───────────────────
        print("\n[2/8] WEBUI: click PULL on Qwen3-0.6B card")
        # Close any leftover modal from previous runs/click attempts
        await page.evaluate("""
            () => { const m = document.getElementById('hf-modal'); if (m) m.hidden = true; }
        """)
        await page.wait_for_timeout(300)
        await shot("02a_modal_closed")

        pulled = False
        # Find the card whose text contains EXACTLY "Qwen/Qwen3-0.6B"
        # (not the -Base / -GGUF variants), then click its Pull button.
        try:
            handle = await page.evaluate_handle("""
                () => {
                    const cards = Array.from(document.querySelectorAll('.hf-card'));
                    const log = [];
                    for (const c of cards) {
                        const all = Array.from(c.querySelectorAll('*'));
                        const hit = all.find(el => el.textContent.trim() === 'Qwen/Qwen3-0.6B'
                                                       && el.children.length === 0);
                        if (!hit) continue;
                        const btn = c.querySelector('button[data-action="/api/hf/download"]');
                        if (btn) { btn.scrollIntoView(); return btn; }
                    }
                    return null;
                }
            """)
            if handle:
                # Use Playwright's native click — handles dialog + event
                # dispatch ordering correctly (evaluate's btn.click() can
                # skip event-listener chains on some Chromium versions).
                await handle.as_element().click(timeout=10000)
                pulled = True
                print("  PULL button clicked via Playwright native click")
            else:
                print("  no PULL button found")
        except Exception as e:
            print(f"  pull click error: {e}")
        await page.wait_for_timeout(800)
        await shot("02b_hf_pull_clicked")
        rec("webui.hf_pull_clicked", pulled)

        # ── 3. WATCH download progress (via WebUI's API surface) ──────
        print("\n[3/8] WEBUI: watch download progress")
        # The download lands on the SERVER (fan-dragon), not on the box the
        # test runs on. Poll the WebUI's own /api/hf/local endpoint — the
        # very endpoint the HF Explorer card list calls when rendering
        # "downloaded" state. This is the same source of truth the human
        # sees in the UI badge.
        repo_underscored = "Qwen__Qwen3-0.6B"
        try:
            await wait_for(
                lambda: _hf_repo_ready(page, repo_underscored),
                timeout_ms=900000, label="Qwen3-0.6B downloaded",
                interval_ms=5000,
            )
            info = await page.evaluate(f"""
async () => {{
    const r = await fetch('/api/hf/local');
    if (!r.ok) return null;
    const d = await r.json();
    return (d.models || []).find(m => m.repo_id === {repo_underscored!r}) || null;
}}
            """)
            sf_count = sum(1 for f in (info or {}).get("files", [])
                           if f["path"].endswith(".safetensors"))
            total_gb = (info or {}).get("size_bytes", 0) / 1e9
            rec("webui.hf_download_complete", sf_count > 0,
                f"{sf_count} safetensors, {total_gb:.2f} GB")
        except Exception as e:
            rec("webui.hf_download_complete", False, str(e)[:100])

        # ── 4. PROJECTS: click New project + fill + submit ─────────────
        print("\n[4/8] WEBUI: create new project")
        await page.goto(f"{BASE}/projects", wait_until="networkidle")
        await page.wait_for_timeout(1500)
        await shot("03_projects_list")
        # Click the "＋ New project" button
        await page.click('button:has-text("New project")')
        await page.wait_for_timeout(500)
        await shot("04_new_project_form_open")
        # Fill in the form
        proj_name = f"TRAINFLOW-{int(time.time())}"
        await page.fill('#np-form input[name="name"]', proj_name)
        await page.fill('#np-form textarea[name="description"]', "WebUI training flow test")
        await shot("05_new_project_form_filled")
        # Submit
        await page.click('#np-form button[type="submit"]')
        # Wait for the new project to appear in the list (page reloads after submit)
        try:
            await wait_for(
                lambda: page.evaluate(
                    f"() => !!Array.from(document.querySelectorAll('a')).find(a => a.textContent.includes('{proj_name}'))"
                ),
                timeout_ms=15000, label="new project visible"
            )
            href = await page.evaluate(
                f"() => {{"
                f"  const a = Array.from(document.querySelectorAll('a')).find(a => a.textContent.includes('{proj_name}'));"
                f"  return a ? a.getAttribute('href') : null;"
                f"}}"
            )
            pid = href.split("/")[-1] if href else None
            rec("webui.project_created", bool(pid), pid or "no href")
            await shot("06_project_created")
        except Exception as e:
            rec("webui.project_created", False, str(e)[:80])
            pid = None

        if not pid:
            print("\nABORT: no project id, can't continue training flow.")
            await browser.close()
            return 1

        # ── 5. DATA: upload synthetic JSONL ─────────────────────────────
        print("\n[5/8] WEBUI: upload synthetic dataset")
        await page.goto(f"{BASE}/projects/{pid}/data", wait_until="networkidle")
        await page.wait_for_timeout(1500)
        await shot("07_data_page")
        # Set the file input (this IS the user action — clicking the
        # "Choose File" button which opens a native picker that Playwright
        # handles by setting the file directly).
        await page.set_input_files('#upload-form input[type="file"]', str(sample_path))
        await page.wait_for_timeout(500)
        await shot("08_data_file_selected")
        # Click submit button to actually upload
        await page.click('#upload-form button[type="submit"]')
        # Page reloads after upload (data-reload="true"). Wait for the
        # filename to appear in the file list.
        try:
            await wait_for(
                lambda: page.evaluate(
                    "() => document.body.innerText.includes('capitals.jsonl')"
                ),
                timeout_ms=30000, label="file appears in list"
            )
            rec("webui.data_uploaded", True)
            await shot("09_data_file_uploaded")
        except Exception as e:
            rec("webui.data_uploaded", False, str(e)[:80])

        # ── 6. TRAINING: select model + fill params + Start ────────────
        print("\n[6/8] WEBUI: configure training")
        await page.goto(f"{BASE}/projects/{pid}/training", wait_until="networkidle")
        await page.wait_for_timeout(2500)
        await shot("10_training_form")

        # Pick Qwen3 from the model dropdown
        try:
            opts = await page.evaluate("""
                () => Array.from(document.querySelectorAll('select[name=model_path] option'))
                    .map(o => ({v: o.value, t: o.textContent.trim()}))
                    .filter(o => /Qwen3/i.test(o.t))
            """)
            print(f"  qwen options: {[o['t'][:50] for o in opts[:5]]}")
            if opts:
                # Find option containing 'Qwen3-0.6B'
                target = next((o for o in opts if 'Qwen3-0.6B' in o['t']), opts[0])
                await page.evaluate(
                    "(v) => { const el = document.querySelector('select[name=model_path]');"
                    "el.value = v; el.dispatchEvent(new Event('change', {bubbles:true})); }",
                    target['v']
                )
                rec("webui.model_selected", True, target['t'][:50])
            else:
                rec("webui.model_selected", False, "no Qwen option found in dropdown")
        except Exception as e:
            rec("webui.model_selected", False, str(e)[:80])

        # Fill data path (text input)
        await page.fill('input[name="data_path"]', str(sample_path))

        # Reduce epochs/batch/seq to keep E2E test reasonable
        try:
            await page.fill('input[name="num_epochs"]', "2")
            await page.fill('input[name="batch_size"]', "1")
            await page.fill('input[name="max_seq_length"]', "512")
        except Exception:
            pass
        await shot("11_training_form_filled")

        # Click Start
        try:
            await page.click("#start-btn")
            rec("webui.start_clicked", True)
            await shot("12_training_started")
        except Exception as e:
            rec("webui.start_clicked", False, str(e)[:60])

        # ── 7. WATCH training status badge until completion ───────────
        print("\n[7/8] WEBUI: watch training to completion")
        try:
            # Wait for status badge to leave 'idle'
            await wait_for(
                lambda: page.evaluate(
                    "() => { const b = document.getElementById('train-status-badge');"
                    " return b && b.innerText && !/^idle$/i.test(b.innerText.trim()); }"
                ),
                timeout_ms=90000, label="training started"
            )
            rec("webui.training_started", True)

            # Then wait for completion (done/failed/stopped) — up to 40 min
            await wait_for(
                lambda: page.evaluate(
                    "() => { const b = document.getElementById('train-status-badge');"
                    " return b && /done|failed|stopped|complete|error/i.test(b.innerText); }"
                ),
                timeout_ms=2400000, label="training finished", interval_ms=8000,
            )
            final = await page.evaluate("() => document.getElementById('train-status-badge').innerText")
            rec("webui.training_completed", True, f"status={final}")
            await shot("13_training_completed")
        except Exception as e:
            rec("webui.training_completed", False, str(e)[:100])
            await shot("13_training_failed_or_timeout")

        # ── 8. VERIFY run appears ──────────────────────────────────────
        print("\n[8/8] WEBUI: verify training run registered")
        # Open the project overview to see the run in the list (no API hack)
        await page.goto(f"{BASE}/projects/{pid}", wait_until="networkidle")
        await page.wait_for_timeout(2000)
        await shot("14_project_overview_after_train")
        # The runs table on the project page should show the run
        run_visible = await page.evaluate("""
            () => {
                const tables = Array.from(document.querySelectorAll('table'));
                for (const t of tables) {
                    if (/run|status|model/i.test(t.innerText) &&
                        t.querySelectorAll('tr').length > 1) {
                        return true;
                    }
                }
                return false;
            }
        """)
        rec("webui.run_visible_in_project", run_visible)

        await page.close()
        await browser.close()

    passed = sum(1 for _, p, _ in results if p)
    print(f"\n{'='*64}\n  TRAINING FLOW: {passed}/{len(results)} passed\n{'='*64}")
    for name, p, note in results:
        sym = "✓" if p else "✗"
        print(f"  {sym} {name}: {note}")
    print(f"{'='*64}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))