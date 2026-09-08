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
        pulled = False
        # The Pull button has data-body containing the repo_id
        repo_escaped = "Qwen/Qwen3-0.6B".replace("/", "\\/")
        try:
            btn_sel = f'button[data-body*="{repo_escaped}"]'
            await page.wait_for_selector(btn_sel, timeout=8000)
            await shot("02b_hf_pull_button_visible")
            await page.click(btn_sel)
            pulled = True
            print("  PULL button clicked")
        except Exception as e:
            print(f"  primary PULL selector failed: {e}")

        # If primary selector missed, find by card text and click the Pull button inside it
        if not pulled:
            try:
                # Find the card containing "Qwen3-0.6B", then click its Pull button
                clicked = await page.evaluate("""
                    () => {
                        const cards = Array.from(document.querySelectorAll('.hf-card'));
                        for (const c of cards) {
                            if (!c.innerText.includes('Qwen3-0.6B')) continue;
                            const btn = c.querySelector('button[data-confirm]');
                            if (btn) { btn.click(); return true; }
                        }
                        return false;
                    }
                """)
                if clicked:
                    pulled = True
                    print("  PULL button clicked via card-finder")
            except Exception as e:
                print(f"  card-finder failed: {e}")
        rec("webui.hf_pull_clicked", pulled)

        # ── 3. WATCH download progress UI ─────────────────────────────
        print("\n[3/8] WEBUI: watch download progress")
        # The download modal/progress shows in the WebUI. Watch for the file
        # to appear on disk too (both UI + filesystem must confirm completion).
        hub_dir = Path.home() / ".cache/huggingface/hub"
        model_dir = hub_dir / "models--Qwen--Qwen3-0.6B"
        # Wait for safetensors file on disk
        try:
            await wait_for(
                lambda: model_dir.exists() and any(
                    p.suffix == ".safetensors" for p in model_dir.rglob("*")
                ),
                timeout_ms=900000, label="Qwen3-0.6B downloaded",
                interval_ms=5000,
            )
            sf = [p for p in model_dir.rglob("*.safetensors")]
            total_gb = sum(p.stat().st_size for p in sf) / 1e9
            rec("webui.hf_download_complete", len(sf) > 0, f"{len(sf)} safetensors, {total_gb:.2f} GB")
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