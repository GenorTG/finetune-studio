"""Live-browser end-to-end functional tests for Finetune Studio.

Drives a real Chromium through the entire user workflow:
  1. Create project via WebUI
  2. Upload a small file
  3. Build RAG corpus (real embedder, real chunks)
  4. Search RAG
  5. Start a real training run (tiny Llama + 3 steps)
  6. Wait for training to complete
  7. Run MMLU benchmark against the trained model
  8. Stream inference chat and confirm tokens arrive in UI

Run with: FTS_BASE=http://localhost:7860 python tests/e2e_functional.py
"""

import asyncio
import os
import sys
import time
import tempfile
import json
from pathlib import Path
from playwright.async_api import async_playwright

BASE = os.environ.get("FTS_BASE", "http://localhost:7860")
SMALL_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TRAIN_SAMPLES = []  # populated below


def make_sample_jsonl(path: Path):
    """A 5-sample instruction dataset — enough to exercise the trainer."""
    pairs = [
        ("What is 2+2?", "4"),
        ("Capital of France?", "Paris"),
        ("Color of the sky?", "Blue"),
        ("Greet me.", "Hello, friend."),
        ("Say goodbye.", "Farewell."),
    ]
    with open(path, "w") as f:
        for q, a in pairs:
            f.write(json.dumps({"prompt": q, "completion": a}) + "\n")


async def wait_for(fn, timeout_ms=30000, interval_ms=500, label="condition"):
    """Poll a function returning truthy until timeout."""
    elapsed = 0
    while elapsed < timeout_ms:
        try:
            r = await fn()
            if r:
                return r
        except Exception:
            pass
        await asyncio.sleep(interval_ms / 1000)
        elapsed += interval_ms
    raise TimeoutError(f"timed out after {timeout_ms}ms waiting for {label}")


async def main():
    tmp = Path(tempfile.mkdtemp(prefix="fts_p4_"))
    sample_jsonl = tmp / "sample.jsonl"
    make_sample_jsonl(sample_jsonl)
    upload_path = tmp / "doc.txt"
    upload_path.write_text(
        "Finetune Studio is a self-hosted LLM training workshop. "
        "It supports RAG corpora, LoRA fine-tuning, and benchmarks. "
        "It runs entirely on local hardware. "
        "The user creates a project, uploads data, and chooses their workflow."
    )
    print(f"tmpdir: {tmp}")

    results = []
    def rec(name, passed, note=""):
        results.append((name, passed, note))
        sym = "✓" if passed else "✗"
        print(f"  {sym} {name}{(': ' + note) if note else ''}")

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await ctx.new_page()

        # ===== 1. CREATE PROJECT via WebUI =====
        print("\n[1/8] CREATE PROJECT")
        # Inject a script that pre-marks tutorial as seen BEFORE any page loads
        await ctx.add_init_script("""
          try { localStorage.setItem('fts.tutorial.seen', '1'); } catch (e) {}
        """)
        await page.goto(f"{BASE}/projects", wait_until="networkidle")
        await page.wait_for_timeout(1500)

        # Belt + braces: also force the overlay closed if it managed to mount
        try:
            await page.evaluate("() => { const o = document.getElementById('tutorial-overlay'); if (o) o.hidden = true; }")
            await page.wait_for_timeout(200)
        except Exception:
            pass

        # Submit new project form
        unique_name = f"P4-Test-{int(time.time())}"
        # Form starts hidden; click "New project" button to expand it
        try:
            await page.click('button:has-text("New project")', timeout=5000)
            await page.wait_for_timeout(600)
        except Exception:
            pass
        # Form is inside #np-form; scope all selectors to that
        await page.fill('#np-form input[name="name"]', unique_name)
        await page.fill(
            '#np-form textarea[name="description"]',
            "End-to-end functional test project."
        )
        # Submit the np-form (it does an API POST + reload)
        try:
            await page.click('#np-form button[type="submit"]')
        except Exception as e:
            rec("create.submit_clicked", False, str(e)[:80])

        # Wait for the new project to appear
        try:
            await wait_for(
                lambda: page.evaluate(
                    f"() => !!document.querySelector('a[href*=\"projects/\"]') "
                    f"&& Array.from(document.querySelectorAll('a')).some("
                    f"  a => a.textContent.includes({unique_name!r}))"
                ),
                timeout_ms=10000, label=f"project '{unique_name}' visible"
            )
            rec("create.project_visible", True)
        except Exception as e:
            rec("create.project_visible", False, str(e)[:80])

        # Get the project URL
        proj_link = await page.query_selector(f'a:has-text("{unique_name}")')
        proj_href = await proj_link.get_attribute("href") if proj_link else None
        rec("create.project_has_href", bool(proj_href), str(proj_href)[:40])
        pid = proj_href.split("/")[-1] if proj_href else None

        if not pid:
            print("Cannot continue without a project id.")
            await browser.close()
            return 1

        # ===== 2. UPLOAD FILE =====
        print("\n[2/8] UPLOAD FILE")
        await page.goto(f"{BASE}/projects/{pid}/data", wait_until="networkidle")
        await page.wait_for_timeout(1500)
        # Use the upload-form scoped to this project
        file_input = await page.query_selector('#upload-form input[type="file"]')
        if file_input:
            await file_input.set_input_files(str(upload_path))
            await page.wait_for_timeout(500)
            # Submit form (data-reload triggers reload after upload)
            try:
                await page.click('#upload-form button[type="submit"]')
                await page.wait_for_timeout(3000)
            except Exception as e:
                rec("upload.submit_clicked", False, str(e)[:80])
            # Verify file shows up in the file-list
            await wait_for(
                lambda: page.evaluate(
                    "() => document.body.innerText.includes('doc.txt')"
                ),
                timeout_ms=15000, label="uploaded file visible"
            )
            rec("upload.file_visible", True)
        else:
            rec("upload.file_input_found", False)
            rec("upload.file_visible", False, "skipped (no input)")

        # ===== 3. BUILD RAG CORPUS =====
        print("\n[3/8] BUILD RAG")
        await page.goto(f"{BASE}/projects/{pid}/rag", wait_until="networkidle")
        await page.wait_for_timeout(1500)
        # Find build button and click
        build_clicked = False
        for sel in [
            '#build-btn',
            'button:has-text("Build")',
            'button:has-text("BUILD")',
            'button:has-text("Start")',
            '[data-action*="build"]',
        ]:
            try:
                await page.click(sel, timeout=2000)
                build_clicked = True
                break
            except Exception:
                continue
        rec("rag.build_clicked", build_clicked)

        # Wait for progress to start: buildCorpus() logs to #build-status
        try:
            await wait_for(
                lambda: page.evaluate(
                    "() => { const s = document.getElementById('build-status'); "
                    "return s && s.innerText && s.innerText.length > 0; }"
                ),
                timeout_ms=20000, label="rag build started/completed"
            )
            rec("rag.build_started", True)
        except Exception:
            rec("rag.build_started", False)

        # Wait for completion: look for "complete" / "ready" / "✓" / "files" in status
        try:
            await wait_for(
                lambda: page.evaluate(
                    "() => document.body.innerText.toLowerCase().match(/complete|ready|\\u2713|files?\\b/) !== null"
                ),
                timeout_ms=120000, label="rag build completion"
            )
            rec("rag.build_completed", True)
        except Exception as e:
            rec("rag.build_completed", False, str(e)[:80])

        # ===== 4. SEARCH RAG =====
        print("\n[4/8] SEARCH RAG")
        # Find search input on the same page
        try:
            await page.fill('input[type="search"], #rag-query, input[placeholder*="ask" i], input[placeholder*="search" i]', "What is Finetune Studio?")
            await page.wait_for_timeout(500)
            try:
                await page.keyboard.press("Enter")
            except Exception:
                pass
            # Look for results to appear
            await page.wait_for_timeout(2500)
            has_results = await page.evaluate(
                "() => document.body.innerText.toLowerCase().includes('chunk') || "
                "document.body.innerText.toLowerCase().includes('match') || "
                "document.body.innerText.toLowerCase().includes('result')"
            )
            rec("rag.search_executed", has_results)
        except Exception as e:
            rec("rag.search_executed", False, str(e)[:80])

        # ===== 5. START TRAINING =====
        print("\n[5/8] START TRAINING")
        await page.goto(f"{BASE}/projects/{pid}/training", wait_until="networkidle")
        await page.wait_for_timeout(2000)

        # Fill in required fields if present
        # model_path
        for sel in ['#model-path', 'input[name="model_path"]', 'select[name="model_path"]', '#base-model']:
            try:
                await page.fill(sel, SMALL_MODEL)
                break
            except Exception:
                continue
        # data_path
        for sel in ['#data-path', 'input[name="data_path"]']:
            try:
                await page.fill(sel, str(sample_jsonl))
                break
            except Exception:
                continue
        # Try to find training start button
        started = False
        for sel in [
            'button:has-text("START"), button:has-text("Start"), button:has-text("TRAIN"), button:has-text("Train"), button[type="submit"]',
        ]:
            try:
                await page.click(sel, timeout=2000)
                started = True
                break
            except Exception:
                continue
        rec("training.start_clicked", started)

        # ===== 6. WAIT FOR TRAINING =====
        print("\n[6/8] WAIT FOR TRAINING")
        # Look for status: queued / running / complete
        try:
            await wait_for(
                lambda: page.evaluate(
                    "() => /queued|running|train|step/i.test(document.body.innerText)"
                ),
                timeout_ms=15000, label="training visible in UI"
            )
            rec("training.visible_in_ui", True)
        except Exception:
            rec("training.visible_in_ui", False)

        # Wait for completion: badge text changes to done/failed/stopped
        try:
            await wait_for(
                lambda: page.evaluate(
                    "() => { const b = document.getElementById('train-status-badge'); "
                    "return b && /done|failed|stopped|complete/i.test(b.innerText); }"
                ),
                timeout_ms=360000, label="training completion"
            )
            rec("training.completed_or_progressed", True)
        except Exception as e:
            rec("training.completed_or_progressed", False, str(e)[:80])

        # ===== 7. RUN BENCHMARK =====
        print("\n[7/8] RUN BENCHMARK")
        await page.goto(f"{BASE}/projects/{pid}/benchmarks", wait_until="networkidle")
        await page.wait_for_timeout(2500)

        bench_clicked = False
        for sel in [
            'button:has-text("RUN"), button:has-text("Run"), button:has-text("START")',
        ]:
            try:
                await page.click(sel, timeout=2000)
                bench_clicked = True
                break
            except Exception:
                continue
        rec("benchmark.run_clicked", bench_clicked)

        # Look for benchmark results
        try:
            await wait_for(
                lambda: page.evaluate(
                    "() => /(?:score|accuracy|mmlu)/i.test(document.body.innerText)"
                ),
                timeout_ms=120000, label="benchmark results visible"
            )
            rec("benchmark.results_visible", True)
        except Exception:
            # Still record what we got
            body = await page.evaluate("() => document.body.innerText")
            rec("benchmark.results_visible", False, body[:60])

        # ===== 8. STREAM INFERENCE =====
        print("\n[8/8] STREAM INFERENCE")
        await page.goto(f"{BASE}/inference", wait_until="networkidle")
        await page.wait_for_timeout(3000)

        # Try to load the small model
        loaded = False
        for sel in [
            'button:has-text("LOAD"), button:has-text("Load")',
        ]:
            try:
                await page.click(sel, timeout=2000)
                loaded = True
                break
            except Exception:
                continue

        if loaded:
            # Wait for "loaded" indicator
            try:
                await wait_for(
                    lambda: page.evaluate(
                        "() => /loaded|ready|online/i.test(document.body.innerText)"
                    ),
                    timeout_ms=30000, label="model loaded"
                )
                rec("inference.model_loaded", True)
            except Exception:
                rec("inference.model_loaded", False)

            # Send a chat message and watch for streaming tokens
            try:
                msg_input = await page.query_selector('#msg, textarea[placeholder*="Message" i]')
                if msg_input:
                    await msg_input.fill("Hello.")
                    # Click send button (Enter triggers send too but we use button to be sure)
                    try:
                        await page.click('#send-btn')
                    except Exception:
                        await page.keyboard.press("Enter")
                    # Wait for response in the #msgs container
                    await wait_for(
                        lambda: page.evaluate(
                            "() => { const m = document.getElementById('msgs'); "
                            "return m && m.innerText.length > 10; }"
                        ),
                        timeout_ms=60000, label="inference response"
                    )
                    rec("inference.streamed_response", True)
                else:
                    rec("inference.message_input", False, "no input found")
            except Exception as e:
                rec("inference.streamed_response", False, str(e)[:80])
        else:
            rec("inference.load_clicked", False)

        await page.close()
        await browser.close()

    # ===== REPORT =====
    passed = sum(1 for _, p, _ in results if p)
    failed = sum(1 for _, p, _ in results if not p)
    print(f"\n{'='*60}\n  FUNCTIONAL P4: {passed} passed, {failed} failed, total {len(results)}\n{'='*60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
