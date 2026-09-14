#!/usr/bin/env python3
"""Phase B + Phase D driver — actually train via UI + benchmark the result.

B) Full training via UI:
   - Open /projects/<id>/training
   - Screenshot empty form
   - Fill all fields with a small fast config
   - Click Start training
   - Screenshot immediately after click
   - Poll every 10s, re-screenshot every 30s
   - Look for state transitions: queued -> running -> completed/failed
   - Bail if no transition within 60s

D) Benchmark:
   - After training done (or bailed), navigate to /projects/<id>/benchmarks
   - Screenshot idle state
   - Inspect what's available
   - If a model dropdown exists, select the trained one
   - Try to run a benchmark
   - Poll + screenshot result

Outputs:
  /home/genorbox1/.openclaw/media/outbound/phase-b-prefill.png
  phase-b-filled.png
  phase-b-started.png
  phase-b-progress-{N}.png
  phase-b-final.png
  phase-b-summary.json
  phase-d-benchmarks-*.png
  phase-d-summary.json

Also visits all project pages during Phase B for visibility (continuation
of Phase C).
"""
import json
import re
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path("/home/genorbox1/.openclaw/media/outbound")
OUT.mkdir(parents=True, exist_ok=True)
PROJECT_ID = "264f8765"
BASE = f"http://fan-dragon:7860"

# Small config — fastest possible training that still proves it works
SMALL_CONFIG = {
    "dataset-select": "a0cd4cf6",      # capitals_qa · 40 pairs · 4.0 KB
    "output_dir":     "output-e2e-amy", # unique so we can find this run
    "lora_rank":      "8",
    "learning_rate":  "1e-4",           # slightly higher for fast convergence
    "num_epochs":     "1",
    "batch_size":     "2",
    "max_seq_length": "512",
}

POLL_INTERVAL_SEC = 10
PROGRESS_EVERY_SEC = 30
HARD_TIMEOUT_SEC = 600  # 10 min wall clock for the whole phase B run
TRAINING_START_GRACE_SEC = 60  # wait this long for queued->running transition


def shot(page, name, full_page: bool = True):
    """Save screenshot, return absolute path."""
    png = page.screenshot(full_page=full_page)
    p = OUT / name
    p.write_bytes(png)
    return str(p), len(png)


def get_state(page) -> str:
    """Extract current training state from DOM as best we can.

    Heuristics:
      - Look for elements with text "running", "completed", "failed", "queued"
      - Look for progress bar %
      - Look for "Start training" button being disabled/hidden (means we started)
    """
    return page.evaluate("""
        (function(){
            var txt = (document.body.textContent || '').toLowerCase();
            var state = 'unknown';
            if (/training.*(started|started|active)/.test(txt) || txt.indexOf('running') >= 0) {
                state = 'running';
            } else if (txt.indexOf('completed') >= 0 || txt.indexOf('finished') >= 0) {
                state = 'completed';
            } else if (txt.indexOf('failed') >= 0 || txt.indexOf('error') >= 0) {
                state = 'failed';
            } else if (txt.indexOf('queued') >= 0 || txt.indexOf('waiting') >= 0) {
                state = 'queued';
            }
            var pct = null;
            var match = txt.match(/(\\d+(?:\\.\\d+)?)\\s*%/);
            if (match) pct = parseFloat(match[1]);
            var startBtn = document.getElementById('start-btn');
            return {
                state: state,
                pct: pct,
                startBtnVisible: startBtn ? (startBtn.offsetWidth > 0) : null,
                startBtnText: startBtn ? (startBtn.textContent||'').trim() : null,
                bodyTextSnippet: txt.substring(0, 500)
            };
        })()
    """)


def dismiss_overlays(page):
    """Hide any tutorial/modal overlays that block page interactions.

    The webui shows a tutorial overlay on first visit per session. It
    intercepts all pointer events so .click() times out waiting for the
    target to be unobstructed. We hide (don't remove) it so the page's
    own state machine is preserved — next time the page loads, it will
    decide what to show again based on localStorage / cookies.
    """
    page.evaluate("""
        (function(){
            var sels = [
                '#tutorial-overlay',
                '.tutorial-overlay',
                '.modal-backdrop',
                '[role=dialog]',
                '.onboarding-overlay'
            ];
            for (var i = 0; i < sels.length; i++){
                var nodes = document.querySelectorAll(sels[i]);
                for (var j = 0; j < nodes.length; j++){
                    var n = nodes[j];
                    n.style.display = 'none';
                    n.style.pointerEvents = 'none';
                    n.setAttribute('aria-hidden', 'true');
                }
            }
            // Also try clicking any visible "Skip" / "Got it" / close button
            var buttons = document.querySelectorAll('button, a');
            var keywords = ['skip','close','dismiss','got it','ok','×','✕','\\u2715'];
            for (var k = 0; k < buttons.length; k++){
                var b = buttons[k];
                var t = (b.textContent||'').trim().toLowerCase();
                if (!t) continue;
                for (var m = 0; m < keywords.length; m++){
                    if (t === keywords[m] || t.indexOf(keywords[m]) >= 0){
                        try { b.click(); } catch(e){}
                        break;
                    }
                }
            }
            return document.querySelectorAll('#tutorial-overlay, .tutorial-overlay').length;
        })()
    """)


def page_text(page) -> str:
    return page.evaluate("(document.body.textContent||'').trim().substring(0, 3000)")


def visit_pages(page, names: list) -> list:
    """Screenshot every project page. Returns list of dicts."""
    out = []
    for name, path in names:
        url = f"{BASE}{path}?bust=phase-b-{name}"
        try:
            resp = page.goto(url, wait_until="networkidle", timeout=10000)
            status = resp.status if resp else None
        except Exception as e:
            status = f"error: {e}"
        png_path, _ = shot(page, f"phase-b-{name}.png")
        layout = page.evaluate("""
            (function(){
                var m = document.querySelector('.main');
                var doc = document.documentElement.scrollWidth;
                return {mainW: m ? Math.round(m.getBoundingClientRect().width) : null,
                        docW: doc, hasHoriz: doc > window.innerWidth};
            })()
        """)
        out.append({"page": name, "url": path, "status": status,
                    "screenshot": png_path, "layout": layout})
        print(f"  [PAGE] {name:14} status={status} {layout}")
    return out


def main() -> int:
    summary = {"phase_a": "see phase-ac-summary.json",
               "phase_b_started": time.time()}
    print(f"== Phase B + D driver ==")
    print(f"  config: {SMALL_CONFIG}")
    with sync_playwright() as p:
        b = p.chromium.launch()
        try:
            ctx = b.new_context(viewport={"width": 1200, "height": 800})
            page = ctx.new_page()
            page.set_default_timeout(15000)

            # ---- Phase B: training ----
            print("\n-- Phase B: training --")
            page.goto(f"{BASE}/projects/{PROJECT_ID}/training?bust=phase-b-form",
                      wait_until="networkidle", timeout=15000)
            time.sleep(1)
            dismiss_overlays(page)  # tutorial was blocking #start-btn on first run
            time.sleep(0.5)
            prefill_path, _ = shot(page, "phase-b-prefill.png")
            print(f"  prefill -> {prefill_path}")

            # Fill form
            page.select_option("select[name=model_path]", index=0)
            for sel_name, val in SMALL_CONFIG.items():
                if sel_name == "dataset-select":
                    page.select_option("#dataset-select", val)
                else:
                    page.fill(f"[name='{sel_name}']", val)
            time.sleep(0.5)
            filled_path, _ = shot(page, "phase-b-filled.png")
            print(f"  filled -> {filled_path}")

            # Capture form data for verification
            filled_data = page.evaluate("""
                (function(){
                    var out = {};
                    var inputs = document.querySelectorAll('input, select');
                    inputs.forEach(function(el){
                        if (!el.name) return;
                        out[el.name] = el.value;
                    });
                    return out;
                })()
            """)
            summary["phase_b_filled_form"] = filled_data

            # Click Start
            start_t = time.time()
            page.click("#start-btn")
            print(f"  clicked Start training at {start_t}")
            time.sleep(2)
            started_path, _ = shot(page, "phase-b-started.png")
            print(f"  started -> {started_path}")

            # Poll loop
            progress_shots = []
            last_shot_t = 0
            state_history = []
            state = {"state": "unknown"}
            training_completed = False

            while time.time() - start_t < HARD_TIMEOUT_SEC:
                elapsed = time.time() - start_t
                state = get_state(page)
                state_history.append({"t": round(elapsed, 1), **state})

                # Re-screenshot every 30s
                if time.time() - last_shot_t > PROGRESS_EVERY_SEC:
                    name = f"phase-b-progress-t{int(elapsed)}s.png"
                    path, _ = shot(page, name)
                    progress_shots.append(path)
                    last_shot_t = time.time()

                # Termination conditions
                if state["state"] in ("completed", "failed"):
                    print(f"  state={state['state']} at t={elapsed:.0f}s")
                    training_completed = (state["state"] == "completed")
                    break
                if state["state"] == "running" and elapsed > TRAINING_START_GRACE_SEC:
                    # only bail if it's been running long enough to also detect completion
                    # don't break here, keep polling
                    pass
                if elapsed > TRAINING_START_GRACE_SEC and state["state"] == "unknown" and state["startBtnVisible"]:
                    # Button is back to visible and we're past grace — training didn't stick
                    print(f"  Start button visible again after {elapsed:.0f}s — training did not start")
                    break

                time.sleep(POLL_INTERVAL_SEC)

            final_t = time.time() - start_t
            final_path, _ = shot(page, "phase-b-final.png")
            final_state = get_state(page)
            final_text = page_text(page)
            summary["phase_b_training"] = {
                "elapsed_sec": round(final_t, 1),
                "final_state": final_state,
                "completed": training_completed,
                "progress_shots": progress_shots,
                "started_shot": started_path,
                "final_shot": final_path,
                "state_history_tail": state_history[-8:],
                "body_snippet": final_text[:1500],
            }
            print(f"  final -> {final_path}  state={final_state['state']}")

            # ---- Phase B continued: visit all pages ----
            print("\n-- Phase B continuation: visit all project pages --")
            pages_info = visit_pages(page, [
                ("overview", f"/projects/{PROJECT_ID}"),
                ("data",     f"/projects/{PROJECT_ID}/data"),
                ("data-prep",f"/projects/{PROJECT_ID}/data-prep"),
                ("rag",      f"/projects/{PROJECT_ID}/rag"),
                ("testing",  f"/projects/{PROJECT_ID}/testing"),
                ("chat",     f"/projects/{PROJECT_ID}/chat"),
            ])
            summary["phase_b_pages"] = pages_info

            # ---- Phase D: benchmarks ----
            print("\n-- Phase D: benchmarks --")
            bench_url = f"{BASE}/projects/{PROJECT_ID}/benchmarks?bust=phase-d"
            try:
                resp = page.goto(bench_url, wait_until="networkidle", timeout=15000)
                bench_status = resp.status if resp else None
            except Exception as e:
                bench_status = f"error: {e}"
            idle_path, _ = shot(page, "phase-d-benchmarks-idle.png")
            print(f"  idle -> {idle_path} (status={bench_status})")

            bench_form = page.evaluate("""
                (function(){
                    var inputs = document.querySelectorAll('input,select,textarea,button');
                    var out = [];
                    inputs.forEach(function(el){
                        var r = el.getBoundingClientRect();
                        if (r.width === 0) return;
                        if (el.type === 'hidden') return;
                        out.push({
                            tag: el.tagName, type: el.type || null, id: el.id || null,
                            name: el.name || null, value: el.value || null,
                            text: (el.textContent||'').trim().substring(0,80),
                            options: el.tagName === 'SELECT' ?
                                Array.from(el.options).map(function(o){
                                    return {v: o.value, t: o.text.trim().substring(0,80), selected: o.selected};
                                }) : null
                        });
                    });
                    return out;
                })()
            """)
            summary["phase_d_form"] = bench_form
            print(f"  form fields discovered: {len(bench_form)}")

            # Try to run benchmark if there's a sensible run button + model select
            run_summary = {"attempted": False, "reason_skipped": None, "shots": []}
            model_select = None
            run_btn = None
            for el in bench_form:
                if el["tag"] == "SELECT" and el["options"] and len(el["options"]) > 1:
                    if not model_select or "model" in (el.get("name") or "").lower() or \
                       any("/home" in (o.get("v") or "") or "/" in (o.get("v") or "") for o in el["options"]):
                        model_select = el
                if el["tag"] == "BUTTON" and el.get("text", "").lower().startswith("run"):
                    run_btn = el

            if model_select:
                # Select the most recent / our trained model
                opts = [o for o in model_select["options"] if o["v"]]
                chosen = None
                for o in opts:
                    if "e2e" in o["t"].lower() or "e2e" in o["v"].lower():
                        chosen = o["v"]
                        break
                if not chosen and opts:
                    chosen = opts[-1]["v"]  # last = newest usually
                if chosen:
                    page.select_option(f"#{model_select['id']}" if model_select.get("id") else f"select[name='{model_select['name']}']", chosen)
                    print(f"  selected model: {chosen}")
                    run_summary["selected_model"] = chosen

            if run_btn and model_select:
                print(f"  clicking Run: {run_btn.get('text')}")
                run_summary["attempted"] = True
                try:
                    btn_sel = f"#{run_btn['id']}" if run_btn.get("id") else f"button:has-text('{run_btn['text']}')"
                    page.click(btn_sel)
                    time.sleep(3)
                    p, _ = shot(page, "phase-d-benchmarks-running.png")
                    run_summary["shots"].append(p)
                    # Poll briefly
                    for i in range(12):  # 2 min
                        time.sleep(10)
                        p, _ = shot(page, f"phase-d-benchmarks-poll-{i}.png")
                        run_summary["shots"].append(p)
                        bt = page_text(page).lower()
                        if "complete" in bt or "passed" in bt or "failed" in bt:
                            break
                    p, _ = shot(page, "phase-d-benchmarks-results.png")
                    run_summary["shots"].append(p)
                    run_summary["results_text_snippet"] = page_text(page)[:2000]
                except Exception as e:
                    run_summary["error"] = str(e)
                    print(f"  run failed: {e}")
            else:
                run_summary["reason_skipped"] = (
                    f"no model select + run button combo "
                    f"(select={bool(model_select)}, run_btn={bool(run_btn)})")
                print(f"  skipping run: {run_summary['reason_skipped']}")

            summary["phase_d_run"] = run_summary
            summary["phase_d_idle_shot"] = idle_path
            summary["phase_d_bench_status"] = bench_status

            ctx.close()
        finally:
            b.close()

    summary["finished_at"] = time.time()
    summary_path = OUT / "phase-bd-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nJSON summary: {summary_path}")
    print("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
