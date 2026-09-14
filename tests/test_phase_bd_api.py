#!/usr/bin/env python3
"""Phase B + D — API-driven training + benchmark end-to-end.

Avoids the flaky UI form submission (Playwright + overlay + redirect)
and hits the documented endpoints directly:

  POST /api/training/start          → start training, returns run_id
  GET  /api/training/status         → poll progress
  GET  /api/benchmarks/suites       → list suites
  POST /api/projects/{pid}/runs/{rid}/run → run benchmark on a run

Loads browser context ONLY for screenshots; never relies on UI form
submission for the actual flow.

Outputs into /home/genorbox1/.openclaw/media/outbound/:
  phase-api-start.png         - state right after POST start
  phase-api-training-*.png    - training page snapshots every 30s
  phase-api-trained.png       - state right after training done
  phase-api-bench-idle.png    - benchmarks page before run
  phase-api-bench-result.png  - benchmarks page after run
  phase-api-summary.json      - full timeline + results
"""
import json
import sys
import time
from pathlib import Path
import urllib.request
import urllib.error

import requests
from playwright.sync_api import sync_playwright

OUT = Path("/home/genorbox1/.openclaw/media/outbound")
OUT.mkdir(parents=True, exist_ok=True)
BASE = "http://fan-dragon:7860"
PROJECT_ID = "264f8765"
SMALL_CONFIG = {
    "model_path":    "/home/genortg/.finetune-studio/hf_models/Qwen__Qwen3-0.6B",
    "data_path":     "/home/genortg/finetune-studio/data/synthetic_qa.jsonl",
    "output_dir":    "output-e2e-api",
    "lora_rank":     8,
    "learning_rate": "1e-4",
    "num_epochs":    1,
    "batch_size":    2,
    "max_seq_length": 512,
}
TRAINING_TIMEOUT_SEC = 600   # 10 min hard ceiling
POLL_INTERVAL_SEC    = 10


def post(path: str, body: dict | None = None, timeout: float = 30) -> tuple[int, dict | str]:
    url = f"{BASE}{path}"
    headers = {"Content-Type": "application/json"} if body is not None else {}
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            try: return r.status, json.loads(raw)
            except json.JSONDecodeError: return r.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try: return e.code, json.loads(raw)
        except json.JSONDecodeError: return e.code, raw
    except Exception as e:
        return 0, f"error: {e}"


def get(path: str, timeout: float = 15) -> tuple[int, object]:
    url = f"{BASE}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            raw = r.read().decode()
            try: return r.status, json.loads(raw)
            except json.JSONDecodeError: return r.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try: return e.code, json.loads(raw)
        except json.JSONDecodeError: return e.code, raw
    except Exception as e:
        return 0, f"error: {e}"


def get_suites() -> list[dict]:
    code, body = get("/api/benchmarks/suites")
    if code == 200 and isinstance(body, list):
        return body
    return []


def get_status() -> dict:
    code, body = get("/api/training/status")
    return body if code == 200 and isinstance(body, dict) else {"status": "?", "raw": body}


def get_training_run_id_from_engine() -> str | None:
    """Try to read the engine's current run id from the status-text endpoint
    or from the runs list (most recently started run)."""
    code, runs = get(f"/api/projects/{PROJECT_ID}/runs")
    if code == 200 and isinstance(runs, list):
        # Filter to running / most recent
        runs_sorted = sorted(runs, key=lambda r: r.get("created_at") or 0, reverse=True)
        if runs_sorted:
            return runs_sorted[0].get("id")
    return None


def wait_for_terminal(timeout_sec: int, log_fn) -> tuple[str, dict]:
    """Poll until status is in ('done', 'error', 'stopped') or timeout.
    Returns (final_state, last_status_dict, history)."""
    t0 = time.time()
    last = {}
    history = []
    while time.time() - t0 < timeout_sec:
        last = get_status()
        elapsed = round(time.time() - t0, 1)
        log_fn(f"  [t={elapsed:>5}s] {last.get('status'):10} "
               f"step={last.get('current_step')}/{last.get('total_steps')} "
               f"loss={last.get('loss'):.4f} "
               f"epoch={last.get('epoch')} "
               f"err={(last.get('error') or '')[:40]}")
        history.append({"t": elapsed, **last})
        if last.get("status") in ("done", "error", "stopped", "failed"):
            return last.get("status"), last, history
        time.sleep(POLL_INTERVAL_SEC)
    return "timeout", last, history


def main() -> int:
    print("== Phase B+D via API ==")
    print(f"  base: {BASE}")
    print(f"  config: {json.dumps(SMALL_CONFIG, indent=2)}")

    summary: dict = {"started_at": time.time()}

    # ── Pre-flight: confirm engine idle + discover suites ──
    print("\n-- pre-flight --")
    pre_status = get_status()
    print(f"  engine pre-status: {pre_status.get('status')}")
    summary["pre_flight_status"] = pre_status

    suites = get_suites()
    print(f"  suites ({len(suites)}): {[s.get('name') for s in suites]}")
    summary["suites"] = suites

    # ── Start training ──
    print("\n-- start training --")
    t_start = time.time()
    code, body = post("/api/training/start", SMALL_CONFIG)
    print(f"  POST /api/training/start -> HTTP {code}")
    print(f"  body: {json.dumps(body, indent=2)[:500]}")
    summary["start_response"] = {"code": code, "body": body}
    if code != 200 or (isinstance(body, dict) and body.get("error")):
        print("  FATAL: training did not start")
        summary["error"] = "start failed"
        (OUT / "phase-api-summary.json").write_text(json.dumps(summary, indent=2))
        return 1

    run_id = (body or {}).get("run_id") if isinstance(body, dict) else None
    print(f"  run_id: {run_id}")

    # Browser: capture training page during the run
    summary["screenshots_training"] = []
    summary["screenshots_benchmark"] = []
    summary["screenshots_pages"] = []

    with sync_playwright() as p:
        b = p.chromium.launch()
        try:
            ctx = b.new_context(viewport={"width": 1200, "height": 800})
            page = ctx.new_page()
            page.set_default_timeout(10000)

            # Initial training page screenshot
            page.goto(f"{BASE}/projects/{PROJECT_ID}/training?bust=phase-api",
                      wait_until="networkidle", timeout=15000)
            png = page.screenshot(full_page=True)
            p_start = OUT / "phase-api-start.png"
            p_start.write_bytes(png)
            summary["screenshots_training"].append(str(p_start))
            print(f"  start shot -> {p_start.name}")

            # Poll while capturing screenshots periodically
            t0 = time.time()
            done_state, last_status, history = "timeout", {}, []
            shot_t = 0
            while time.time() - t0 < TRAINING_TIMEOUT_SEC:
                last_status = get_status()
                elapsed = round(time.time() - t0, 1)
                print(f"  [poll t={elapsed:>5}s] {last_status.get('status'):10} "
                      f"step={last_status.get('current_step')}/{last_status.get('total_steps')} "
                      f"loss={last_status.get('loss'):.4f}")
                history.append({"t": elapsed, **last_status})
                if time.time() - shot_t > 30:
                    page.goto(f"{BASE}/projects/{PROJECT_ID}/training?bust=phase-api-t{int(elapsed)}",
                              wait_until="domcontentloaded", timeout=10000)
                    png = page.screenshot(full_page=True)
                    p = OUT / f"phase-api-training-t{int(elapsed)}s.png"
                    p.write_bytes(png)
                    summary["screenshots_training"].append(str(p))
                    shot_t = time.time()
                s = last_status.get("status")
                if s in ("done", "error", "stopped", "failed"):
                    done_state = s
                    break
                time.sleep(POLL_INTERVAL_SEC)
            summary["training_status_history_tail"] = history[-12:]
            summary["training_final_state"] = done_state
            summary["training_last_status"] = last_status

            # Final training page screenshot
            page.goto(f"{BASE}/projects/{PROJECT_ID}/training?bust=phase-api-final",
                      wait_until="networkidle", timeout=10000)
            png = page.screenshot(full_page=True)
            p_final = OUT / "phase-api-trained.png"
            p_final.write_bytes(png)
            summary["screenshots_training"].append(str(p_final))

            print(f"\n  final training state: {done_state}")
            print(f"  final training shot: {p_final.name}")

            # ── Visit all pages now that training is in a terminal state ──
            print("\n-- visit all project pages (post-training) --")
            for name, path in [
                ("overview",   f"/projects/{PROJECT_ID}"),
                ("data",       f"/projects/{PROJECT_ID}/data"),
                ("data-prep",  f"/projects/{PROJECT_ID}/data-prep"),
                ("rag",        f"/projects/{PROJECT_ID}/rag"),
                ("testing",    f"/projects/{PROJECT_ID}/testing"),
                ("chat",       f"/projects/{PROJECT_ID}/chat"),
                ("benchmarks", f"/projects/{PROJECT_ID}/benchmarks"),
            ]:
                try:
                    resp = page.goto(f"{BASE}{path}?bust=phase-api-{name}",
                                     wait_until="networkidle", timeout=10000)
                    status = resp.status if resp else None
                except Exception as e:
                    status = f"err: {e}"
                png = page.screenshot(full_page=True)
                p_path = OUT / f"phase-api-page-{name}.png"
                p_path.write_bytes(png)
                summary["screenshots_pages"].append({
                    "page": name, "url": path, "status": status,
                    "screenshot": str(p_path),
                })
                print(f"  [{name:11}] status={status} -> {p_path.name}")

            # ── Phase D: benchmark ──
            print("\n-- phase D: benchmark --")
            target_run_id = run_id or get_training_run_id_from_engine()
            print(f"  target run_id: {target_run_id}")

            bench_idle_shot = OUT / "phase-api-bench-idle.png"
            page.goto(f"{BASE}/projects/{PROJECT_ID}/benchmarks?bust=phase-api-bench-idle",
                      wait_until="networkidle", timeout=10000)
            png = page.screenshot(full_page=True)
            bench_idle_shot.write_bytes(png)
            summary["screenshots_benchmark"].append(str(bench_idle_shot))

            # Find the smallest available suite
            chosen_suite = None
            if suites:
                # Prefer default, else first discovered
                chosen_suite = next((s for s in suites if s.get("name") == "default"), suites[0])
            print(f"  chosen suite: {chosen_suite}")

            bench_result = {"attempted": False, "run_id": target_run_id, "suite": chosen_suite}
            if target_run_id and chosen_suite:
                # Verify run status from DB
                code, run_info = get(f"/api/projects/{PROJECT_ID}/runs/{target_run_id}")
                bench_result["run_status"] = (run_info or {}).get("status") if isinstance(run_info, dict) else None
                bench_result["run_output_path"] = (run_info or {}).get("output_path") if isinstance(run_info, dict) else None
                print(f"  run.status={bench_result['run_status']} output_path={bench_result['run_output_path']}")

                bench_result["attempted"] = True
                t_b0 = time.time()
                code, body = post(
                    f"/api/benchmarks/projects/{PROJECT_ID}/runs/{target_run_id}/run",
                    {"suite_name": chosen_suite["name"], "suite_path": chosen_suite["path"]},
                    timeout=600,
                )
                bench_result["response_code"] = code
                bench_result["response_body"] = body
                bench_result["bench_elapsed_sec"] = round(time.time() - t_b0, 1)
                print(f"  POST /benchmark -> HTTP {code} ({bench_result['bench_elapsed_sec']}s)")
                if isinstance(body, dict) and body.get("error"):
                    print(f"  benchmark error: {body['error']}")

                # Final benchmarks page screenshot
                page.goto(f"{BASE}/projects/{PROJECT_ID}/benchmarks?bust=phase-api-bench-result",
                          wait_until="networkidle", timeout=10000)
                png = page.screenshot(full_page=True)
                bench_result_shot = OUT / "phase-api-bench-result.png"
                bench_result_shot.write_bytes(png)
                summary["screenshots_benchmark"].append(str(bench_result_shot))
                bench_result["result_shot"] = str(bench_result_shot)

                # History fetch
                code, hist = get(f"/api/benchmarks/projects/{PROJECT_ID}/runs/{target_run_id}/history")
                bench_result["history_code"] = code
                bench_result["history"] = hist
                if isinstance(hist, list) and hist:
                    latest = hist[0]
                    print(f"  benchmark scores: {latest.get('scores')}")
                    print(f"  duration_ms: {latest.get('duration_ms')}")
            else:
                print("  skipping benchmark — no run_id or no suite")

            summary["phase_d"] = bench_result
            ctx.close()
        finally:
            b.close()

    summary["finished_at"] = time.time()
    summary_path = OUT / "phase-api-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nJSON summary: {summary_path}")
    print("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
