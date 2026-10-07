"""Sandboxed browser proof that a GGUF export no longer freezes the app.

Starts a sandbox uvicorn on its own port (temp cwd, redirected HOME / FTS_ROOT / FTS_DB — the live
:7860 service and its data are never touched) with a FAKE llama.cpp toolchain (``tests/_fake_llama.py``:
a convert script and a ``llama-quantize`` that sleep for seconds as real child processes), seeds one
project with a merged run, then drives Chromium through the Export page while a hammer thread requests
``/api/projects`` and the project page every 200 ms. Reports the slowest answer and the timeouts.

    # with the fix (this checkout):
    PYTHONPATH=src .venv/bin/python tests/e2e_export_async_sandbox.py --port 7897 --label after
    # against old code, e.g. a worktree of origin/main (its UI is the old synchronous page):
    .venv/bin/python tests/e2e_export_async_sandbox.py --port 7898 --label before --src /path/to/main/src

Screenshots go to ``.tmp/qa-shots/export-async/`` (``FTS_QA_SHOTS`` overrides); the summary JSON next to them.
Real GGUF conversion is not used: the toolchain is fake on purpose (no GPU, deterministic, seconds long).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests._fake_llama import CONVERT, QUANTIZE  # noqa: E402

SEED = r"""
import sys
from pathlib import Path
from finetune_studio import db
db.init_db()
p = db.create_project(name="Export async sandbox", base_model="Qwen__Qwen3-0.6B")
pid = p["id"] if isinstance(p, dict) else p
out = Path(sys.argv[1]); merged = out / "merged"; merged.mkdir(parents=True)
(merged / "model.safetensors").write_bytes(b"w" * 64)
(merged / "config.json").write_text("{}")
run = db.create_run(project_id=pid, name="sandbox-run", base_model="Qwen/Qwen3-0.6B", data_path="/d")
db.update_run(run["id"], status="done", output_path=str(out))
print(pid)
"""


class Hammer(threading.Thread):
    """Requests two URLs every 200 ms and records every latency / failure."""

    def __init__(self, urls: list[str], timeout: float) -> None:
        super().__init__(daemon=True)
        self.urls, self.timeout = urls, timeout
        self.stop_flag = threading.Event()
        self.samples: list[tuple[float, str, float, str]] = []   # (t, url, seconds, error)

    def run(self) -> None:
        t00 = time.monotonic()
        while not self.stop_flag.is_set():
            for url in self.urls:
                t0 = time.monotonic()
                err = ""
                try:
                    urllib.request.urlopen(url, timeout=self.timeout).read()
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    err = type(exc).__name__
                self.samples.append((t0 - t00, url, time.monotonic() - t0, err))
            self.stop_flag.wait(0.2)

    def summary(self) -> dict:
        lat = [s[2] for s in self.samples if not s[3]]
        return {
            "requests": len(self.samples),
            "errors": sum(1 for s in self.samples if s[3]),
            "max_latency_s": round(max((s[2] for s in self.samples), default=0.0), 3),
            "p95_latency_s": round(statistics.quantiles(lat, n=20)[-1], 3) if len(lat) >= 20 else None,
            "slow_over_1s": sum(1 for s in self.samples if s[2] > 1.0),
        }


def alive_fake_children() -> list[str]:
    out = subprocess.run(["pgrep", "-af", "convert_hf_to_gguf.py|llama-quantize"], capture_output=True, text=True, check=False).stdout
    return [ln for ln in out.splitlines() if "e2e_export_async_sandbox" not in ln and "pgrep" not in ln]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7897)
    ap.add_argument("--label", default="after")
    ap.add_argument("--src", default=str(ROOT / "src"), help="src dir of the code under test")
    ap.add_argument("--seconds", type=float, default=10.0, help="how long each fake convert/quantize step sleeps")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    fixed = args.label != "before"

    from playwright.sync_api import sync_playwright

    shots = Path(os.environ.get("FTS_QA_SHOTS") or ROOT / ".tmp" / "qa-shots") / "export-async"
    shots.mkdir(parents=True, exist_ok=True)
    sandbox = ROOT / ".tmp" / f"export-sandbox-{args.label}"
    shutil.rmtree(sandbox, ignore_errors=True)
    for d in ("cwd", "home", "fts", "llama/build/bin"):
        (sandbox / d).mkdir(parents=True)
    (sandbox / "llama" / "convert_hf_to_gguf.py").write_text(CONVERT)
    quant = sandbox / "llama" / "build" / "bin" / "llama-quantize"
    quant.write_text(QUANTIZE)
    quant.chmod(0o755)

    env = {
        **os.environ,
        "HOME": str(sandbox / "home"), "FTS_ROOT": str(sandbox / "fts"),
        "FTS_DB": str(sandbox / "cwd" / "data" / "finetune_studio.db"),
        "PYTHONPATH": args.src, "CUDA_VISIBLE_DEVICES": "", "FTS_IDLE_TIMEOUT": "0",
        "LLAMA_CPP_DIR": str(sandbox / "llama"), "FAKE_CONVERT_SECONDS": str(args.seconds),
        "FAKE_QUANT_SECONDS": str(args.seconds),
    }
    env.pop("FTS_SKIP_EXPORT", None)
    py = sys.executable
    seeded = subprocess.run([py, "-c", SEED, str(sandbox / "run-out")], cwd=sandbox / "cwd", env=env,
                            capture_output=True, text=True, check=False)
    if seeded.returncode:
        print(seeded.stdout, seeded.stderr)
        return 2
    pid = seeded.stdout.strip().splitlines()[-1]
    server = subprocess.Popen([py, "-m", "uvicorn", "finetune_studio.webui.app:app", "--host", "127.0.0.1", "--port", str(args.port)],
                              cwd=sandbox / "cwd", env=env, stdout=(sandbox / "server.log").open("w"), stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{args.port}"
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{' — ' + detail if detail and not ok else ''}")

    hammer = Hammer([f"{base}/api/projects", f"{base}/projects/{pid}/export"], timeout=60)
    phases: list[tuple[float, str]] = []
    result: dict = {"label": args.label, "src": args.src, "fake_step_seconds": args.seconds}
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"{base}/api/system/version", timeout=2)
                break
            except OSError:
                time.sleep(0.5)
        else:
            print("sandbox server did not start; see", sandbox / "server.log")
            return 2

        exe = next(iter(sorted((Path.home() / ".cache/ms-playwright").glob("chromium-*/chrome-linux64/chrome"))), None)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=str(exe) if exe else None, args=["--no-sandbox", "--disable-gpu"])
            ctx = browser.new_context(viewport={"width": 1440, "height": 1000})
            ctx.add_init_script("try{localStorage.setItem('fts-theme','dark');localStorage.setItem('fts.tutorial.seen','1')}catch(e){}")
            page = ctx.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"{base}/projects/{pid}/export", wait_until="domcontentloaded")
            page.wait_for_selector(".export-run-radio")
            page.check("input[name=gguf-quant][value=f16]")          # f16 + q4_k_m: convert, then quantize
            check("Export button is enabled for the selected run", not page.is_disabled("#export-btn"))
            hammer.start()
            time.sleep(1.0)                                            # baseline latency before the export
            t_click = time.monotonic()
            if fixed:
                page.click("#export-btn")
                page.wait_for_selector("#export-job:not([hidden])", timeout=15000)
                check("job panel shows within 15 s of clicking Export", True)
                check("click returned immediately (no held-open request)", time.monotonic() - t_click < 5, f"{time.monotonic() - t_click:.1f}s")
                seen_disabled = page.is_disabled("#export-btn") and page.is_disabled("input[name=gguf-quant][value=q8_0]")
                check("conflicting controls are disabled while the job runs", seen_disabled)
                shot_taken = reload_done = False
                while True:
                    state = page.inner_text("#export-job-state").strip().lower()   # CSS uppercases the pill
                    detail = page.inner_text("#export-job-detail")
                    elapsed = page.inner_text("#export-job-elapsed")
                    phases.append((round(time.monotonic() - t_click, 1), f"{state} | {detail} | {elapsed}"))
                    if not shot_taken and "Quantiz" in detail:
                        page.screenshot(path=str(shots / f"{args.label}-1-running.png"))
                        shot_taken = True
                    if shot_taken and not reload_done:                  # reload mid-job: the panel must come back
                        page.reload(wait_until="domcontentloaded")
                        page.wait_for_selector("#export-job:not([hidden])", timeout=15000)
                        check("page reload re-attaches to the running job", page.is_disabled("#export-btn"),
                              page.inner_text("#export-job-detail"))
                        page.screenshot(path=str(shots / f"{args.label}-2-reattached.png"))
                        reload_done = True
                    if state in ("done", "failed", "cancelled"):
                        break
                    if time.monotonic() - t_click > 120:
                        break
                    time.sleep(0.5)
                names = [p[1].split(" | ")[1].split(" — ")[0] for p in phases]
                check("live phases were shown (Converting and Quantizing)",
                      any("Converting" in n for n in names) and any("Quantiz" in n for n in names), "; ".join(sorted(set(names))))
                check("export finished with status done", state == "done", f"{state}: {page.inner_text('#export-job-error')}")
                page.screenshot(path=str(shots / f"{args.label}-3-done.png"))
                result["export_seconds"] = round(time.monotonic() - t_click, 1)
                rows = json.loads(urllib.request.urlopen(f"{base}/api/projects/{pid}/exports", timeout=10).read())
                check("one DB row per quant, files on disk", sorted(r["quant"] for r in rows) == ["f16", "q4_k_m"]
                      and all(r["status"] == "done" and Path(r["output_path"]).stat().st_size > 0 for r in rows), str(rows)[:200])

                # Scenario B: cancel mid-run through the UI.
                time.sleep(2.0)                                        # page reloads itself after success
                page.goto(f"{base}/projects/{pid}/export", wait_until="domcontentloaded")
                page.wait_for_selector(".export-run-radio")
                page.check("#export-force")
                page.click("#export-btn")
                page.wait_for_selector("#export-job:not([hidden])", timeout=15000)
                page.wait_for_function("document.querySelector('#export-job-detail').textContent.includes('Converting')", timeout=30000)
                page.click("#export-cancel-btn")
                page.wait_for_function("document.querySelector('#export-job-state').textContent.trim() === 'cancelled'", timeout=30000)
                page.screenshot(path=str(shots / f"{args.label}-4-cancelled.png"))
                check("Cancel stops the job (state cancelled)", True)
                check("controls unlock after cancel", not page.is_disabled("#export-btn"))
                time.sleep(1.0)
                check("no orphan convert/quantize processes after cancel", not alive_fake_children(), str(alive_fake_children()))
                # Scenario C: two formats in one click run one after the other (server is single-flight).
                before_ids = {r["id"] for r in json.loads(urllib.request.urlopen(f"{base}/api/projects/{pid}/exports", timeout=10).read())}
                page.goto(f"{base}/projects/{pid}/export", wait_until="domcontentloaded")
                page.wait_for_selector(".export-run-radio")
                page.uncheck("input[name=gguf-quant][value=q4_k_m]")
                page.check("input[name=export-format][value=merged]")
                page.check("#export-force")
                page.click("#export-btn")
                chained: list[dict] = []
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    rows = json.loads(urllib.request.urlopen(f"{base}/api/projects/{pid}/exports", timeout=10).read())
                    chained = [r for r in rows if r["id"] not in before_ids]
                    if {r["format"] for r in chained if r["status"] == "done"} >= {"gguf", "merged"}:
                        break
                    time.sleep(1.0)
                check("two selected formats ran back to back without a 409", {r["format"] for r in chained if r["status"] == "done"} >= {"gguf", "merged"},
                      str([(r["format"], r["status"], r["error"]) for r in chained]))
            else:
                # Old synchronous page: the click holds one POST open and the server stops answering.
                page.click("#export-btn")
                try:
                    page.wait_for_function("document.querySelector('#export-status').textContent.includes('Done') || "
                                           "document.querySelector('#export-status').textContent.startsWith('✗')", timeout=180000)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"export never finished: {exc}")
                result["export_seconds"] = round(time.monotonic() - t_click, 1)
                page.screenshot(path=str(shots / f"{args.label}-1-finished.png"))
            check("no page JS errors", not errors, "; ".join(errors)[:300])
            browser.close()
    finally:
        hammer.stop_flag.set()
        hammer.join(timeout=70)
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        result["hammer"] = hammer.summary()
        result["phases"] = phases[::4]
        result["checks"] = [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks]
        (shots / f"{args.label}-summary.json").write_text(json.dumps(result, indent=2))
        if not args.keep:
            shutil.rmtree(sandbox, ignore_errors=True)
    print(json.dumps(result["hammer"], indent=2))
    failed = [c for c in checks if not c[1]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed; screenshots in {shots}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
