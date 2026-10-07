"""Sandboxed browser E2E for the Guide panel — no real model, no live data.

Starts (1) a fake OpenAI-compatible helper that scripts tool calls, (2) a sandbox uvicorn on its own
port with a temp cwd and redirected HOME / FTS_ROOT / FTS_DB (the live :7860 service and its data are
never touched), seeded with one project whose helper seat points at the fake, then drives Chromium
through the Guide: ask → tool cards stream in → navigate + highlight + pre-fill happen on the page →
conversation survives SPA navigation and a hard reload; forced-final-answer and error paths; dark and
light screenshots; optional ``window.__audit`` probe.

    PYTHONPATH=src .venv/bin/python tests/e2e_guide_sandbox.py [--port 7896] [--probe PATH]

Screenshots go to ``.tmp/qa-shots/guide/`` (``FTS_QA_SHOTS`` overrides). Exit code 0 only if every check passed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORCED_MARK = "You have used all your tool rounds"


def call(name: str, **arguments: object) -> str:
    return f'<tool_call>{json.dumps({"name": name, "arguments": arguments})}</tool_call>'


TRAIN_STEPS = [
    call("app_help", query="how do I train a model"),
    call("project_overview"),
    call("recommend_training", goal="memorize my product facts"),
    call("navigate", page="training"),
    call("suggest_settings", page="training", settings={"preset": "balanced", "num_epochs": 6, "lora_rank": 64, "learning_rate": "2e-4"}),
    call("highlight", control_id="training.start"),
    ("I opened the Training page and pre-filled the Baseline settings (6 epochs, rank 64, LR 2e-4). "
     "Pick your dataset, review the values, then press **Start training** yourself."),
]


def brain(messages: list[dict]) -> str:
    """Stateless scripted helper: the step is the number of TOOL_RESULTs since the last real user turn."""
    turns = [m for m in messages if m.get("role") == "user"]
    real = [m for m in turns if not str(m.get("content", "")).startswith("TOOL_RESULT")]
    question = str(real[-1]["content"]).lower() if real else ""
    if messages and FORCED_MARK in str(messages[-1].get("content", "")):
        return "Here is what I found: the lookups above cover your question. Ask me a narrower one to go deeper."
    after = turns[turns.index(real[-1]) + 1:] if real else []
    step = sum(1 for m in after if str(m.get("content", "")).startswith("TOOL_RESULT"))
    if "break the helper" in question:
        raise RuntimeError("scripted helper failure")
    if "loop forever" in question:
        return call("app_help", query=f"loop topic {step}")
    if "qa per chunk" in question:
        script = [call("app_help", query="pairs per chunk setting"),
                  call("highlight", control_id="pairs.qa_per_chunk"),
                  "That is **Pairs per chunk** on the Pairs page; 3 is the default. I pointed at it."]
        return script[min(step, len(script) - 1)]
    if "readiness" in question:
        return call("inspect_project_readiness") if step == 0 else "You have 99 approved pairs and 12 sources."
    return TRAIN_STEPS[min(step, len(TRAIN_STEPS) - 1)]


class Stub(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        try:
            text, status = brain(body.get("messages", [])), 200
            time.sleep(0.4)   # make streaming visible: each model turn takes a moment
        except RuntimeError as exc:
            text, status = str(exc), 500
        payload = json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]} if status == 200 else {"error": text}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_: object) -> None:
        pass


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


SEED = r"""
import json, os, sys
from pathlib import Path
from finetune_studio import db
from finetune_studio.data.fs import qa as qa_fs
from finetune_studio.models.manager import get_manager
db.init_db()
p = db.create_project(name="Guide sandbox", base_model="Qwen__Qwen3-4B")
pid = p["id"] if isinstance(p, dict) else p
qa_fs.write_qa_source(pid, {"id": "s1", "filename": "aurora_spec.csv", "chunk_count": 2, "status": "generated"})
qa_fs.write_qa_source(pid, {"id": "s2", "filename": "warranty.md", "chunk_count": 1, "status": "generated"})
for i in range(6):
    qa_fs.write_qa_pair(pid, {"id": f"qa{i}", "source_id": "s1", "question": f"Q{i}?", "answer": f"A{i}.",
                              "chunk_idx": 1, "status": "approved" if i < 4 else "pending"})
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
ds = out / "approved.jsonl"
ds.write_text("\n".join(json.dumps({"messages": [
    {"role": "user", "content": f"What is the warranty of kettle model {i}?"},
    {"role": "assistant", "content": f"Kettle model {i} has a warranty of {i + 1} years."}]}) for i in range(40)) + "\n")
db.create_dataset(pid, "approved-export", str(ds), source="data-prep-export", qa_count=40)
get_manager().upsert_provider(id="local-default", name="Helper (sandbox stub)", kind="openai_compat",
                              model_id="stub", base_url=sys.argv[1])
print(pid)
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7896)
    ap.add_argument("--probe", default=str(Path.home() / ".openclaw/workspace/.tmp/qa-sweep/probe.js"))
    ap.add_argument("--keep", action="store_true", help="leave the sandbox dir in place")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    shots = Path(os.environ.get("FTS_QA_SHOTS") or ROOT / ".tmp" / "qa-shots") / "guide"
    shots.mkdir(parents=True, exist_ok=True)
    sandbox = ROOT / ".tmp" / "guide-sandbox"
    shutil.rmtree(sandbox, ignore_errors=True)
    for d in ("cwd", "home", "fts"):
        (sandbox / d).mkdir(parents=True)

    stub_port = free_port()
    stub = ThreadingHTTPServer(("127.0.0.1", stub_port), Stub)
    threading.Thread(target=stub.serve_forever, daemon=True).start()

    env = {
        **os.environ,
        "HOME": str(sandbox / "home"), "FTS_ROOT": str(sandbox / "fts"),
        "FTS_DB": str(sandbox / "cwd" / "data" / "finetune_studio.db"),
        "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": "", "FTS_IDLE_TIMEOUT": "0",
    }
    py = sys.executable
    seeded = subprocess.run([py, "-c", SEED, f"http://127.0.0.1:{stub_port}/v1", str(sandbox / "datasets")],
                            cwd=sandbox / "cwd", env=env, capture_output=True, text=True, check=False)
    if seeded.returncode:
        print(seeded.stdout, seeded.stderr)
        return 2
    pid = seeded.stdout.strip().splitlines()[-1]
    server = subprocess.Popen([py, "-m", "uvicorn", "finetune_studio.webui.app:app", "--host", "127.0.0.1", "--port", str(args.port)],
                              cwd=sandbox / "cwd", env=env, stdout=(sandbox / "server.log").open("w"), stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{args.port}"
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{' — ' + detail if detail and not ok else ''}")

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

        probe_src = Path(args.probe).read_text() if Path(args.probe).is_file() else None
        exe = next(iter(sorted((Path.home() / ".cache/ms-playwright").glob("chromium-*/chrome-linux64/chrome"))), None)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=str(exe) if exe else None, args=["--no-sandbox", "--disable-gpu"])
            for theme in ("dark", "light"):
                ctx = browser.new_context(viewport={"width": 1440, "height": 900})
                ctx.add_init_script(f"try{{localStorage.setItem('fts-theme','{theme}');localStorage.setItem('fts.tutorial.seen','1')}}catch(e){{}}")
                page = ctx.new_page()
                errors: list[str] = []
                page.on("pageerror", lambda e, sink=errors: sink.append(str(e)))
                page.on("console", lambda m, sink=errors: sink.append(m.text) if m.type == "error" else None)
                run_theme(page, base, pid, theme, shots, probe_src, check, errors)
                ctx.close()
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        stub.shutdown()
        if not args.keep:
            shutil.rmtree(sandbox, ignore_errors=True)

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed; screenshots in {shots}")
    return 1 if failed else 0


def run_theme(page, base, pid, theme, shots, probe_src, check, errors) -> None:
    tag = f"[{theme}] "
    page.goto(f"{base}/projects/{pid}/data", wait_until="domcontentloaded")
    page.wait_for_selector("#sb-guide")
    check(tag + "Guide entry is on the page", page.is_visible("#sb-guide"))
    page.click("#sb-guide")
    page.wait_for_selector("#guide-panel.open")
    check(tag + "panel opens and shows suggestions", page.locator(".guide-chip").count() >= 3)
    page.wait_for_timeout(450)   # let the slide-in transition finish before the screenshot
    page.screenshot(path=str(shots / f"{theme}-1-panel-empty.png"))

    # 1. "how do I train?" — tool cards stream in, then navigate + pre-fill + highlight happen.
    page.fill("#guide-input", "How do I train a model on my documents?")
    t0 = time.time()
    page.click("#guide-send")
    page.wait_for_selector(".guide-tool", timeout=8000)
    first_card_s = time.time() - t0
    check(tag + "first tool card appears before the answer is done", first_card_s < 6 and page.locator(".guide-assistant").count() == 0,
          f"{first_card_s:.1f}s")
    page.wait_for_selector(".guide-assistant", timeout=40000)
    names = page.eval_on_selector_all(".guide-tool-name", "els => els.map(e => e.textContent)")
    check(tag + "all six tools ran in order", names == ["app_help", "project_overview", "recommend_training", "navigate", "suggest_settings", "highlight"], str(names))
    page.wait_for_function("document.querySelectorAll('.guide-effect').length >= 3", timeout=15000)
    check(tag + "navigated to the Training page (SPA, no reload)", page.url.endswith(f"/projects/{pid}/training"), page.url)
    check(tag + "panel stayed open across navigation", page.is_visible("#guide-panel.open"))
    filled = page.locator(".guide-prefilled").count()
    check(tag + "fields were pre-filled and visibly flagged", filled >= 3, f"{filled} flagged")
    check(tag + "preset=balanced, epochs=6 landed", page.input_value("#training-preset") == "balanced" and page.input_value('[name="num_epochs"]') == "6",
          f"{page.input_value('#training-preset')}/{page.input_value('[name=num_epochs]')}")
    check(tag + "Start training was highlighted, not clicked", page.locator("#start-btn.guide-pulse").count() == 1)
    runs = json.loads(urllib.request.urlopen(f"{base}/api/training/runs/{pid}", timeout=5).read())
    check(tag + "no form was submitted (no training run exists)", not (runs.get("runs") if isinstance(runs, dict) else runs), str(runs)[:120])
    page.wait_for_timeout(700)
    page.screenshot(path=str(shots / f"{theme}-2-train-guided.png"))

    # 2. persistence: SPA navigation keeps the conversation; a hard reload restores it.
    before = page.locator(".guide-msg, .guide-tool, .guide-effect").count()
    page.click('a[data-link][href$="/data-prep"] >> nth=0')
    page.wait_for_url(f"**/projects/{pid}/data-prep")
    check(tag + "conversation intact after SPA navigation", page.locator(".guide-msg, .guide-tool, .guide-effect").count() == before)
    # 3. highlight from another page opens the right page first.
    page.fill("#guide-input", "Where is qa per chunk?")
    page.click("#guide-send")
    page.wait_for_selector(".guide-effect:has-text('Pairs per chunk')", timeout=30000)
    check(tag + "highlight pulses the control", page.locator("#prep-qpc.guide-pulse").count() == 1 or page.locator(".guide-pulse").count() >= 1)
    page.screenshot(path=str(shots / f"{theme}-3-highlight.png"))
    page.click('a[data-link][href$="/rag"] >> nth=0')
    page.fill("#guide-input", "Where is qa per chunk? (again)")
    page.click("#guide-send")
    page.wait_for_url(f"**/projects/{pid}/data-prep", timeout=30000)
    check(tag + "highlight from another page navigates there first", page.url.endswith("/data-prep"))
    page.wait_for_selector("#prep-qpc.guide-pulse", timeout=8000)

    # 4. readiness reply is the server's authoritative summary.
    page.fill("#guide-input", "readiness check please")
    page.click("#guide-send")
    page.wait_for_function("[...document.querySelectorAll('.guide-assistant .guide-body')].some(e => e.textContent.includes('2 source(s)'))", timeout=30000)
    check(tag + "readiness answer is the tool summary, not the helper's invented numbers",
          "99 approved" not in page.inner_text("#guide-msgs"))

    # 5. forced final answer + error path.
    page.fill("#guide-input", "loop forever")
    page.click("#guide-send")
    page.wait_for_function("[...document.querySelectorAll('.guide-assistant .guide-body')].some(e => e.textContent.includes('Here is what I found'))", timeout=90000)
    check(tag + "round limit still ends with an assistant message", True)
    page.fill("#guide-input", "please break the helper")
    page.click("#guide-send")
    page.wait_for_selector(".guide-error", timeout=30000)
    check(tag + "helper failure is shown as an error", "scripted helper failure" in page.inner_text(".guide-error .guide-body") or "500 Server Error" in page.inner_text(".guide-error .guide-body"),
          page.inner_text(".guide-error"))
    page.screenshot(path=str(shots / f"{theme}-4-final-and-error.png"))

    # Hard reload (full page load): history and open state come back from sessionStorage.
    # Done last on purpose: a full load of a page whose inline script has a column-0 `const` makes the
    # SPA's later re-entry of that page fall back to a full load (pre-existing spa.js behaviour).
    users_before = page.locator(".guide-user").count()
    page.reload(wait_until="domcontentloaded")
    page.wait_for_selector("#guide-panel.open", timeout=8000)
    check(tag + "conversation restored after a hard reload", users_before >= 4 and page.locator(".guide-user").count() == users_before)

    # 6. chat page: Agent radio opens the same panel.
    page.goto(f"{base}/projects/{pid}/chat?mode=agent", wait_until="domcontentloaded")
    page.wait_for_selector("#guide-panel.open", timeout=8000)
    check(tag + "Chat page Agent mode opens the Guide", page.is_checked('input[name="chat-mode"][value="agent"]'))
    page.screenshot(path=str(shots / f"{theme}-5-chat-agent.png"))

    # 7. audit probe + console cleanliness (panel open on the training page).
    page.goto(f"{base}/projects/{pid}/training", wait_until="domcontentloaded")
    page.wait_for_selector("#guide-panel.open", timeout=8000)
    page.wait_for_timeout(1200)
    if probe_src:
        page.evaluate("(s) => { window.__audit = () => eval(s); }", probe_src)
        res = page.evaluate("() => window.__audit()")
        for _ in range(2):
            if not res.get("partial"):
                break
            page.wait_for_timeout(1300)
            res = page.evaluate("() => window.__audit()")
        guide_issues = {k: [i for i in (res.get(k) or []) if "guide" in json.dumps(i).lower()]
                        for k in ("overflow", "clipped", "tiny", "low", "cryptic")}
        check(tag + "window.__audit has no findings on the Guide", not any(guide_issues.values()), json.dumps(guide_issues)[:300])
        print(f"      audit {theme}: clean={res.get('clean')} counts={res.get('counts')}")
    real = [e for e in errors if "favicon" not in e and "fonts.g" not in e and "Failed to load resource" not in e]
    check(tag + "no JS errors in the console", not real, "; ".join(real)[:300])


if __name__ == "__main__":
    sys.exit(main())
