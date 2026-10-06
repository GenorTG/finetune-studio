"""Finetune Studio — user-style walkthrough: real browser, real clicks, real GPU.

Mimics a person using the app, phase by phase (see tests/E2E_MANUAL_GUIDE.md for the human version of the
same path: pages, order, what to wait for, what to check). Every step takes a screenshot; a background
monitor prints GPU/VRAM, helper/training state and journal errors while long jobs run.

    FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_user_walkthrough.py --phase create
    FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_user_walkthrough.py --phase upload --phase prep ...
    FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_user_walkthrough.py --list

State (project id) lives in .tmp/manual-e2e/state.json so phases can be run one at a time and reviewed.
It mutates live data: it creates a throwaway project (name starts with ``ux-walk``) and the `cleanup`
phase deletes it with its models/exports. Never point it at a project you care about.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from playwright.async_api import Page, async_playwright

REPO = Path(__file__).resolve().parents[1]
BASE = os.environ.get("FTS_BASE", "http://127.0.0.1:7860").rstrip("/")
WORK = REPO / ".tmp" / "manual-e2e"
SHOTS = Path(os.environ.get("FTS_QA_SHOTS", REPO / ".tmp" / "qa-shots")) / "manual-e2e"
STATE = WORK / "state.json"
CORPUS = REPO / "tests" / "fixtures" / "coverage_docs"
PROJECT_NAME = "ux-walk-1"
BASE_MODEL_REPO = "Qwen/Qwen3-0.6B"


# ── tiny helpers ────────────────────────────────────────────────────────────
def now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def api(path: str, method: str = "GET", body: dict | None = None, timeout: float = 60.0) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(**kw: Any) -> dict:
    WORK.mkdir(parents=True, exist_ok=True)
    st = {**load_state(), **kw}
    STATE.write_text(json.dumps(st, indent=2))
    return st


def gpu_used_mib() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=False).stdout.strip()
        mem, util = [x.strip() for x in out.split(",")]
        return f"VRAM {int(mem) / 1024:.1f} GiB, util {util}%"
    except Exception:  # noqa: BLE001 — a monitor line must never break the walkthrough
        return "VRAM ?"


class Monitor:
    """Background watcher: prints a line whenever GPU/helper/training state changes, and any journal error."""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._last = ""
        self._journal_since = time.strftime("%Y-%m-%d %H:%M:%S")
        self.errors: list[str] = []

    def _snapshot(self) -> str:
        parts = [gpu_used_mib()]
        try:
            inf = api("/api/inference/status", timeout=5)
            if inf.get("loaded"):
                off = inf.get("offload") or {}
                parts.append(f"model {Path(inf.get('model') or '').name} "
                             f"({off.get('mode', '?')} {off.get('gpu_layers', '?')}/{off.get('total_layers', '?')})")
            else:
                parts.append("no model")
            tr = api("/api/training/status", timeout=5)
            if tr.get("status") not in (None, "idle"):
                parts.append(f"train {tr.get('status')} {tr.get('step')}/{tr.get('total_steps')} loss={tr.get('loss')}")
        except Exception as e:  # noqa: BLE001
            parts.append(f"api? {type(e).__name__}")
        return " | ".join(parts)

    def _journal(self) -> list[str]:
        try:
            out = subprocess.run(
                ["journalctl", "--user", "-u", "finetune-studio", "--since", self._journal_since, "--no-pager", "-o", "cat"],
                capture_output=True, text=True, timeout=10, check=False).stdout
        except Exception:  # noqa: BLE001
            return []
        self._journal_since = time.strftime("%Y-%m-%d %H:%M:%S")
        bad = ("Traceback", "ERROR", "CUDA error", "out of memory", "OutOfMemory", "ABRT", "Main process exited")
        return [ln[:200] for ln in out.splitlines() if any(b in ln for b in bad)]

    async def _run(self, every: float) -> None:
        while True:
            snap = self._snapshot()
            if snap != self._last:
                log("monitor: " + snap)
                self._last = snap
            for ln in self._journal():
                self.errors.append(ln)
                log("monitor JOURNAL: " + ln)
            await asyncio.sleep(every)

    def start(self, every: float = 5.0) -> None:
        self._task = asyncio.create_task(self._run(every))

    def stop(self) -> None:
        if self._task:
            self._task.cancel()


class Walk:
    """One browser page + step logging + console-error capture."""

    def __init__(self, page: Page, phase: str) -> None:
        self.page, self.phase, self.n = page, phase, 0
        self.js_errors: list[str] = []
        page.on("pageerror", lambda e: self.js_errors.append(str(e)[:200]))
        page.on("console", lambda m: self.js_errors.append("console.error: " + m.text[:160]) if m.type == "error" else None)

    async def goto(self, path: str, settle: float = 1.2) -> None:
        log(f"open {path}")
        await self.page.goto(BASE + path, wait_until="networkidle")
        await self.page.wait_for_timeout(int(settle * 1000))

    async def shot(self, name: str, full: bool = False) -> str:
        self.n += 1
        SHOTS.mkdir(parents=True, exist_ok=True)
        path = SHOTS / f"{self.phase}-{self.n:02d}-{name}.png"
        await self.page.screenshot(path=str(path), full_page=full)
        log(f"shot {path.name}")
        return str(path)

    async def text(self, selector: str) -> str:
        return (await self.page.inner_text(selector)).strip()

    async def wait_for(self, what: str, predicate: Callable[[], Awaitable[bool]], timeout: float, every: float = 3.0) -> bool:
        """Poll ``predicate`` like a patient user; log progress every ~30 s; False on timeout (caller decides)."""
        start, last_log = time.time(), 0.0
        while time.time() - start < timeout:
            if await predicate():
                log(f"done waiting for {what} after {time.time() - start:.0f}s")
                return True
            if time.time() - last_log > 30:
                log(f"still waiting for {what} ({time.time() - start:.0f}s)")
                last_log = time.time()
            await asyncio.sleep(every)
        log(f"TIMEOUT waiting for {what} after {timeout:.0f}s")
        return False


class Results:
    def __init__(self) -> None:
        self.checks: list[tuple[bool, str]] = []

    def check(self, ok: bool, what: str) -> bool:
        self.checks.append((ok, what))
        log(("PASS " if ok else "FAIL ") + what)
        return ok


R = Results()


def pid() -> str:
    p = load_state().get("pid", "")
    if not p:
        raise SystemExit("no project yet: run --phase create first")
    return p


# ── the test corpus: fictional facts in every common document type ─────────
# Ground truth the later phases check against: (question, keywords that must appear in a correct answer).
FACTS: list[tuple[str, list[str], str]] = [
    ("How long does the Aurora Kettle warranty last?", ["three years", "3 years"], "aurora_warranty_policy.docx"),
    ("How many mirrors does the Orrin Deep beacon use?", ["nine", "9"], "tidewarden_logbook.pdf"),
    ("How many documents can Meridian Sync 4.2 store in offline mode?", ["5000", "5,000"], "meridian_features.html"),
    ("What is the power of the 2025 Aurora Kettle?", ["2200"], "aurora_spec_table.csv"),
    ("Who is the harbour master of Harrowgate quay?", ["Ilsabet Corr"], "harrowgate_facts.xlsx"),
    ("How many crowns per annum is a Ledger-Keeper paid?", ["1875"], "vaelindrath_charter.txt"),
    ("Between which hours does Quiet Hours pause background syncing?", ["22:00"], "meridian_release_notes_v4.txt"),
    ("What must never be used to descale the kettle?", ["vinegar"], "aurora_kettle_support_handbook.txt"),
    ("Who gave the keynote at the Aurora launch event?", ["Maren Voss"], "aurora_launch.pptx"),
    ("What is the seal of the Salt-Speaker carved from?", ["grey coral"], "salt_speaker_seal.rtf"),
    ("What is the dock fee at Port Selene per day?", ["14"], "port_selene_fees.xls"),
    ("How many days does the Frostmonth festival last?", ["eleven", "11"], "frostmonth_festival.doc"),
]


def make_corpus() -> list[Path]:
    """Create (idempotently) a mixed-format corpus: the repo's 5 text fixtures + DOCX, PDF, HTML, CSV, XLSX."""
    import shutil

    import docx
    import openpyxl

    out = WORK / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    for f in CORPUS.iterdir():
        shutil.copy(f, out / f.name)
    d = docx.Document()
    d.add_heading("Aurora Kettle Warranty Policy", 1)
    d.add_paragraph("The Aurora Kettle warranty lasts three years from the purchase date. To claim, you must provide the "
                    "receipt and the serial number printed under the base. The warranty is void if the kettle was ever "
                    "descaled with vinegar, because vinegar destroys the silicone seal of the lid hinge.")
    d.add_heading("Returns", 2)
    d.add_paragraph("Unused kettles may be returned within thirty days; the return label is emailed by support.")
    d.save(out / "aurora_warranty_policy.docx")
    log_doc = docx.Document()
    log_doc.add_heading("Tide-Warden Logbook, Orrin Deep", 1)
    log_doc.add_paragraph("The Orrin Deep beacon was first lit on the 14th of Frostmonth in year 212. The beacon uses nine "
                          "polished mirrors to throw its light across the strait. The keeper on night watch records the "
                          "oil level every second hour.")
    log_doc.save(WORK / "tidewarden_logbook.docx")
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(out), str(WORK / "tidewarden_logbook.docx")],
                   capture_output=True, timeout=120, check=False)
    (out / "meridian_features.html").write_text(
        "<html><head><title>Meridian Sync 4.2</title></head><body><h1>Meridian Sync 4.2</h1>"
        "<p>Version 4.2 adds offline mode. Offline mode stores up to 5000 documents locally and syncs them when the "
        "connection returns.</p><h2>Known limits</h2><ul><li>Files over 200 MB are never stored offline.</li></ul>"
        "</body></html>", encoding="utf-8")
    (out / "aurora_spec_table.csv").write_text(
        "model,capacity_l,power_w,cord_m\nAurora 2023,1.2,1800,0.60\nAurora 2025,1.6,2200,0.75\n", encoding="utf-8")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Harrowgate"
    ws.append(["fact", "value"])
    ws.append(["Harrowgate quay population", 8412])
    ws.append(["Harbour master", "Ilsabet Corr"])
    ws.append(["Quay opened in year", 190])
    wb.save(out / "harrowgate_facts.xlsx")
    import pptx
    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Aurora Kettle launch event"
    slide.placeholders[1].text = "Held in Port Selene on 9 March. The keynote was given by Dr. Maren Voss."
    prs.save(out / "aurora_launch.pptx")
    (out / "salt_speaker_seal.rtf").write_text(
        r"{\rtf1\ansi\deff0 {\fonttbl{\f0 Times;}}\f0 The seal of the Salt-Speaker is carved from grey coral.\par "
        r"It is worn on a chain during every election.\par}", encoding="utf-8")
    import xlwt
    xl = xlwt.Workbook()
    sh = xl.add_sheet("Fees")
    for i, row in enumerate([("item", "crowns"), ("Port Selene dock fee per day", 14), ("Pilot fee", 30)]):
        for j, cell in enumerate(row):
            sh.write(i, j, cell)
    xl.save(str(out / "port_selene_fees.xls"))
    festival = docx.Document()
    festival.add_paragraph("The Frostmonth festival lasts eleven days and ends with the lighting of the Orrin Deep beacon.")
    festival.save(WORK / "frostmonth_festival.docx")
    subprocess.run(["soffice", "--headless", "--convert-to", "doc", "--outdir", str(out), str(WORK / "frostmonth_festival.docx")],
                   capture_output=True, timeout=120, check=False)
    return sorted(p for p in out.iterdir() if p.is_file())


# ── phases ──────────────────────────────────────────────────────────────────
async def phase_create(w: Walk) -> None:
    """Projects page → NEW PROJECT form → CREATE. Expect the card to appear; the page stays on /projects."""
    await w.goto("/projects")
    await w.shot("projects-empty-or-list")
    await w.page.click("#new-project-btn")
    await w.page.fill("input[placeholder^='e.g. Helios']", PROJECT_NAME)
    await w.page.fill("textarea[placeholder^='What is this']", "User-style walkthrough: fictional Aurora/Vaelindrath documents")
    await w.shot("new-project-form")
    await w.page.click("text=CREATE >> nth=0")
    await w.page.wait_for_selector(f"text={PROJECT_NAME}", timeout=15000)
    await w.shot("project-card")
    projects = api("/api/projects")
    mine = [p for p in projects if p["name"] == PROJECT_NAME]
    R.check(len(mine) == 1, f"project '{PROJECT_NAME}' exists exactly once via API")
    if mine:
        save_state(pid=mine[0]["id"])
    await w.page.click(f"text={PROJECT_NAME}")
    await w.page.wait_for_load_state("networkidle")
    await w.shot("project-overview")
    R.check(f"/projects/{mine[0]['id']}" in w.page.url, "clicking the card opens the project overview")


async def phase_upload(w: Walk) -> None:
    """File library: UPLOAD the mixed corpus like a user, wait for parsing, check every file parsed."""
    files = make_corpus()
    R.check(len(files) >= 14, f"corpus has {len(files)} files (txt, md, docx, doc, pdf, html, csv, xlsx, xls, pptx, rtf)")
    await w.goto(f"/projects/{pid()}/data")
    await w.shot("library-empty")
    await w.page.click("#fb-upload-toggle")
    await w.page.set_input_files("#fb-upload-input", [str(f) for f in files])
    await w.shot("files-chosen")
    await w.page.click("#fb-upload-form button[type=submit]")   # choosing files does not start the upload; UPLOAD does
    await w.shot("upload-started")

    async def all_listed() -> bool:
        return len(api(f"/api/projects/{pid()}/data-prep/sources").get("sources", [])) >= len(files)

    R.check(await w.wait_for("all files listed as sources", all_listed, 120), "every uploaded file became a source")

    async def all_parsed() -> bool:
        srcs = api(f"/api/projects/{pid()}/data-prep/sources").get("sources", [])
        return len(srcs) >= len(files) and all(s.get("status") == "ready" for s in srcs)

    R.check(await w.wait_for("every file parsed (status ready)", all_parsed, 180), "every file parsed")
    await w.page.reload(wait_until="networkidle")
    await w.shot("library-parsed", full=True)
    srcs = api(f"/api/projects/{pid()}/data-prep/sources")["sources"]
    for sdef in srcs:
        log(f"  {sdef['filename']:42s} {sdef.get('status'):8s} parser={sdef.get('parser'):10s} chars={sdef.get('char_count'):5} chunks={sdef.get('chunk_count')}")
    empty = [s["filename"] for s in srcs if not s.get("char_count")]
    R.check(not empty, f"no file parsed to empty text {empty}")
    body = await w.text("body")
    R.check(all(f.name in body for f in files), "the library table lists every file by name")


async def _select_all_parsed_sources(w: Walk) -> int:
    """Tick every row of the 'Parsed sources' table (the one with a STAGE column); returns how many."""
    return await w.page.evaluate("""() => {
        const table = [...document.querySelectorAll('table')].find(t => /STAGE/i.test(t.innerText));
        const boxes = [...table.querySelectorAll('tbody input[type=checkbox]')];
        boxes.forEach(b => { if (!b.checked) b.click(); });
        return boxes.length;
    }""")


async def _stage_texts(w: Walk) -> dict[str, str]:
    return await w.page.evaluate("""() => {
        const table = [...document.querySelectorAll('table')].find(t => /STAGE/i.test(t.innerText));
        const out = {};
        table.querySelectorAll('tbody tr').forEach(tr => { const c = tr.querySelectorAll('td'); out[c[1].innerText.trim()] = c[2].innerText.trim() + ' | pairs ' + c[5].innerText.trim(); });
        return out;
    }""")


async def phase_prep(w: Walk) -> None:
    """Pairs page: GENERATE PAIRS FOR SELECTED on every parsed source; the helper model mines Q&A while we watch."""
    await w.goto(f"/projects/{pid()}/data-prep")
    n = await _select_all_parsed_sources(w)
    R.check(n >= 10, f"selected {n} parsed sources")
    await w.shot("sources-selected")
    await w.page.click("#dp-generate-selected")
    await w.page.wait_for_timeout(2500)
    await w.shot("generate-clicked")
    started = time.time()

    async def finished() -> bool:
        srcs = api(f"/api/projects/{pid()}/data-prep/sources")["sources"]
        return all(x["status"] in ("generated", "generated_incomplete", "error", "failed") for x in srcs)

    R.check(await w.wait_for("every source mined by the helper", finished, 1200, every=10), "helper finished every source")
    log(f"mining took {time.time() - started:.0f}s for {n} sources")
    await w.page.click("button:has-text('REFRESH') >> nth=0")        # the table is not live: a user presses REFRESH
    await w.page.wait_for_timeout(1500)
    await w.shot("generation-finished", full=True)
    srcs = api(f"/api/projects/{pid()}/data-prep/sources")["sources"]
    for x in srcs:
        log(f"  {x['filename']:36s} {x['status']:20s} stage={x['stage']:15s} pairs={x['pairs_total']} "
            f"(appr {x['pairs_approved']}, pend {x['pairs_pending']}) coverage={x['coverage_pct']}%")
    R.check(all(x["pairs_total"] > 0 for x in srcs), "every source produced at least one pair")
    qa = api(f"/api/projects/{pid()}/data-prep/qa")["items"]
    helper_pairs = [q for q in qa if q.get("status") == "pending"]
    R.check(len(helper_pairs) >= len(srcs), f"{len(helper_pairs)} helper-written pairs are waiting for review (pending)")
    blank = [q for q in qa if not str(q.get("question", "")).strip() or not str(q.get("answer", "")).strip()]
    R.check(not blank, "no pair with an empty question or answer")
    midword = [q for q in qa if str(q.get("chunk_text", "")).split("\n", 1)[0][:2].islower() and len(q.get("chunk_text", "")) > 0
               and q.get("chunk_text", "")[:1].islower() and " " not in q.get("chunk_text", "")[:3]]
    R.check(not midword, f"no chunk opens with a word fragment ({len(midword)} found)")
    answers = " || ".join(str(q.get("answer", "")) for q in qa).lower()
    got, missing = [], []
    for question, keys, fname in FACTS:
        (got if any(k.lower() in answers for k in keys) else missing).append(f"{fname}: {keys[0]}")
    log(f"facts present in some pair's answer: {len(got)}/{len(FACTS)}; missing: {missing}")
    R.check(len(got) >= len(FACTS) - 3, "most ground-truth facts appear in at least one generated answer")


PHASES: dict[str, Callable[[Walk], Awaitable[None]]] = {"create": phase_create, "upload": phase_upload, "prep": phase_prep}


async def run(phases: list[str], headed: bool) -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    mon = Monitor()
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not headed, args=["--no-sandbox", "--disable-gpu"])
        ctx = await browser.new_context(viewport={"width": 1440, "height": 1000})
        await ctx.add_init_script("try{localStorage.setItem('fts.tutorial.seen','1')}catch(e){}")
        mon.start()
        for name in phases:
            page = await ctx.new_page()
            w = Walk(page, name)
            log(f"=== phase {name} ===")
            try:
                await PHASES[name](w)
            except Exception as e:  # noqa: BLE001
                R.check(False, f"phase {name} crashed: {type(e).__name__}: {str(e)[:300]}")
                await w.shot("crash")
            R.check(not w.js_errors, f"phase {name}: no JS/console errors {w.js_errors[:3] if w.js_errors else ''}")
            await page.close()
        mon.stop()
        await browser.close()
    R.check(not mon.errors, f"journal clean during the run {mon.errors[:2] if mon.errors else ''}")
    failed = [what for ok, what in R.checks if not ok]
    log(f"RESULT: {len(R.checks) - len(failed)}/{len(R.checks)} passed" + (f"; FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", action="append", default=[], help="phase to run (repeatable, in order)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--headed", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("\n".join(f"{k}: {(v.__doc__ or '').strip().splitlines()[0]}" for k, v in PHASES.items()))
        return 0
    if os.environ.get("FTS_ALLOW_LIVE_E2E") != "1":
        print("Refusing live walkthrough; set FTS_ALLOW_LIVE_E2E=1 to opt in (it mutates live data).", file=sys.stderr)
        return 2
    unknown = [p for p in a.phase if p not in PHASES]
    if unknown:
        print(f"unknown phase(s) {unknown}; known: {list(PHASES)}", file=sys.stderr)
        return 2
    return asyncio.run(run(a.phase, a.headed))


if __name__ == "__main__":
    raise SystemExit(main())
