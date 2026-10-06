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
        page.on("response", lambda r: log(f"HTTP {r.status} {r.request.method} {r.url[len(BASE):][:110]}") if r.status >= 400 else None)

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
    try:
        await w.page.wait_for_url(f"**/projects/{mine[0]['id']}*", timeout=10000)   # SPA navigation is not instant
    except Exception as e:  # noqa: BLE001
        log(f"no navigation within 10 s: {type(e).__name__}")
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
    midword = []
    for q in qa:
        first = str(q.get("chunk_text", "")).split(None, 1)[:1]
        if q.get("chunk_idx", 1) > 1 and first and len(first[0]) <= 2 and first[0].isalpha() and first[0].lower() not in ("a", "i", "an", "of", "to", "in", "on", "or", "is", "it", "at", "by"):
            midword.append(first[0])
    R.check(not midword, f"no later chunk opens with a word fragment {midword[:3]}")
    answers = " || ".join(str(q.get("answer", "")) for q in qa).lower()
    got, missing = [], []
    for question, keys, fname in FACTS:
        (got if any(k.lower() in answers for k in keys) else missing).append(f"{fname}: {keys[0]}")
    log(f"facts present in some pair's answer: {len(got)}/{len(FACTS)}; missing: {missing}")
    R.check(len(got) >= len(FACTS) - 3, "most ground-truth facts appear in at least one generated answer")


async def phase_review(w: Walk) -> None:
    """Pairs page: review like a curator — reject the vague scanned-memo pairs, approve the rest, check counts."""
    await w.goto(f"/projects/{pid()}/data-prep")
    await w.page.click("#dp-filter-pending")
    await w.page.wait_for_timeout(800)
    await w.shot("pending-list", full=True)
    before = api(f"/api/projects/{pid()}/data-prep/qa")["items"]
    pend = [q for q in before if q["status"] == "pending"]
    log(f"{len(pend)} pending of {len(before)} total before review")
    vague = w.page.locator("#dp-results tr, table tr", has_text="committee")
    n_vague = await vague.count()
    log(f"rows mentioning the vague committee memo: {n_vague}")
    for k in range(n_vague):
        await vague.nth(k).locator("input[type=checkbox]").first.check()
    if n_vague:
        await w.page.click("#dp-bulk-reject")
        await w.page.wait_for_timeout(1200)
        await w.shot("vague-rejected")
    await w.page.click("#dp-approve-all-pending")
    await w.page.wait_for_timeout(2500)
    await w.shot("approved-all", full=True)
    after = api(f"/api/projects/{pid()}/data-prep/qa")["items"]
    st = {k: sum(1 for q in after if q["status"] == k) for k in ("approved", "pending", "rejected")}
    log(f"after review: {st}")
    R.check(st["pending"] == 0, "no pairs left pending after APPROVE ALL PENDING")
    R.check(st["rejected"] >= n_vague > 0, f"the {n_vague} vague pairs were rejected, not approved")
    R.check(st["approved"] >= 30, f"{st['approved']} approved pairs")


async def phase_rag(w: Walk) -> None:
    """RAG page: QUICK INDEX, then SEARCH like a user and check the right document is on top."""
    await w.goto(f"/projects/{pid()}/rag")
    await w.shot("rag-before")
    await w.page.click("#quick-index-btn")

    async def built() -> bool:
        try:
            d = api(f"/api/projects/{pid()}/rag/build/status")
            return bool(d.get("ok", True)) and not d.get("building") and int(d.get("chunks") or d.get("chunk_count") or 0) > 0
        except Exception:  # noqa: BLE001
            return False

    ok = await w.wait_for("RAG index built", built, 180, every=4)
    R.check(ok, "quick index finished")
    await w.page.reload(wait_until="networkidle")
    await w.shot("rag-built", full=True)
    hits_ok = 0
    for question, keys, fname in FACTS[:6]:
        await w.page.fill("#q-text", question)
        await w.page.click("#q-btn")
        await w.page.wait_for_timeout(3500)
        body = (await w.text("body")).lower()
        found = any(k.lower() in body for k in keys)
        hits_ok += found
        log(f"search '{question[:50]}' -> {'hit' if found else 'MISS'} ({keys[0]})")
    await w.shot("rag-search-results")
    R.check(hits_ok >= 5, f"RAG search surfaced the expected fact for {hits_ok}/6 questions")


async def phase_export(w: Walk) -> None:
    """Pairs page: EXPORT APPROVED -> TRAINING with retrieved-context rows on; check the registered dataset."""
    await w.goto(f"/projects/{pid()}/data-prep")
    grounded = w.page.locator("#dp-grounded")
    if not await grounded.is_checked():
        await grounded.check()
    await w.shot("export-options")
    await w.page.click("#dp-export-approved")
    await w.page.wait_for_timeout(4000)
    await w.shot("export-clicked", full=True)
    ds = api(f"/api/projects/{pid()}/datasets").get("datasets", [])
    R.check(len(ds) >= 1, f"a dataset was registered ({[d['name'] for d in ds]})")
    if ds:
        log(f"dataset: {ds[0]['name']} rows={ds[0]['qa_count']}")
        R.check(ds[0]["qa_count"] >= 30, "dataset has the approved rows")
        R.check("retrieved context" in ds[0]["name"], "dataset name says how many rows carry retrieved context")


async def phase_model(w: Walk) -> None:
    """Model Library: search the Hub for the small base model and DOWNLOAD it like a user."""
    await w.goto("/models/explore")
    await w.page.fill("#hf-q", BASE_MODEL_REPO.split("/")[1])
    await w.page.click("button:has-text('SEARCH')")
    card = w.page.locator(".hf-card", has_text=BASE_MODEL_REPO).first
    await card.wait_for(timeout=30000)
    await w.shot("search-results")
    await card.locator("button:has-text('DOWNLOAD')").click()
    await w.shot("download-confirm")
    await w.page.click("button:has-text('OK')")      # "Download all files of ...?" — the user confirms
    await w.shot("download-clicked")

    def local_size() -> int:
        try:
            for m in api("/api/hf/local").get("models", []):
                if "Qwen3-0.6B" in m["repo_id"]:
                    return int(m["size_bytes"])
        except Exception:  # noqa: BLE001
            return 0
        return 0

    async def done() -> bool:
        return local_size() > 1_400_000_000

    R.check(await w.wait_for("base model download", done, 240, every=3), f"{BASE_MODEL_REPO} downloaded ({local_size() / 1e9:.2f} GB)")
    api("/api/models/refresh", "POST")
    await w.shot("downloaded", full=True)


async def phase_train(w: Walk) -> None:
    """Training page: pick the base model + this project's dataset + a preset, START, watch the live panel."""
    await w.goto(f"/projects/{pid()}/training")
    base = w.page.locator("#train-base-model")
    opts = await base.locator("option").all_inner_texts()
    choice = next((o for o in opts if "Qwen3-0.6B" in o), None)
    R.check(choice is not None, f"base model dropdown offers Qwen3-0.6B ({len(opts)} models listed)")
    await base.select_option(label=choice)
    await w.page.click("button:has-text('From this project')")
    ds = w.page.locator("#dataset-select")
    dopts = await ds.locator("option").all_inner_texts()
    log(f"dataset options: {dopts}")
    pick = next((o for o in dopts if "sharegpt" in o), None)
    R.check(pick is not None, "this project's exported dataset is selectable")
    await ds.select_option(label=pick)
    await w.page.select_option("#training-preset", "precision")
    await w.page.wait_for_timeout(500)
    vals = await w.page.evaluate("() => [...document.querySelectorAll('input[type=number], input[type=text]')].filter(e => e.offsetParent).map(e => e.value)")
    log(f"fields after choosing the Precision preset: {vals}")
    await w.shot("configured", full=True)
    vram_before = gpu_used_mib()
    await w.page.click("#start-btn")
    await w.page.wait_for_timeout(4000)
    await w.shot("training-started", full=True)
    log(f"before start {vram_before}; helper unloaded by the start? -> {api('/api/inference/status').get('loaded')}")

    async def finished() -> bool:
        return api("/api/training/status").get("status") in ("done", "error", "stopped", "idle")

    shot_at = time.time()
    while not await finished() and time.time() - shot_at < 900:
        await asyncio.sleep(20)
        st = api("/api/training/status")
        log(f"training {st.get('status')} step {st.get('step')}/{st.get('total_steps')} loss={st.get('loss')} eta={st.get('eta')}")
        if time.time() - shot_at > 40 and not (w.n % 4):
            await w.shot("training-progress")
    st = api("/api/training/status")
    await w.shot("training-finished", full=True)
    R.check(st.get("status") == "done", f"training finished: {st.get('status')} step {st.get('step')}/{st.get('total_steps')} loss {st.get('loss')}")
    R.check((st.get("loss") or 9) < 0.5, f"final loss {st.get('loss')} < 0.5 (facts memorised)")
    runs = api(f"/api/training/runs/{pid()}")
    R.check(bool(runs) and runs[0].get("status") == "done", "the run is listed as done on the project")
    if runs:
        save_state(run_id=runs[0]["id"], run_path=runs[0].get("output_path", ""))


async def _results_text(w: Walk) -> str:
    return (await w.page.locator("text=RESULTS").first.locator("xpath=ancestor::div[contains(@class,'card')][1]").inner_text())[:1500]


async def phase_test(w: Walk) -> None:
    """Testing page: run the project quiz on the trained model, then the held-out eval and the RAG-grounded suite."""
    await w.goto(f"/projects/{pid()}/testing")
    model_opts = await w.page.locator("#t-model option").all_inner_texts()
    log(f"model options: {model_opts}")
    suite_opts = await w.page.locator("#t-suite option").all_inner_texts()
    log(f"suite options: {[o[:60] for o in suite_opts]}")
    mine = next((o for o in suite_opts if "sharegpt" in o or "ux-walk" in o or "approved" in o), None)
    R.check(mine is not None, "the project's auto-generated quiz is offered in the suite dropdown")
    if mine:
        await w.page.select_option("#t-suite", label=mine)
        await w.shot("suite-picked")
        await w.page.click("#t-run-btn")
        await w.page.wait_for_timeout(3000)
        await w.shot("quiz-running")

        async def has_result() -> bool:
            return await w.page.locator("#case-scores-summary").count() > 0

        R.check(await w.wait_for("quiz results", has_result, 300, every=4), "quiz produced a result")
        await w.shot("quiz-results", full=True)
        log("RESULTS: " + (await w.text("#case-scores-summary")))
        if await w.page.locator("#case-scores-split").count():
            log("SPLIT:   " + (await w.text("#case-scores-split")))
    await w.page.select_option("#t-eval-kind", index=0)
    await w.page.click("#t-train-eval-btn")
    await w.page.wait_for_timeout(8000)
    await w.shot("heldout-eval", full=True)
    body = (await w.text("body")).replace("\n", " ")
    i = max(body.lower().find("held-out"), 0)
    log("HELD-OUT: " + body[i:i + 350])


async def phase_gguf(w: Walk) -> None:
    """Export page: pick the finished run, keep GGUF + q4_k_m (the recommended default), add q6_k, EXPORT."""
    await w.goto(f"/projects/{pid()}/export")
    run_card = w.page.locator("text=" + load_state().get("run_id", "run")).first
    if await run_card.count():
        await run_card.click()
    await w.shot("run-picked")
    for quant in ("q6_k",):
        box = w.page.locator("label", has_text=quant).locator("input[type=checkbox]").first
        if not await box.is_checked():
            await box.check()
    await w.page.click("#export-btn")
    await w.page.wait_for_timeout(3000)
    await w.shot("export-started")

    async def exported() -> bool:
        runs = api(f"/api/projects/{pid()}/exports")
        text = json.dumps(runs)
        return "q4_k_m" in text.lower() and "q6_k" in text.lower()

    R.check(await w.wait_for("GGUF export", exported, 240, every=4), "q4_k_m and q6_k GGUF files were written")
    await w.page.reload(wait_until="networkidle")
    await w.shot("exports-listed", full=True)
    ggufs = list((REPO / "output" / "projects" / pid()).rglob("*.gguf"))
    log("files: " + ", ".join(f"{g.name} {g.stat().st_size / 1e6:.0f}MB" for g in ggufs))
    R.check(len(ggufs) >= 2, "GGUF files exist on disk")


async def phase_chat(w: Walk) -> None:
    """Chat page: LOAD the exported GGUF, then ask every ground-truth question plain (recall) and see the answers."""
    # The Testing step leaves the merged model resident, and then the inline model dropdown comes up empty (open
    # finding, see HANDOFF): free the GPU first like a user pressing the eject chip.
    api("/api/models/unload", "POST")
    await w.goto(f"/projects/{pid()}/chat")
    opts = await w.page.locator("#chat-inline-model option").all_inner_texts()
    log(f"chat model options: {opts}")
    pick = next((o for o in opts if "q4_k_m" in o.lower()), None)
    R.check(pick is not None, "the exported q4_k_m model is selectable in Chat")
    await w.page.select_option("#chat-inline-model", label=pick)
    await w.page.click("#chat-inline-load-btn")

    async def loaded() -> bool:
        return bool(api("/api/inference/status").get("loaded"))

    R.check(await w.wait_for("model loaded in Chat", loaded, 120, every=2), "model loaded")
    await w.shot("loaded")
    st = api("/api/inference/status")
    log(f"placement: ctx={st.get('n_ctx')} layers={st.get('n_gpu_layers')} offload={st.get('offload')}")
    right = 0
    for question, keys, fname in FACTS:
        await w.page.fill("#chat-input", question)
        await w.page.click("#chat-send")
        await w.page.wait_for_timeout(2500)

        async def answered() -> bool:
            return not await w.page.locator("#chat-stop").is_enabled()

        await w.wait_for("answer", answered, 60, every=1)
        body = (await w.text("body")).lower()
        last = body.rsplit(question.lower()[:30], 1)[-1][:300]
        ok = any(k.lower() in last for k in keys)
        right += ok
        log(f"recall {'OK  ' if ok else 'MISS'} {question[:55]:55s} -> {last[:90].strip()!r}")
    await w.shot("recall-answers", full=True)
    R.check(right >= len(FACTS) // 2, f"model recalls {right}/{len(FACTS)} trained facts without any retrieval")


async def phase_bench(w: Walk) -> None:
    """Benchmarks page: run the offline synthetic knowledge suite on the base model and read the score."""
    await w.goto(f"/projects/{pid()}/benchmarks")
    row = w.page.locator("tr", has_text="Run " + load_state().get("run_id", "")[:8]).first   # the trained run, not the base row
    suite_sel = row.locator("select").first
    suite_opts = await suite_sel.locator("option").all_inner_texts()
    pick = next((o for o in suite_opts if "knowledge" in o.lower() and "offline" in o.lower()), suite_opts[-1])
    log(f"suite: {pick[:80]}")
    await suite_sel.select_option(label=pick)
    await w.shot("configured")
    await row.locator("button:has-text('RUN')").first.click()

    async def scored() -> bool:
        body = await w.text("body")
        return "%" in body.split("Recent scores", 1)[-1][:2500] and "no benchmarks" not in body.lower()

    R.check(await w.wait_for("benchmark score", scored, 240, every=4), "benchmark finished and a score is listed")
    await w.shot("scores", full=True)
    log("SCORES: " + (await w.text("body")).split("Recent scores", 1)[-1][:300].replace("\n", " "))


async def phase_cleanup(w: Walk) -> None:
    """Project overview: DELETE the throwaway project like a user, then verify nothing is left on disk."""
    p = pid()
    await w.goto(f"/projects/{p}")
    await w.shot("before-delete")
    await w.page.click("button:has-text('DELETE')")
    await w.page.wait_for_timeout(600)
    await w.shot("confirm-dialog")
    await w.page.click("button:has-text('OK')")
    await w.page.wait_for_timeout(3000)
    R.check(not [x for x in api("/api/projects") if x["id"] == p], "project is gone from the API")
    leftovers = [str(d) for d in (REPO / "output" / "projects" / p, REPO / "data" / "projects" / p,
                                  Path.home() / ".finetune-studio" / "projects" / p,
                                  Path.home() / ".finetune-studio" / "rag_corpora" / p) if d.exists()]
    R.check(not leftovers, f"no project directories left on disk {leftovers}")
    names = [m["name"] for m in api("/api/models/list")]
    R.check(not [n for n in names if "ux-walk" in n], f"model pickers no longer offer the project's exports ({names})")
    await w.goto("/models/explore")
    api("/api/models/unload", "POST")
    deleted = api("/api/hf/local/Qwen__Qwen3-0.6B", "DELETE")
    log(f"base model removed: {deleted}")
    R.check(not [m for m in api("/api/hf/local")["models"] if "Qwen3-0.6B" in m["repo_id"]], "downloaded base model removed")


PHASES: dict[str, Callable[[Walk], Awaitable[None]]] = {"bench": phase_bench, "cleanup": phase_cleanup, "gguf": phase_gguf, "chat": phase_chat, "test": phase_test, "model": phase_model, "train": phase_train, "create": phase_create, "upload": phase_upload, "prep": phase_prep, "review": phase_review, "rag": phase_rag, "export": phase_export}


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
