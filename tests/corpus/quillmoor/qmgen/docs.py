"""Turn the world into documents: decide what each file says, in which structure and which file type.

Two passes: ``plan()`` lists every document with the file types its structure allows, ``assign_formats()`` picks one
type per document so that every type gets used about equally, then ``build()`` writes the blocks (or the raw text for
code, config, log and chat files) and records a FactMark for every value it put on the page.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import names as N
from .model import Block, Doc, FactMark
from .specs import spec_for
from .world import Fact, World

PROSE = [".docx", ".pdf", ".odt", ".md", ".markdown", ".html", ".htm", ".epub", ".rtf", ".doc", ".txt"]
TABLE = [".xlsx", ".csv", ".tsv", ".ods", ".xls", ".html", ".htm", ".json", ".jsonl", ".xml", ".yaml", ".yml", ".pdf",
         ".docx", ".md"]
IMAGES = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"]
SLIDES = [".pptx", ".odp"]
CONFIG = [".ini", ".cfg", ".conf", ".yaml", ".yml"]
CODE_ERR = [".py", ".js", ".ts"]
CODE_UI = [".tsx", ".jsx"]


@dataclass
class Plan:
    doc_id: str
    kind: str
    exts: list[str]
    payload: dict[str, Any] = field(default_factory=dict)
    ext: str = ""


def slug(s: str) -> str:
    out = "".join(c.lower() if c.isalnum() else "-" for c in s)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")[:60]


class Builder:
    def __init__(self, w: World) -> None:
        self.w = w
        self.r = random.Random(f"docs-{w.seed}-{w.tier}")

    # ── fact writing ──
    def put(self, doc: Doc, f: Fact, style: str = "", *, with_unit: bool = True, old: bool = False) -> str:
        raw = f.old[-1] if old else None
        written = f.write(style, raw, with_unit=with_unit)
        doc.facts.append(FactMark(f.fid, written, superseded=old))
        (f.old_in if old else f.stated_in).append(doc.doc_id)
        return written

    def put_line(self, doc: Doc, f: Fact, line: str, value: str, old: bool = False) -> str:
        """Config / code lines: the mark is the whole line, which holds the value verbatim."""
        assert value in line, (line, value)
        doc.facts.append(FactMark(f.fid, line, superseded=old))
        (f.old_in if old else f.stated_in).append(doc.doc_id)
        return line

    def state(self, doc: Doc, f: Fact, style: str = "", old: bool = False) -> str:
        sp = spec_for(f.etype, f.attr, f.ctx.get("label", ""), f.ctx.get("para", ""), f.ctx.get("code", ""))
        v = self.put(doc, f, style, old=old)
        return self.r.choice(sp.states).format(k=f.ekey, v=v, t=f.ctx.get("t", ""), s=f.ctx.get("s", ""))

    def col(self, f: Fact) -> str:
        return spec_for(f.etype, f.attr, f.ctx.get("label", ""), f.ctx.get("para", ""), f.ctx.get("code", "")).col

    def filler(self, register: str, n: int = 2) -> list[Block]:
        pool = N.FILLER[register]
        return [Block("para", " ".join(self.r.sample(pool, k=min(len(pool), self.r.randint(2, 3))))) for _ in range(n)]

    def dstyle(self) -> str:
        return self.r.choice(["iso", "dmy", "long"])

    def mstyle(self) -> str:
        return self.r.choice(["suffix", "prefix", "symbol"])

    # ── plan ──
    def plan(self) -> list[Plan]:
        w = self.w
        P: list[Plan] = []
        sites = w.e["site"]
        for s in sites:
            sl = slug(s.a["code"])
            P += [Plan(f"site-profile-{sl}", "site_profile", PROSE, {"site": s.key}),
                  Plan(f"fire-notice-{sl}", "site_notice", IMAGES, {"site": s.key}),
                  Plan(f"telemetry-{sl}", "site_config", CONFIG, {"site": s.key}),
                  Plan(f"asset-register-{sl}", "asset_register", TABLE, {"site": s.key}),
                  Plan(f"alarm-log-{sl}", "incident_log", [".log", ".txt"], {"site": s.key}),
                  Plan(f"team-chat-{sl}", "chat_log", [".txt", ".log", ".md", ".json"], {"site": s.key})]
            certified = [p for p in s.a["staff"] if f"person:{p}:certification" in w.facts]
            if certified:
                P.append(Plan(f"competence-matrix-{sl}", "cert_matrix", [".xlsx", ".pdf", ".docx", ".csv", ".pptx", ".odp", ".html"],
                              {"people": certified, "site": s.key}))
            assets = list(s.a["assets"])
            for i in range(0, len(assets), 9):
                P.append(Plan(f"maintenance-plan-{sl}-{i // 9 + 1}", "maintenance_plan", PROSE + SLIDES, {"assets": assets[i:i + 9], "site": s.key}))
        people = [p.key for p in w.e["person"]]
        for i in range(0, len(people), 24):
            P.append(Plan(f"staff-directory-{i // 24 + 1:03d}", "staff_directory", TABLE, {"people": people[i:i + 24]}))
        for a in w.e["asset"]:
            if f"asset:{a.key}:mass_kg" in w.facts:
                P.append(Plan(f"nameplate-{slug(a.key)}", "nameplate", IMAGES, {"asset": a.key}))
        for inc in w.e["incident"]:
            P.append(Plan(f"incident-{slug(inc.key)}", "incident_report", PROSE + [".eml"], {"incident": inc.key}))
            if f"incident:{inc.key}:insurer_ref" in w.facts:
                P.append(Plan(f"claim-letter-{slug(inc.key)}", "insurance_scan", [".pdf"], {"incident": inc.key}))
        sups = [s.key for s in w.e["supplier"]]
        for i in range(0, len(sups), 8):
            P.append(Plan(f"approved-suppliers-{i // 8 + 1:03d}", "supplier_list", TABLE, {"suppliers": sups[i:i + 8]}))
        for c in w.e["contract"]:
            P.append(Plan(f"contract-{slug(c.key)}", "contract_doc", [".pdf", ".docx", ".doc", ".rtf", ".odt", ".txt", ".md", ".epub"],
                          {"contract": c.key}))
            if w.get("contract", c.key, "value").old or w.get("contract", c.key, "end_date").old:
                P.append(Plan(f"variation-{slug(c.key)}", "contract_amendment", [".eml", ".docx", ".pdf", ".md", ".txt"], {"contract": c.key}))
        custs = [c.key for c in w.e["customer"]]
        for i in range(0, len(custs), 10):
            P.append(Plan(f"customer-accounts-{i // 10 + 1:03d}", "customer_accounts", TABLE, {"customers": custs[i:i + 10]}))
        parts = [p.key for p in w.e["part"]]
        for i in range(0, len(parts), 20):
            P.append(Plan(f"parts-catalogue-{i // 20 + 1:03d}", "parts_catalog", TABLE + SLIDES, {"parts": parts[i:i + 20]}))
        for p in w.e["policy"]:
            sl = slug(p.a["number"])
            if p.a["revised"]:
                P.append(Plan(f"policy-{sl}-v1", "policy_doc", PROSE, {"policy": p.key, "version": 1}))
                P.append(Plan(f"policy-{sl}-v2", "policy_doc", PROSE, {"policy": p.key, "version": 2}))
                P.append(Plan(f"policy-change-{sl}", "policy_notice", [".eml", ".pptx", ".md", ".docx", ".odp"], {"policy": p.key}))
            else:
                P.append(Plan(f"policy-{sl}", "policy_doc", PROSE, {"policy": p.key, "version": 1}))
        for s in w.e["service"]:
            sl = slug(s.key)
            P += [Plan(f"{sl}-config", "service_config", CONFIG + [".json", ".xml"], {"service": s.key}),
                  Plan(f"{sl}-errors", "service_errors", CODE_ERR, {"service": s.key}),
                  Plan(f"{sl}-board", "service_board", CODE_UI, {"service": s.key}),
                  Plan(f"{sl}-theme", "service_css", [".css"], {"service": s.key}),
                  Plan(f"{sl}-runbook", "runbook", [".md", ".markdown", ".txt", ".html", ".htm", ".docx", ".pdf"], {"service": s.key})]
        for p in w.e["project"]:
            P.append(Plan(f"{slug(p.key)}-update", "project_update", SLIDES + [".eml", ".docx", ".md", ".pdf", ".html"], {"project": p.key}))
        for c in w.e["chemical"]:
            P.append(Plan(f"sds-{slug(c.key)}", "chemical_sds", [".pdf", ".docx", ".html", ".xml", ".odt", ".htm", ".rtf"], {"chemical": c.key}))
        vessels = [v.key for v in w.e["vessel"]]
        half = len(vessels) // 2
        for i in range(0, half, 6):
            P.append(Plan(f"vessel-register-{i // 6 + 1:02d}", "vessel_register", [".xml", ".yaml", ".json", ".csv", ".xlsx"], {"vessels": vessels[i:i + 6]}))
        for v in vessels[half:]:
            P.append(Plan(f"harbour-letter-{slug(v)}", "vessel_scan", [".pdf"], {"vessel": v}))
        n_distract = max(6, len(P) // 14)
        for i in range(n_distract):
            P.append(Plan(f"staff-news-{i + 1:03d}", "distractor", PROSE + [".eml", ".pptx", ".xlsx", ".png", ".jpg"], {"n": i}))
        return P

    def assign_formats(self, plans: list[Plan]) -> None:
        used: Counter[str] = Counter()
        scans = {"insurance_scan", "vessel_scan"}   # image-only PDFs count apart, so text PDFs keep their share
        # narrow choices first, so the flexible documents fill the gaps the narrow ones leave
        for p in sorted(plans, key=lambda p: (len(p.exts), p.doc_id)):
            if p.kind in scans:
                p.ext = ".pdf"
                continue
            low = min(used[e] for e in p.exts)
            p.ext = self.r.choice([e for e in p.exts if used[e] == low])
            used[p.ext] += 1

    # ── build ──
    def build(self, p: Plan) -> Doc:
        fn: Callable[[Plan], Doc] = getattr(self, f"b_{p.kind}")
        return fn(p)

    def doc(self, p: Plan, title: str, subdir: str, **meta: Any) -> Doc:
        d = Doc(p.doc_id, p.kind, p.ext, title, subdir=subdir,
                meta={"author": self.r.choice([x.key for x in self.w.e["person"]]), "company": "Quillmoor Utilities plc", **meta})
        if p.ext in IMAGES:
            d.meta.setdefault("image_style", "notice")
        return d

    def b_site_profile(self, p: Plan) -> Doc:
        w, s = self.w, self.w.by_key[("site", p.payload["site"])]
        d = self.doc(p, f"Site profile — {s.key}", "operations/sites", header="Quillmoor Utilities plc — Asset Information",
                     footer="Uncontrolled when printed")
        g = lambda a: w.get("site", s.key, a)
        kv_attrs = self.r.sample(["code", "island", "commissioned", "capacity"], k=self.r.randint(2, 4))
        d.blocks.append(Block("heading", "At a glance", 2))
        d.blocks.append(Block("kv", pairs=[(self.col(g(a)), self.put(d, g(a))) for a in kv_attrs]))
        d.blocks += self.filler("corporate", 1)
        d.blocks.append(Block("heading", "Operations", 2))
        prose = [self.state(d, g(a)) for a in ("code", "island", "commissioned", "capacity") if a not in kv_attrs]
        prose += [self.state(d, g("manager")), self.state(d, g("headcount")), self.state(d, g("permit"))]
        self.r.shuffle(prose)
        d.blocks.append(Block("para", " ".join(prose[:3])))
        d.blocks += self.filler("engineering", 1)
        if prose[3:]:
            d.blocks.append(Block("para", " ".join(prose[3:])))
        d.blocks.append(Block("note", "Figures in this profile are reviewed every year by the asset information team."))
        return d

    def b_site_notice(self, p: Plan) -> Doc:
        s = self.w.by_key[("site", p.payload["site"])]
        d = self.doc(p, f"FIRE ACTION — {s.key}", "operations/safety", image_style="notice")
        d.blocks += [Block("para", "If you discover a fire, raise the alarm at the nearest call point and leave by the nearest exit."),
                     Block("para", self.state(d, self.w.get("site", s.key, "muster_point"))),
                     Block("bullets", items=["Do not stop to collect belongings.", "Do not re-enter until told it is safe.",
                                             "Report to the fire marshal at the assembly point."])]
        return d

    def b_site_config(self, p: Plan) -> Doc:
        w, s = self.w, self.w.by_key[("site", p.payload["site"])]
        d = self.doc(p, f"telemetry config {s.a['code']}", "it/config")
        poll, ret = w.get("site", s.key, "telemetry_poll_s"), w.get("site", s.key, "retention_days")
        pv, rv = str(poll.raw), str(ret.raw)
        host = f"hist-{s.a['code'].lower()}.qmu.internal"
        if p.ext in (".yaml", ".yml"):
            l1, l2 = f"  poll_interval_seconds: {pv}", f"  retention_days: {rv}"
            d.raw = (f"# Telemetry outstation settings for {s.key} ({s.a['code']})\n# Managed by OT engineering. Do not edit on site.\n"
                     f"site: \"{s.key}\"\ntelemetry:\n{l1}\n  protocol: dnp3\n  outstations: {self.r.randint(4, 40)}\n"
                     f"historian:\n  host: {host}\n{l2}\n  compression: swinging-door\n")
        else:
            l1, l2 = f"poll_interval_seconds = {pv}", f"retention_days = {rv}"
            d.raw = (f"; Telemetry outstation settings for {s.key} ({s.a['code']})\n; Managed by OT engineering. Do not edit on site.\n\n"
                     f"[site]\nname = {s.key}\ncode = {s.a['code']}\n\n[telemetry]\n{l1}\nprotocol = dnp3\n"
                     f"outstations = {self.r.randint(4, 40)}\n\n[historian]\nhost = {host}\n{l2}\ncompression = swinging-door\n")
        self.put_line(d, poll, l1, pv)
        self.put_line(d, ret, l2, rv)
        return d

    def table_doc(self, p: Plan, title: str, subdir: str, cols: list[str], rows: list[list[str]], intro: str = "",
                  caption: str = "") -> Doc:
        d = self.doc(p, title, subdir, title_row=self.r.random() < 0.5)
        if intro and p.ext not in (".csv", ".tsv", ".jsonl"):
            d.blocks.append(Block("para", intro))
        d.blocks.append(Block("table", columns=cols, rows=rows, caption=caption or "Register"))
        return d

    def b_staff_directory(self, p: Plan) -> Doc:
        w = self.w
        cols = ["Name", "Employee ID", "Role", "Site", "Ext.", "Start date"]
        ds = self.dstyle()
        d = self.table_doc(p, "Staff directory", "people", cols, [], "Internal telephone and staff directory. Extensions change; check the chat if in doubt.",
                           "Directory")
        rows = []
        for k in p.payload["people"]:
            g = lambda a, k=k: w.get("person", k, a)
            ext = g("extension")
            rows.append([k, self.put(d, g("employee_id")), self.put(d, g("role")), self.put(d, g("site")),
                         self.put(d, ext, old=bool(ext.old)), self.put(d, g("start_date"), ds)])
        self._fill_rows(d, rows)
        return d

    def _fill_rows(self, d: Doc, rows: list[list[str]]) -> None:
        """Put the rows in the doc's table; workbooks and decks with many rows get them over two sheets / tables."""
        i = next(n for n, b in enumerate(d.blocks) if b.kind == "table")
        t = d.blocks[i]
        if d.ext in (".xlsx", ".xls", ".ods", ".pptx", ".odp") and len(rows) > 14:
            half = len(rows) // 2
            d.blocks[i:i + 1] = [Block("table", columns=t.columns, rows=rows[:half], caption=f"{t.caption} 1"),
                                 Block("table", columns=t.columns, rows=rows[half:], caption=f"{t.caption} 2")]
        else:
            t.rows = rows

    def b_cert_matrix(self, p: Plan) -> Doc:
        w = self.w
        ds = self.dstyle()
        d = self.table_doc(p, f"Competence matrix — {p.payload['site']}", "people/training", ["Name", "Certification", "Expires"], [],
                           "Certificates held by site staff. Renewals are booked by the training coordinator.", "Competence")
        rows = [[k, self.put(d, w.get("person", k, "certification")), self.put(d, w.get("person", k, "cert_expiry"), ds)]
                for k in p.payload["people"]]
        self._fill_rows(d, rows)
        return d

    def b_chat_log(self, p: Plan) -> Doc:
        w, r = self.w, self.r
        s = w.by_key[("site", p.payload["site"])]
        d = self.doc(p, f"#{s.a['code'].lower()}-ops chat export", "comms/chat")
        msgs: list[tuple[str, str]] = []
        staff = s.a["staff"]
        for k in staff:
            ext = w.get("person", k, "extension")
            if ext.old:
                v = self.put(d, ext)
                msgs.append((k, r.choice([f"fyi my desk moved, new ext is {v} (old one is dead)",
                                          f"new extension: {v}. directory is out of date, sorry",
                                          f"you can reach me on {v} now, the old number goes to voicemail"])))
        for _ in range(r.randint(6, 14)):
            msgs.insert(r.randint(0, len(msgs)), (r.choice(staff), r.choice(N.FILLER["casual"])))
        stamps = sorted(f"2025-{r.randint(1, 12):02d}-{r.randint(1, 28):02d} {r.randint(6, 22):02d}:{r.randint(0, 59):02d}" for _ in msgs)
        if p.ext == ".json":
            d.raw = json.dumps({"channel": f"#{s.a['code'].lower()}-ops", "messages": [
                {"ts": t, "user": u, "text": m} for t, (u, m) in zip(stamps, msgs)]}, indent=2, ensure_ascii=False)
        elif p.ext == ".md":
            d.raw = f"# #{s.a['code'].lower()}-ops\n\n" + "\n".join(f"**{u}** ({t[11:]}): {m}  " for t, (u, m) in zip(stamps, msgs)) + "\n"
        elif p.ext == ".log":
            d.raw = "\n".join(f"{t.replace(' ', 'T')}:00 #{s.a['code'].lower()}-ops <{u}> {m}" for t, (u, m) in zip(stamps, msgs)) + "\n"
        else:
            d.raw = "\n".join(f"[{t}] {u}: {m}" for t, (u, m) in zip(stamps, msgs)) + "\n"
        return d

    def b_asset_register(self, p: Plan) -> Doc:
        w = self.w
        s = w.by_key[("site", p.payload["site"])]
        ds = self.dstyle()
        cols = ["Tag", "Type", "Model", "Serial no.", "Installed", "Rating", "Criticality", "Site"]
        d = self.table_doc(p, f"Asset register — {s.key}", "operations/assets", cols, [],
                           "Extract from the computerised maintenance system. Decommissioned assets are not shown.", "Assets")
        rows = []
        for tag in s.a["assets"]:
            g = lambda a, tag=tag: w.get("asset", tag, a)
            rows.append([tag, g("model").ctx["t"], self.put(d, g("model")), self.put(d, g("serial")), self.put(d, g("install_date"), ds),
                         self.put(d, g("rating")), self.put(d, g("criticality")), self.put(d, g("site"))])
        self._fill_rows(d, rows)
        return d

    def b_maintenance_plan(self, p: Plan) -> Doc:
        w = self.w
        ds = self.dstyle()
        d = self.doc(p, f"Planned maintenance — {p.payload['site']}", "operations/maintenance",
                     header="Reliability & Maintenance", footer="Intervals are reviewed after every major failure")
        d.blocks.append(Block("heading", "Scope", 1))
        d.blocks += self.filler("engineering", 1)
        assets = p.payload["assets"]
        in_table = assets[: len(assets) // 2]
        in_prose = assets[len(assets) // 2:]
        if in_table:
            d.blocks.append(Block("heading", "Schedule", 1))
            rows = [[t, self.put(d, w.get("asset", t, "service_interval"), with_unit=False),
                     self.put(d, w.get("asset", t, "last_overhaul"), ds)] for t in in_table]
            d.blocks.append(Block("table", columns=["Tag", "Interval (operating hours)", "Last overhaul"], rows=rows, caption="Schedule"))
        d.blocks.append(Block("heading", "Notes per asset", 1))
        for t in in_prose:
            d.blocks.append(Block("para", self.state(d, w.get("asset", t, "service_interval")) + " " +
                                  self.state(d, w.get("asset", t, "last_overhaul"), ds) + " " + self.r.choice(N.FILLER["engineering"])))
        return d

    def b_nameplate(self, p: Plan) -> Doc:
        w, tag = self.w, p.payload["asset"]
        d = self.doc(p, "QUILLMOOR UTILITIES PLC", "operations/assets/photos", image_style="nameplate")
        g = lambda a: w.get("asset", tag, a)
        d.blocks.append(Block("kv", pairs=[("TAG", tag), ("MODEL", self.put(d, g("model"))), ("SERIAL", self.put(d, g("serial"))),
                                           ("RATED", self.put(d, g("rating"))), ("MASS", self.put(d, g("mass_kg")))]))
        return d

    def b_incident_report(self, p: Plan) -> Doc:
        w, k = self.w, p.payload["incident"]
        g = lambda a: w.get("incident", k, a)
        ds = self.dstyle()
        d = self.doc(p, f"Incident report {k}", "operations/incidents")
        if p.ext == ".eml":
            rep = g("reported_by").raw
            d.meta["email"] = {"from": f"{rep} <{slug(rep).replace('-', '.')}@quillmoor.example>", "to": "ops-incidents@quillmoor.example",
                               "subject": f"Incident report {k}", "date": "Mon, 03 Mar 2025 09:12:00 +0000",
                               "message_id": f"<{slug(k)}@quillmoor.example>"}
        d.blocks.append(Block("kv", pairs=[("Incident ID", k), (self.col(g("date")), self.put(d, g("date"), ds)),
                                           (self.col(g("site")), self.put(d, g("site"))), (self.col(g("asset")), self.put(d, g("asset"))),
                                           (self.col(g("severity")), self.put(d, g("severity"))),
                                           (self.col(g("reported_by")), self.put(d, g("reported_by")))]))
        d.blocks.append(Block("heading", "What happened", 2))
        d.blocks.append(Block("para", self.state(d, g("duration")) + " " + self.r.choice(N.FILLER["engineering"])))
        d.blocks.append(Block("heading", "Investigation", 2))
        d.blocks.append(Block("para", self.state(d, g("root_cause")) + " " + self.r.choice(N.FILLER["engineering"])))
        d.blocks.append(Block("para", self.state(d, g("cost"), self.mstyle())))
        d.blocks += self.filler("corporate", 1)
        return d

    def b_incident_log(self, p: Plan) -> Doc:
        w, r = self.w, self.r
        s = w.by_key[("site", p.payload["site"])]
        d = self.doc(p, f"alarm log {s.a['code']}", "operations/logs")
        lines: list[tuple[str, str]] = []
        for inc in [e for e in w.e["incident"] if e.a["site"] == s.key]:
            f = w.get("incident", inc.key, "alarm_code")
            date = w.get("incident", inc.key, "date").raw
            t = f"{date.isoformat()} {r.randint(0, 23):02d}:{r.randint(0, 59):02d}:{r.randint(0, 59):02d}"
            v = self.put(d, f)
            lines.append((t, f"ALARM {v} raised on {inc.a['asset']} -> {inc.key} opened"))
            lines.append((t[:11] + f"{r.randint(0, 23):02d}:{r.randint(0, 59):02d}:{r.randint(0, 59):02d}", f"ACK {v} by operator"))
        for _ in range(r.randint(25, 60)):
            t = f"2025-{r.randint(1, 12):02d}-{r.randint(1, 28):02d} {r.randint(0, 23):02d}:{r.randint(0, 59):02d}:{r.randint(0, 59):02d}"
            tag = r.choice(s.a["assets"])
            lines.append((t, r.choice([f"INFO {tag} start command accepted", f"INFO heartbeat ok from outstation {r.randint(1, 40)}",
                                       f"WARN {tag} vibration trend {r.randint(2, 9)}.{r.randint(0, 9)} mm/s",
                                       f"INFO {tag} stopped (duty rotation)", "WARN comms retry on link B",
                                       f"INFO historian flush {r.randint(200, 9000)} points"])))
        lines.sort()
        d.raw = "\n".join(f"{t} {m}" for t, m in lines) + "\n"
        return d

    def b_insurance_scan(self, p: Plan) -> Doc:
        w, k = self.w, p.payload["incident"]
        d = self.doc(p, "Seaboard Marine & Utility Insurers", "finance/insurance", scan=True)
        d.blocks += [Block("para", "Dear Sirs,"),
                     Block("para", f"We acknowledge your notification of the loss described as incident {k} at "
                                   f"{w.get('incident', k, 'site').raw}. {self.state(d, w.get('incident', k, 'insurer_ref'))} "
                                   "Please quote it in all correspondence."),
                     Block("para", "A loss adjuster will contact you within ten working days. Yours faithfully, Claims Department")]
        return d

    def b_supplier_list(self, p: Plan) -> Doc:
        w = self.w
        cols = ["Vendor no.", "Supplier", "Category", "Country", "Payment terms", "Supplier rating", "Contact"]
        d = self.table_doc(p, "Approved supplier list", "procurement", cols, [],
                           "Suppliers approved under the procurement standard. Ratings are from the last annual assessment.", "Suppliers")
        rows = []
        for k in p.payload["suppliers"]:
            g = lambda a, k=k: w.get("supplier", k, a)
            rows.append([self.put(d, g("vendor_no")), k, self.put(d, g("category")), self.put(d, g("country")),
                         self.put(d, g("payment_terms")), self.put(d, g("rating")), self.put(d, g("contact"))])
        self._fill_rows(d, rows)
        return d

    def b_contract_doc(self, p: Plan) -> Doc:
        w, k = self.w, p.payload["contract"]
        g = lambda a: w.get("contract", k, a)
        ds, ms = self.dstyle(), self.mstyle()
        d = self.doc(p, f"Agreement {k}", "legal/contracts", header=f"Contract {k} — Commercial in confidence", footer="Page")
        d.blocks.append(Block("para", self.state(d, g("counterparty"))))
        d.blocks.append(Block("heading", "1. Term", 2))
        d.blocks.append(Block("para", f"1.1 {self.state(d, g('start_date'), ds)} 1.2 "
                                      f"{self.state(d, g('end_date'), ds, old=bool(g('end_date').old))}"))
        d.blocks.append(Block("heading", "2. Price", 2))
        d.blocks.append(Block("para", f"2.1 {self.state(d, g('value'), ms, old=bool(g('value').old))} "
                                      "2.2 Invoices are payable in accordance with the payment terms in Schedule 2."))
        d.blocks.append(Block("heading", "3. Delay and termination", 2))
        d.blocks.append(Block("para", f"3.1 {self.state(d, g('penalty'))} 3.2 {self.state(d, g('notice_days'))}"))
        d.blocks += [Block("para", x.text) for x in self.filler("legal", 2)]
        d.blocks.append(Block("para", self.state(d, g("signed_by"))))
        return d

    def b_contract_amendment(self, p: Plan) -> Doc:
        w, k = self.w, p.payload["contract"]
        d = self.doc(p, f"Variation No. 1 to {k}", "legal/contracts")
        if p.ext == ".eml":
            d.meta["email"] = {"from": "contracts@quillmoor.example", "to": "finance@quillmoor.example",
                               "subject": f"Variation No. 1 to {k} — signed", "date": "Thu, 15 May 2025 14:02:00 +0000",
                               "message_id": f"<var-{slug(k)}@quillmoor.example>"}
        for attr, what in (("value", "the Contract Price"), ("end_date", "the expiry date of the term")):
            f = w.get("contract", k, attr)
            if f.old:
                st = self.mstyle() if attr == "value" else self.dstyle()
                old = self.put(d, f, st, old=True)
                new = self.put(d, f, st)
                d.blocks.append(Block("para", f"With effect from the date of this variation, {what} under contract {k} is amended from "
                                              f"{old} to {new}. All other terms remain unchanged."))
        d.blocks += [Block("para", x.text) for x in self.filler("legal", 1)]
        return d

    def b_customer_accounts(self, p: Plan) -> Doc:
        w = self.w
        cols = ["Account no.", "Customer", "Tariff", "Monthly volume (m³)", "Billing day", "Account manager"]
        d = self.table_doc(p, "Non-domestic customer accounts", "commercial/accounts", cols, [],
                           "Extract from the billing system for the account review meeting.", "Accounts")
        rows = []
        for k in p.payload["customers"]:
            g = lambda a, k=k: w.get("customer", k, a)
            rows.append([self.put(d, g("account_no")), k, self.put(d, g("tariff")), self.put(d, g("monthly_volume"), with_unit=False),
                         self.put(d, g("billing_day")), self.put(d, g("account_manager"))])
        self._fill_rows(d, rows)
        return d

    def b_parts_catalog(self, p: Plan) -> Doc:
        w = self.w
        cols = ["SKU", "Description", "Unit price (EUR)", "In stock", "Reorder pt", "Lead time (days)", "Bin", "Supplier"]
        d = self.table_doc(p, "Stores catalogue", "stores", cols, [], "Stock levels as at the last cycle count.", "Catalogue")
        rows = []
        for k in p.payload["parts"]:
            g = lambda a, k=k: w.get("part", k, a)
            rows.append([k, self.put(d, g("description")), self.put(d, g("unit_price"), "bare"), self.put(d, g("stock_qty")),
                         self.put(d, g("reorder_point")), self.put(d, g("lead_time"), with_unit=False), self.put(d, g("bin")),
                         self.put(d, g("supplier"))])
        self._fill_rows(d, rows)
        return d

    def b_policy_doc(self, p: Plan) -> Doc:
        w, k, ver = self.w, p.payload["policy"], p.payload["version"]
        pol = w.by_key[("policy", k)]
        g = lambda a: w.get("policy", k, a)
        current = ver == 2 or not pol.a["revised"]
        d = self.doc(p, k, "people/policies" if "Working" in k or "Leave" in k else "hse/standards",
                     header=f"{k} — version {ver}", footer="Quillmoor Utilities plc")
        ds = self.dstyle()
        meta = [("Document no.", self.put(d, g("number"))), ("Version", str(ver)), ("Owner", self.put(d, g("owner")))]
        if current:
            meta.insert(2, ("Effective", self.put(d, g("effective_date"), ds)))
        else:
            meta.insert(2, ("Status", "SUPERSEDED — see version 2"))
        d.blocks.append(Block("kv", pairs=meta))
        d.blocks.append(Block("heading", "1. Purpose", 2))
        d.blocks += self.filler("hr", 1)
        d.blocks.append(Block("heading", "2. Requirements", 2))
        for i, slug_ in enumerate(pol.a["rules"], 1):
            f = g(f"rule:{slug_}")
            old = bool(f.old) and not current
            d.blocks.append(Block("para", f"2.{i} {self.state(d, f, self.mstyle(), old=old)} " + self.r.choice(N.FILLER["hr"])))
        if ver == 2:
            d.blocks.append(Block("note", "This version replaces version 1. Changes are listed in the change notice."))
        return d

    def b_policy_notice(self, p: Plan) -> Doc:
        w, k = self.w, p.payload["policy"]
        pol = w.by_key[("policy", k)]
        d = self.doc(p, f"What has changed: {k}", "people/policies")
        if p.ext == ".eml":
            d.meta["email"] = {"from": "people-services@quillmoor.example", "to": "all-staff@quillmoor.example",
                               "subject": f"Policy update: {k}", "date": "Tue, 01 Apr 2025 08:00:00 +0000",
                               "message_id": f"<pol-{slug(k)}@quillmoor.example>"}
        f = w.get("policy", k, f"rule:{pol.a['rules'][0]}")
        ms = self.mstyle()
        old, new = self.put(d, f, ms, old=True), self.put(d, f, ms)
        d.blocks.append(Block("heading", "Summary of changes", 1))
        d.blocks.append(Block("bullets", items=[f"The {f.ctx['label']} changes from {old} to {new}.",
                                                "All other requirements are unchanged.", "Managers should brief their teams."]))
        return d

    def b_service_config(self, p: Plan) -> Doc:
        w, k = self.w, p.payload["service"]
        d = self.doc(p, f"{k} config", "it/services")
        port, to, ver = (w.get("service", k, a) for a in ("port", "timeout_s", "version"))
        pv, tv, vv = str(port.raw), str(to.raw), str(ver.raw)
        if p.ext == ".json":
            obj = {"service": k, "release": vv, "listen": {"host": "0.0.0.0", "port": int(pv)}, "requestTimeoutSeconds": int(tv),
                   "logLevel": "info"}
            d.raw = json.dumps(obj, indent=2)
            lines = (f'"port": {pv}', f'"requestTimeoutSeconds": {tv}', f'"release": "{vv}"')
        elif p.ext == ".xml":
            lines = (f'port="{pv}"', f"<requestTimeoutSeconds>{tv}</requestTimeoutSeconds>", f"<release>{vv}</release>")
            d.raw = (f'<?xml version="1.0" encoding="UTF-8"?>\n<service name="{k}">\n  {lines[2]}\n  <listen host="0.0.0.0" {lines[0]}/>\n'
                     f"  {lines[1]}\n  <logLevel>info</logLevel>\n</service>\n")
        elif p.ext in (".yaml", ".yml"):
            lines = (f"  port: {pv}", f"request_timeout_seconds: {tv}", f"release: \"{vv}\"")
            d.raw = f"# {k} service configuration\nservice: {k}\n{lines[2]}\nlisten:\n  host: 0.0.0.0\n{lines[0]}\n{lines[1]}\nlog_level: info\n"
        else:
            lines = (f"port = {pv}", f"request_timeout_seconds = {tv}", f"release = {vv}")
            d.raw = f"; {k} service configuration\n[service]\nname = {k}\n{lines[2]}\n\n[listen]\nhost = 0.0.0.0\n{lines[0]}\n\n[http]\n{lines[1]}\nlog_level = info\n"
        self.put_line(d, port, lines[0], pv)
        self.put_line(d, to, lines[1], tv)
        self.put_line(d, ver, lines[2], vv)
        return d

    def b_service_errors(self, p: Plan) -> Doc:
        w, k = self.w, p.payload["service"]
        svc = w.by_key[("service", k)]
        d = self.doc(p, f"{k} errors", "it/services/src")
        items = []
        for code in svc.a["errors"]:
            f = w.get("service", k, f"error:{code}")
            items.append((code, f))
        if p.ext == ".py":
            body = [f'"""Error codes raised by {k}. Keep in sync with the operator runbook."""', "", "ERRORS = {"]
            for code, f in items:
                line = f'    "{code}": "{f.raw}",'
                body.append(self.put_line(d, f, line, f.raw))
            body += ["}", "", "", "def describe(code: str) -> str:", '    return ERRORS.get(code, "unknown error")', ""]
        else:
            ts = p.ext == ".ts"
            body = [f"// Error codes raised by {k}. Keep in sync with the operator runbook.",
                    f"export const ERRORS{': Record<string, string>' if ts else ''} = {{"]
            for code, f in items:
                line = f'  "{code}": "{f.raw}",'
                body.append(self.put_line(d, f, line, f.raw))
            body += ["};", "", f"export function describe(code{': string' if ts else ''}){': string' if ts else ''} {{",
                     '  return ERRORS[code] ?? "unknown error";', "}", ""]
        d.raw = "\n".join(body)
        return d

    def b_service_board(self, p: Plan) -> Doc:
        w, k = self.w, p.payload["service"]
        d = self.doc(p, f"{k} board", "it/services/src")
        f = w.get("service", k, "refresh_s")
        line = f"const REFRESH_SECONDS = {f.raw};  // how often the {k} board reloads"
        self.put_line(d, f, line, str(f.raw))
        typ = "props: { url: string }" if p.ext == ".tsx" else "props"
        d.raw = (f"import React, {{ useEffect, useState }} from \"react\";\n\n{line}\n\n"
                 f"export default function {''.join(c for c in k if c.isalnum())}Board({typ}) {{\n  const [rows, setRows] = useState([]);\n"
                 "  useEffect(() => {\n    const t = setInterval(() => fetch(props.url).then(r => r.json()).then(setRows), REFRESH_SECONDS * 1000);\n"
                 "    return () => clearInterval(t);\n  }, [props.url]);\n"
                 f"  return <table className=\"board\">{{rows.map(r => <tr key={{r.id}}><td>{{r.name}}</td></tr>)}}</table>;\n}}\n")
        return d

    def b_service_css(self, p: Plan) -> Doc:
        w, k = self.w, p.payload["service"]
        d = self.doc(p, f"{k} theme", "it/services/src")
        own = w.get("service", k, "owner")
        line = f"   Maintainer: {own.raw} (service owner)"
        self.put_line(d, own, line, own.raw)
        d.raw = (f"/* {k} status board theme\n{line}\n*/\n:root {{\n  --qm-navy: #10233f;\n  --qm-teal: #1f8a8a;\n  --qm-alarm: #c0392b;\n}}\n"
                 ".board { font-family: system-ui, sans-serif; border-collapse: collapse; }\n.board td { padding: 4px 8px; }\n"
                 ".board tr.alarm td { background: var(--qm-alarm); color: #fff; }\n")
        return d

    def b_runbook(self, p: Plan) -> Doc:
        w, k = self.w, p.payload["service"]
        d = self.doc(p, f"{k} — operator runbook", "it/runbooks")
        d.blocks.append(Block("heading", "Ownership", 2))
        d.blocks.append(Block("para", self.state(d, w.get("service", k, "owner")) + " " + self.state(d, w.get("service", k, "sla"))))
        d.blocks.append(Block("heading", "First response", 2))
        d.blocks.append(Block("bullets", items=["Check the service health page.", "Look up the error code in the source module.",
                                                "Escalate to the service owner if the board shows red for more than 15 minutes."]))
        d.blocks += self.filler("engineering", 1)
        return d

    def b_project_update(self, p: Plan) -> Doc:
        w, k = self.w, p.payload["project"]
        g = lambda a: w.get("project", k, a)
        ds = self.dstyle()
        d = self.doc(p, f"{k} — monthly update", "projects")
        if p.ext == ".eml":
            d.meta["email"] = {"from": f"{g('pm').raw} <pmo@quillmoor.example>", "to": "programme-board@quillmoor.example",
                               "subject": f"{k} monthly update", "date": "Fri, 07 Mar 2025 17:30:00 +0000",
                               "message_id": f"<{slug(k)}-upd@quillmoor.example>"}
        d.blocks.append(Block("heading", "Headlines", 1))
        d.blocks.append(Block("bullets", items=[self.state(d, g("status")), self.state(d, g("go_live"), ds),
                                                self.state(d, g("budget"), self.mstyle())]))
        d.blocks.append(Block("heading", "Governance", 1))
        d.blocks.append(Block("para", f"{self.state(d, g('sponsor'))} {self.state(d, g('pm'))} {self.state(d, g('start_date'), ds)}"))
        d.blocks += self.filler("corporate", 1)
        return d

    def b_chemical_sds(self, p: Plan) -> Doc:
        w, k = self.w, p.payload["chemical"]
        g = lambda a: w.get("chemical", k, a)
        d = self.doc(p, f"Safety data sheet — {k}", "hse/chemicals")
        d.blocks.append(Block("heading", "Section 1: Identification", 2))
        d.blocks.append(Block("para", f"{k} is a {g('dosage').ctx['t']} used in water treatment. {self.state(d, g('supplier'))}"))
        d.blocks.append(Block("heading", "Section 7: Handling and storage", 2))
        d.blocks.append(Block("para", f"{self.state(d, g('storage_temp'))} {self.state(d, g('max_stock'))} Keep away from acids."))
        d.blocks.append(Block("heading", "Section 14: Transport", 2))
        d.blocks.append(Block("para", self.state(d, g("hazard_class"))))
        d.blocks.append(Block("heading", "Site dosing note", 2))
        d.blocks.append(Block("para", self.state(d, g("dosage")) + " Dose rates are trimmed by the process chemist on site."))
        return d

    def b_vessel_register(self, p: Plan) -> Doc:
        w = self.w
        cols = ["Vessel", "Call sign", "Length overall", "Crew", "Home site", "Built"]
        d = self.table_doc(p, "Marine fleet register", "marine", cols, [], "Workboats and survey vessels.", "Fleet")
        rows = []
        for k in p.payload["vessels"]:
            g = lambda a, k=k: w.get("vessel", k, a)
            rows.append([k, self.put(d, g("call_sign")), self.put(d, g("length")), self.put(d, g("crew")), self.put(d, g("home_site")),
                         self.put(d, g("built_year"))])
        self._fill_rows(d, rows)
        return d

    def b_vessel_scan(self, p: Plan) -> Doc:
        w, k = self.w, p.payload["vessel"]
        g = lambda a: w.get("vessel", k, a)
        d = self.doc(p, "Sarne Isles Harbour Authority", "marine", scan=True)
        d.blocks += [Block("para", "To: Marine Coordinator, Quillmoor Utilities plc"),
                     Block("para", f"Re: annual berth registration for {k}. We confirm the particulars held on our register: "
                                   f"{self.state(d, g('call_sign'))} {self.state(d, g('length'))} {self.state(d, g('crew'))} "
                                   f"{self.state(d, g('built_year'))} {self.state(d, g('home_site'))}"),
                     Block("para", "Please notify the Harbour Office of any change within 14 days. Harbour Master")]
        return d

    def b_distractor(self, p: Plan) -> Doc:
        r = self.r
        topic = N.NEWSLETTER_TOPICS[p.payload["n"] % len(N.NEWSLETTER_TOPICS)]
        site = r.choice(self.w.e["site"]).key
        d = self.doc(p, f"Staff news: {topic}", "comms/newsletter")
        if p.ext == ".eml":
            d.meta["email"] = {"from": "internal-comms@quillmoor.example", "to": "all-staff@quillmoor.example", "subject": f"Staff news: {topic}",
                               "date": "Wed, 12 Feb 2025 12:00:00 +0000", "message_id": f"<news-{p.payload['n']}@quillmoor.example>"}
        d.blocks += [Block("heading", "News from around the islands", 1),
                     Block("para", f"This month we celebrate {topic}. Colleagues at {site} raised {r.randint(120, 2400)} EUR for the lifeboat "
                                   f"charity, and {r.randint(12, 90)} people signed up in the first week."),
                     Block("bullets", items=[f"Canteen opening hours: {r.randint(7, 9)}:00 to {r.randint(14, 16)}:30.",
                                             f"The next town hall is on {r.randint(2, 27)} {r.choice(['May', 'June', 'October', 'November'])}.",
                                             "Please keep the car park clear for the delivery lorries."])]
        if p.ext in (".xlsx",):
            d.blocks.append(Block("table", columns=["Team", "Points"], rows=[[f"Team {c}", str(r.randint(10, 99))] for c in "ABCDE"], caption="Quiz"))
        return d
