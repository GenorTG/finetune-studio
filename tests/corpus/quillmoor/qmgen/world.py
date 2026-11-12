"""The Quillmoor world: seeded entities and the facts the corpus states about them.

Quillmoor Utilities plc runs tidal stations, water works, pumping stations, desalination plants, substations and
reservoirs across the (invented) Sarne Isles. Every entity attribute is a Fact. A Fact knows its typed value and can
write it in several styles (dates as ISO / 03.04.2019 / 3 April 2019, money with or without grouping...); the document
builder records the exact string it wrote. Some facts were changed later (a contract amendment, a policy v2, a new
desk extension): ``old`` holds the outdated values, which older documents still state.
"""
from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass, field
from typing import Any

from . import names as N

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December"]

TIERS = {  # entity counts per tier; documents follow from these
    "s": {"sites": 12, "people_per_site": 10, "assets_per_site": 12, "suppliers": 26, "contracts": 40,
          "customers": 30, "incidents": 50, "parts": 120, "policies": 12, "services": 14, "projects": 14,
          "chemicals": 12, "vessels": 8},
    "m": {"sites": 40, "people_per_site": 14, "assets_per_site": 22, "suppliers": 90, "contracts": 220,
          "customers": 180, "incidents": 360, "parts": 900, "policies": 36, "services": 48, "projects": 70,
          "chemicals": 36, "vessels": 30},
    "l": {"sites": 160, "people_per_site": 18, "assets_per_site": 34, "suppliers": 420, "contracts": 1400,
          "customers": 1300, "incidents": 2600, "parts": 6500, "policies": 110, "services": 180, "projects": 380,
          "chemicals": 120, "vessels": 110},
}


# ── values ────────────────────────────────────────────────────────────────────────────────────────────────────────
def fmt_int(n: int, style: str = "group") -> str:
    return f"{n:,}" if style == "group" else str(n)


def fmt_money(x: float, style: str = "suffix", decimals: int | None = None) -> str:
    d = decimals if decimals is not None else (0 if float(x).is_integer() else 2)
    num = f"{x:,.{d}f}"
    return {"suffix": f"{num} EUR", "prefix": f"EUR {num}", "symbol": f"€{num}", "bare": num}[style]


def fmt_date(d: dt.date, style: str = "iso") -> str:
    if style == "iso":
        return d.isoformat()
    if style == "dmy":
        return d.strftime("%d.%m.%Y")
    if style == "long":
        return f"{d.day} {MONTHS[d.month - 1]} {d.year}"
    raise ValueError(style)


@dataclass
class Fact:
    fid: str
    etype: str
    ekey: str                       # how questions name the entity (tag, name, number)
    attr: str
    kind: str                       # int | dec | money | date | text | year
    raw: Any
    unit: str = ""                  # appended in prose / answers ("h", "kW", "days")
    old: list[Any] = field(default_factory=list)   # outdated raw values, newest last
    ctx: dict[str, str] = field(default_factory=dict)  # extra template fields ({t} type word, {s} site, ...)
    stated_in: list[str] = field(default_factory=list)   # doc ids stating the current value
    old_in: list[str] = field(default_factory=list)      # doc ids stating an outdated value

    def write(self, style: str = "", raw: Any = None, with_unit: bool = True) -> str:
        """The value as a document writes it (``style`` per kind; ``raw`` to write an outdated value)."""
        v = self.raw if raw is None else raw
        if self.kind == "int":
            s = fmt_int(v, style or "group")
        elif self.kind == "dec":
            s = v
        elif self.kind == "money":
            return fmt_money(v, style or "suffix")
        elif self.kind == "date":
            return fmt_date(v, style or "iso")
        else:
            return str(v)
        return f"{s} {self.unit}".rstrip() if (with_unit and self.unit) else s

    def canonical(self, raw: Any = None) -> str:
        return self.write("", raw)

    def accepted(self, raw: Any = None) -> list[str]:
        """Every rendering a judge should accept as this value."""
        v = self.raw if raw is None else raw
        if self.kind == "money":
            return [fmt_money(v, s) for s in ("suffix", "prefix", "symbol")]
        if self.kind == "date":
            return [fmt_date(v, s) for s in ("iso", "long", "dmy")]
        if self.kind == "int":
            return list(dict.fromkeys([self.write("group", v), self.write("plain", v)]))
        return [self.write("", v)]

    @property
    def superseded(self) -> bool:
        return bool(self.old)


@dataclass
class Entity:
    etype: str
    key: str
    a: dict[str, Any] = field(default_factory=dict)   # raw attributes (incl. links to other entities' keys)


class World:
    def __init__(self, tier: str = "s", seed: int = 20261029) -> None:
        self.tier, self.seed = tier, seed
        self.rng = random.Random(f"quillmoor-{seed}-{tier}")
        self.n = TIERS[tier]
        self.e: dict[str, list[Entity]] = {}
        self.by_key: dict[tuple[str, str], Entity] = {}
        self.facts: dict[str, Fact] = {}
        self._used: set[str] = set()
        self._build()

    # ── helpers ──
    def _uniq(self, make, tries: int = 400) -> str:
        for _ in range(tries):
            v = make()
            if v not in self._used:
                self._used.add(v)
                return v
        raise RuntimeError("name space exhausted")

    def _date(self, y0: int, y1: int) -> dt.date:
        start = dt.date(y0, 1, 1).toordinal()
        return dt.date.fromordinal(self.rng.randint(start, dt.date(y1, 12, 28).toordinal()))

    def _add(self, ent: Entity) -> Entity:
        self.e.setdefault(ent.etype, []).append(ent)
        self.by_key[(ent.etype, ent.key)] = ent
        return ent

    def fact(self, etype: str, ekey: str, attr: str, kind: str, raw: Any, unit: str = "", **ctx: str) -> Fact:
        fid = f"{etype}:{ekey}:{attr}"
        f = Fact(fid, etype, ekey, attr, kind, raw, unit, ctx=dict(ctx))
        self.facts[fid] = f
        return f

    def get(self, etype: str, ekey: str, attr: str) -> Fact:
        return self.facts[f"{etype}:{ekey}:{attr}"]

    def person_name(self) -> str:
        return self._uniq(lambda: f"{self.rng.choice(N.FIRST)} {N.surname(self.rng)}")

    # ── the world ──
    def _build(self) -> None:
        r, n = self.rng, self.n
        islands = N.ISLANDS[: max(3, min(len(N.ISLANDS), n["sites"] // 3))]
        # suppliers first (asset models and parts reference them)
        for _ in range(n["suppliers"]):
            cat = r.choice(N.SUPPLIER_CATEGORIES)
            name = self._uniq(lambda: f"{N.surname(r)} {r.choice(N.SUPPLIER_WORDS)} {r.choice(N.SUFFIXES)}")
            s = self._add(Entity("supplier", name, {"category": cat}))
            self.fact("supplier", name, "vendor_no", "text", self._uniq(lambda: f"V-{r.randint(10000, 99999)}"))
            self.fact("supplier", name, "country", "text", r.choice(N.COUNTRIES))
            self.fact("supplier", name, "payment_terms", "text", f"net {r.choice([14, 21, 28, 30, 35, 42, 45, 47, 60, 75])} days")
            self.fact("supplier", name, "rating", "text", f"{r.randint(21, 49) / 10:.1f}/5")
            self.fact("supplier", name, "contact", "text", self.person_name())
            self.fact("supplier", name, "category", "text", cat)
            s.a["brand"] = name.split()[0]
        by_cat: dict[str, list[Entity]] = {}
        for s in self.e["supplier"]:
            by_cat.setdefault(s.a["category"], []).append(s)
        self.by_cat = by_cat

        def supplier_for(cat: str) -> Entity:
            return r.choice(by_cat.get(cat) or self.e["supplier"])

        # sites + people + assets
        kinds_cycle = [k for k in N.SITE_KINDS]
        tag_numbers: list[int] = []
        for i in range(n["sites"]):
            kind, unit, (lo, hi), mix = kinds_cycle[i % len(kinds_cycle)]
            isl = islands[i % len(islands)]
            place = self._uniq(lambda: N.place(r))
            sname = f"{place} {kind}"
            code = self._uniq(lambda isl=isl, kind=kind: f"{isl[:3].upper()}-{''.join(w[0] for w in kind.split())}{r.randint(1, 99):02d}")
            site = self._add(Entity("site", sname, {"kind": kind, "island": isl, "code": code, "mix": mix}))
            cap = round(r.uniform(lo, hi), 1) if unit in ("MW", "MVA") else int(r.uniform(lo, hi) // 10 * 10)
            self.fact("site", sname, "code", "text", code)
            self.fact("site", sname, "island", "text", isl)
            self.fact("site", sname, "commissioned", "year", str(r.randint(1979, 2022)))
            self.fact("site", sname, "capacity", "dec" if unit in ("MW", "MVA") else "int",
                      f"{cap}" if unit in ("MW", "MVA") else cap, unit)
            self.fact("site", sname, "permit", "text", self._uniq(lambda: f"PRM-{r.randint(1000, 9999)}-{r.randint(10, 99)}"))
            self.fact("site", sname, "muster_point", "text", r.choice([
                "the east gatehouse car park, bay 3", "the old boathouse slipway", "the grass bank behind the chemical store",
                "the visitor car park by the flagpole", "the turning circle at the north gate", "the lay-by opposite the transformer pen",
                "the helipad marker on the upper terrace", "the bus shelter at the site entrance"]))
            self.fact("site", sname, "telemetry_poll_s", "int", r.choice([5, 10, 12, 15, 20, 30, 45, 60, 90]), "s")
            self.fact("site", sname, "retention_days", "int", r.choice([30, 45, 60, 90, 120, 180, 365, 400, 730]), "days")
            staff = []
            for j in range(n["people_per_site"]):
                role = "Site Manager" if j == 0 else r.choice(N.ROLES[1:])
                p = self._add(Entity("person", self.person_name(), {"site": sname, "role": role}))
                staff.append(p)
                self.fact("person", p.key, "employee_id", "text", self._uniq(lambda: f"E{r.randint(10000, 99999)}"))
                self.fact("person", p.key, "role", "text", role)
                self.fact("person", p.key, "site", "text", sname)
                self.fact("person", p.key, "start_date", "date", self._date(2003, 2025))
                ext = self.fact("person", p.key, "extension", "text", self._uniq(lambda: str(r.randint(2000, 7999))))
                if r.random() < 0.08:  # moved desk: the directory still lists the old number, the chat has the new one
                    ext.old = [ext.raw]
                    ext.raw = self._uniq(lambda: str(r.randint(2000, 7999)))
                if r.random() < 0.55:
                    self.fact("person", p.key, "certification", "text", r.choice(N.CERTS))
                    self.fact("person", p.key, "cert_expiry", "date", self._date(2026, 2029))
            site.a["staff"] = [p.key for p in staff]
            self.fact("site", sname, "manager", "text", staff[0].key)
            self.fact("site", sname, "headcount", "int", r.randint(len(staff) + 3, len(staff) * 4 + 9))
            assets = []
            for _ in range(n["assets_per_site"]):
                t = r.choice(mix)
                tname, runit, (alo, ahi), cat = N.ASSET_TYPES[t]
                if tag_numbers and r.random() < 0.35:   # near-duplicate tags (same number, other letter) on purpose
                    num = r.choice(tag_numbers)
                else:
                    num = r.randint(1000, 9999)
                    tag_numbers.append(num)
                tag = self._uniq(lambda t=t, num=num: f"{t}-{num}-{r.choice('ABCDEFGHJKLMNPRSTUVWXZ')}")
                sup = supplier_for(cat)
                a = self._add(Entity("asset", tag, {"type": t, "site": sname, "supplier": sup.key}))
                assets.append(a)
                ctx = {"t": tname, "s": sname}
                self.fact("asset", tag, "model", "text", f"{sup.a['brand']} {r.choice(['HX', 'Q', 'TL', 'M', 'RX', 'V', 'Nova'])}-{r.randint(20, 990)}{r.choice(['', 'e', 'S', 'X'])}", **ctx)
                self.fact("asset", tag, "serial", "text", self._uniq(lambda: f"SN{r.randint(10**7, 10**8 - 1)}"), **ctx)
                self.fact("asset", tag, "install_date", "date", self._date(1998, 2024), **ctx)
                rv = round(r.uniform(alo, ahi)) if runit != "bar" else r.choice([6, 10, 16, 25, 40])
                self.fact("asset", tag, "rating", "int", rv, runit, **ctx)
                self.fact("asset", tag, "service_interval", "int", r.choice(range(1000, 16001, 250)), "operating hours", **ctx)
                self.fact("asset", tag, "last_overhaul", "date", self._date(2019, 2025), **ctx)
                self.fact("asset", tag, "site", "text", sname, **ctx)
                self.fact("asset", tag, "criticality", "text", r.choice("ABC"), **ctx)
                if r.random() < 0.18:
                    self.fact("asset", tag, "mass_kg", "int", r.randint(40, 24000), "kg", **ctx)
            site.a["assets"] = [a.key for a in assets]

        people = self.e["person"]

        def pick_person(roles: tuple[str, ...] = ()) -> str:
            pool = [p for p in people if p.a["role"] in roles] if roles else people
            return r.choice(pool or people).key

        # customers
        for _ in range(n["customers"]):
            name = self._uniq(lambda: f"{N.place(r)} {r.choice(N.BUSINESSES)}")
            self._add(Entity("customer", name))
            self.fact("customer", name, "account_no", "text", self._uniq(lambda: f"AC-{r.randint(10**6, 10**7 - 1)}"))
            self.fact("customer", name, "tariff", "text", f"QT-{r.choice('ABCDEFGHK')}{r.randint(1, 9)}")
            self.fact("customer", name, "monthly_volume", "int", r.randint(120, 96000), "m³")
            self.fact("customer", name, "billing_day", "int", r.randint(1, 28))
            self.fact("customer", name, "account_manager", "text", pick_person(("Customer Account Manager", "Commercial Lead", "Billing Analyst")))

        # contracts (some amended later)
        counterparties = [("supplier", s.key) for s in self.e["supplier"]] + [("customer", c.key) for c in self.e["customer"]]
        for _ in range(n["contracts"]):
            yr = r.randint(2016, 2025)
            num = self._uniq(lambda yr=yr: f"QC-{yr}-{r.randint(1, 999):03d}")
            ctype, cp = r.choice(counterparties)
            self._add(Entity("contract", num, {"cp_type": ctype}))
            start = dt.date(yr, r.randint(1, 12), r.randint(1, 28))
            end = dt.date(yr + r.randint(2, 7), start.month, start.day)
            self.fact("contract", num, "counterparty", "text", cp)
            val = self.fact("contract", num, "value", "money", r.randint(18, 4800) * 250)
            self.fact("contract", num, "start_date", "date", start)
            endf = self.fact("contract", num, "end_date", "date", end)
            self.fact("contract", num, "penalty", "text", f"{r.choice([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5])}% per week of delay")
            self.fact("contract", num, "notice_days", "int", r.choice([30, 45, 60, 90, 120, 180]), "days")
            self.fact("contract", num, "signed_by", "text", pick_person(("Contracts Manager", "Legal Counsel", "Commercial Lead", "Procurement Officer")))
            roll = r.random()
            if roll < 0.15:
                val.old = [val.raw]
                val.raw = val.raw + r.choice([-1, 1]) * r.randint(4, 120) * 250
            elif roll < 0.25:
                endf.old = [endf.raw]
                endf.raw = endf.raw + dt.timedelta(days=r.choice([182, 365, 547, 730]))

        # incidents
        sites = self.e["site"]
        for _ in range(n["incidents"]):
            yr = r.randint(2019, 2025)
            iid = self._uniq(lambda yr=yr: f"INC-{yr}-{r.randint(1, 9999):04d}")
            site = r.choice(sites)
            tag = r.choice(site.a["assets"])
            self._add(Entity("incident", iid, {"site": site.key, "asset": tag}))
            self.fact("incident", iid, "date", "date", dt.date(yr, r.randint(1, 12), r.randint(1, 28)))
            self.fact("incident", iid, "site", "text", site.key)
            self.fact("incident", iid, "asset", "text", tag)
            self.fact("incident", iid, "duration", "dec", f"{r.randint(2, 365) / 10:.1f}", "hours")
            self.fact("incident", iid, "root_cause", "text", r.choice(N.ROOT_CAUSES))
            self.fact("incident", iid, "cost", "money", r.randint(3, 900) * 125)
            self.fact("incident", iid, "reported_by", "text", r.choice(site.a["staff"]))
            self.fact("incident", iid, "severity", "text", f"S{r.choice([1, 2, 2, 3, 3, 3, 4, 4])}")
            self.fact("incident", iid, "alarm_code", "text", self._uniq(lambda: f"ALM-{r.randint(1000, 9999)}"))
            if r.random() < 0.12:
                self.fact("incident", iid, "insurer_ref", "text", self._uniq(lambda: f"CLM/{r.randint(10, 99)}/{r.randint(10000, 99999)}"))

        # parts
        sku_bases: list[int] = []
        for _ in range(n["parts"]):
            if sku_bases and r.random() < 0.3:
                base = r.choice(sku_bases)
            else:
                base = r.randint(10000, 99999)
                sku_bases.append(base)
            sku = self._uniq(lambda base=base: f"QP-{base}{r.randint(0, 9)}")
            sup = r.choice(self.e["supplier"])
            self._add(Entity("part", sku, {"supplier": sup.key}))
            desc = self._uniq(lambda: f"{r.choice(['impeller', 'mechanical seal kit', 'bearing', 'gasket set', 'pressure transmitter', 'contactor', 'membrane element', 'UV lamp', 'filter cartridge', 'actuator gearbox', 'coupling insert', 'cable gland', 'relay module', 'diaphragm', 'shaft sleeve', 'O-ring kit', 'level probe', 'fuse link', 'non-return valve', 'wear ring'])}, {r.choice(['bronze', 'stainless 316', 'duplex', 'EPDM', 'Viton', 'cast iron', 'PTFE', 'nylon', 'ceramic'])}, {r.choice([12, 16, 25, 32, 40, 50, 63, 80, 100, 125, 150, 200, 250, 280, 315, 400])} mm" + r.choice(['', '', ', PN10', ', PN16', ', PN25', ', ATEX', ', marine grade', ', long life', ', OEM', ', heavy duty', ', low temperature', ', food grade']), tries=4000)
            self.fact("part", sku, "description", "text", desc)
            self.fact("part", sku, "unit_price", "money", round(r.uniform(3.5, 9800), 2))
            self.fact("part", sku, "stock_qty", "int", r.randint(0, 640))
            self.fact("part", sku, "reorder_point", "int", r.randint(1, 60))
            self.fact("part", sku, "lead_time", "int", r.randint(2, 140), "days")
            self.fact("part", sku, "bin", "text", f"R{r.randint(1, 24):02d}-S{r.randint(1, 8)}-B{r.randint(1, 40):02d}")
            self.fact("part", sku, "supplier", "text", sup.key)

        # policies (rules; some revised in a v2)
        templates = POLICY_TEMPLATES
        for i in range(n["policies"]):
            title, rules = templates[i % len(templates)]
            if i >= len(templates):
                title = f"{title} ({islands[(i // len(templates) - 1) % len(islands)]} region)"
            num = self._uniq(lambda: f"POL-{r.randint(100, 999)}")
            pol = self._add(Entity("policy", title, {"number": num, "rules": []}))
            self.fact("policy", title, "number", "text", num)
            self.fact("policy", title, "effective_date", "date", self._date(2020, 2025))
            self.fact("policy", title, "owner", "text", pick_person(("HR Advisor", "HSE Advisor", "Legal Counsel", "Quality Auditor", "Duty Manager")))
            revised = r.random() < 0.45
            for k, (slug, label, kind, unit, gen, para) in enumerate(rules):
                f = self.fact("policy", title, f"rule:{slug}", kind, gen(r), unit, label=label, para=para)
                pol.a["rules"].append(slug)
                if revised and k == 0:
                    newv = gen(r)
                    while newv == f.raw:
                        newv = gen(r)
                    f.old, f.raw = [f.raw], newv
            pol.a["revised"] = revised

        # services
        for _ in range(n["services"]):
            name = self._uniq(lambda: f"{r.choice(N.SERVICE_A)}{r.choice(N.SERVICE_B)}")
            self._add(Entity("service", name, {"errors": []}))
            self.fact("service", name, "version", "text", f"{r.randint(2, 9)}.{r.randint(0, 14)}.{r.randint(0, 30)}")
            self.fact("service", name, "port", "text", self._uniq(lambda: str(r.randint(10000, 39999))))
            self.fact("service", name, "owner", "text", pick_person(("SCADA Engineer", "Network Engineer", "IT Support Analyst", "Data Analyst")))
            self.fact("service", name, "sla", "text", r.choice(["99.5%", "99.8%", "99.9%", "99.95%", "99.0%", "98.5%"]))
            self.fact("service", name, "timeout_s", "int", r.choice([5, 8, 10, 15, 20, 25, 30, 45, 60, 90, 120]), "s")
            self.fact("service", name, "refresh_s", "int", r.choice([3, 5, 7, 10, 12, 15, 20, 30, 45, 60]), "s")
            pre = "".join(c for c in name if c.isupper())[:2] or "QX"
            for _ in range(r.randint(3, 5)):
                code = self._uniq(lambda pre=pre: f"{pre}-{r.randint(100, 999)}")
                self.by_key[("service", name)].a["errors"].append(code)
                self.fact("service", name, f"error:{code}", "text", r.choice(N.ERROR_MEANINGS), code=code)

        # projects
        for _ in range(n["projects"]):
            name = self._uniq(lambda: f"Project {r.choice(N.PROJECT_WORDS)}" + (f" {r.choice(['II', 'III', 'IV', 'North', 'South', 'East', 'West', 'Phase 2', 'Phase 3', 'Upgrade', 'Renewal'])}" if r.random() < 0.5 else ""), tries=4000)
            self._add(Entity("project", name))
            start = self._date(2021, 2025)
            self.fact("project", name, "budget", "money", r.randint(40, 9600) * 1000)
            self.fact("project", name, "sponsor", "text", pick_person(("Site Manager", "Duty Manager", "Commercial Lead", "Finance Business Partner")))
            self.fact("project", name, "pm", "text", pick_person(("Project Manager", "Planning Engineer", "Civil Engineer", "Commissioning Engineer")))
            self.fact("project", name, "start_date", "date", start)
            self.fact("project", name, "go_live", "date", start + dt.timedelta(days=r.randint(120, 1100)))
            self.fact("project", name, "status", "text", r.choice(N.PROJECT_STATUSES))

        # chemicals
        for _ in range(n["chemicals"]):
            name = self._uniq(lambda: f"{r.choice(N.CHEM_A)}{r.choice(N.CHEM_B)}-{r.randint(2, 95)}")
            kind = r.choice(N.CHEM_KINDS)
            self._add(Entity("chemical", name, {"kind": kind}))
            lo = r.randint(0, 12)
            self.fact("chemical", name, "dosage", "dec", f"{r.randint(2, 480) / 10:.1f}", "mg/L", t=kind)
            self.fact("chemical", name, "storage_temp", "text", f"{lo} to {lo + r.randint(12, 30)} °C", t=kind)
            self.fact("chemical", name, "hazard_class", "text", r.choice(N.HAZARD_CLASSES), t=kind)
            self.fact("chemical", name, "max_stock", "int", r.randint(2, 900) * 25, "kg", t=kind)
            self.fact("chemical", name, "supplier", "text", supplier_for("chemicals").key, t=kind)

        # vessels
        for _ in range(n["vessels"]):
            name = self._uniq(lambda: f"MV {r.choice(N.VESSEL_A)} {r.choice(N.VESSEL_B)}" + (f" {r.randint(2, 9)}" if r.random() < 0.3 else ""))
            self._add(Entity("vessel", name))
            self.fact("vessel", name, "call_sign", "text", self._uniq(lambda: "MQ" + "".join(r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(4))))
            self.fact("vessel", name, "length", "dec", f"{r.randint(85, 640) / 10:.1f}", "m")
            self.fact("vessel", name, "crew", "int", r.randint(2, 14))
            self.fact("vessel", name, "home_site", "text", r.choice(sites).key)
            self.fact("vessel", name, "built_year", "year", str(r.randint(1984, 2023)))

    # ── links used by multi-hop questions ──
    def asset(self, tag: str) -> Entity:
        return self.by_key[("asset", tag)]


def _money(lo: float, hi: float, step: float = 0.05):
    return lambda r: round(round(r.uniform(lo, hi) / step) * step, 2)


def _ints(*choices: int):
    return lambda r: r.choice(choices)


def _dec(lo: int, hi: int, div: int = 10, fmt: str = "{:.1f}"):
    return lambda r: fmt.format(r.randint(lo, hi) / div)


# (slug, label in documents, kind, unit, generator, paraphrased question with {p} = policy title)
POLICY_TEMPLATES: list[tuple[str, list[tuple]]] = [
    ("Night Shift Working Policy", [
        ("max_nights", "maximum number of consecutive night shifts", "int", "", _ints(3, 4, 5, 6),
         "How many night shifts in a row may an employee work under the {p}?"),
        ("night_meal", "night shift meal allowance", "money", "", _money(9.5, 21.75),
         "How much does the {p} pay towards a meal on a night shift?"),
        ("min_rest", "minimum rest between two shifts", "dec", "hours", _dec(105, 135),
         "What is the shortest break the {p} allows between the end of one shift and the start of the next?")]),
    ("Lone Working Procedure", [
        ("checkin", "check-in interval for lone workers", "int", "minutes", _ints(20, 30, 40, 45, 60, 75, 90),
         "How often must someone working alone report in, according to the {p}?"),
        ("max_lone", "maximum continuous lone working time", "dec", "hours", _dec(20, 80),
         "For how long at most may a person work alone without a break under the {p}?"),
        ("escalate", "time after a missed check-in before escalation", "int", "minutes", _ints(5, 10, 12, 15, 20),
         "If a lone worker misses a call, how long does the {p} wait before raising the alarm?")]),
    ("Business Travel and Expenses Policy", [
        ("per_diem", "daily subsistence allowance", "money", "", _money(28, 74),
         "What daily amount does the {p} allow for subsistence when travelling?"),
        ("mileage", "private car mileage rate", "text", "", lambda r: f"{r.randint(28, 61) / 100:.2f} EUR per km",
         "How much is reimbursed per kilometre for using your own car under the {p}?"),
        ("hotel_cap", "hotel cap per night outside the Sarne Isles", "money", "", _money(95, 260, 5),
         "What is the most the {p} will pay for one night in a hotel off the islands?")]),
    ("Confined Space Entry Standard", [
        ("o2_min", "minimum oxygen level for entry", "text", "", lambda r: f"{r.choice([19.5, 19.8, 20.0, 20.2])}% by volume",
         "Below what oxygen concentration is entry forbidden under the {p}?"),
        ("h2s_max", "hydrogen sulphide alarm level", "text", "", lambda r: f"{r.choice([3, 5, 7, 10])} ppm",
         "At what H2S reading does the {p} require people to leave?"),
        ("standby", "minimum number of standby persons", "int", "", _ints(1, 2, 3),
         "How many people must stand by outside during an entry under the {p}?")]),
    ("Overtime and On-Call Policy", [
        ("callout", "call-out payment per attendance", "money", "", _money(38, 140, 0.5),
         "What is paid each time someone on call is called out, under the {p}?"),
        ("oncall_week", "weekly on-call standby payment", "money", "", _money(90, 340, 5),
         "How much does a week of standby earn under the {p}?"),
        ("max_ot", "maximum overtime per month", "int", "hours", _ints(24, 30, 32, 36, 40, 48),
         "What monthly overtime ceiling does the {p} set?")]),
    ("IT Acceptable Use Policy", [
        ("pw_len", "minimum password length", "int", "characters", _ints(12, 14, 15, 16, 18, 20),
         "How long must a password be at minimum under the {p}?"),
        ("lock", "screen lock timeout", "int", "minutes", _ints(3, 5, 7, 10, 12, 15),
         "After how much idle time must a screen lock under the {p}?"),
        ("usb", "number of approved USB storage devices per employee", "int", "", _ints(0, 1, 2),
         "How many approved USB sticks may one employee hold according to the {p}?")]),
    ("Chemical Delivery Procedure", [
        ("notice_h", "advance notice required for a bulk delivery", "int", "hours", _ints(12, 24, 36, 48, 72),
         "How much warning does the {p} require before a tanker delivery?"),
        ("max_wind", "maximum wind speed for offloading", "int", "knots", _ints(25, 28, 30, 32, 35, 40),
         "Above what wind speed must offloading stop under the {p}?"),
        ("witness", "minimum number of witnesses at offloading", "int", "", _ints(1, 2, 3),
         "How many people must witness an offload according to the {p}?")]),
    ("Vehicle and Fleet Policy", [
        ("speed_site", "speed limit on operational sites", "int", "km/h", _ints(10, 15, 16, 20, 25),
         "How fast may vehicles drive inside an operational site under the {p}?"),
        ("service_km", "vehicle service interval", "int", "km", _ints(12000, 15000, 18000, 20000, 25000, 30000),
         "After how many kilometres is a fleet vehicle serviced according to the {p}?"),
        ("pool_book", "maximum pool car booking length", "int", "days", _ints(2, 3, 4, 5, 7, 10),
         "For how many days at most can a pool car be booked under the {p}?")]),
    ("Remote and Hybrid Working Policy", [
        ("office_days", "minimum office days per week", "int", "", _ints(1, 2, 3),
         "How many days a week must staff be in the office under the {p}?"),
        ("home_allow", "monthly home working allowance", "money", "", _money(12, 55, 0.5),
         "How much is the monthly allowance for working from home under the {p}?"),
        ("abroad", "maximum days working abroad per year", "int", "days", _ints(10, 14, 20, 25, 30, 40),
         "How many days a year can someone work from abroad under the {p}?")]),
    ("Permit to Work Procedure", [
        ("validity", "maximum validity of a permit", "int", "hours", _ints(8, 10, 12, 14, 24),
         "For how long does a work permit stay valid at most under the {p}?"),
        ("handover", "handover meeting length at shift change", "int", "minutes", _ints(10, 15, 20, 30),
         "How long is the shift change handover meeting under the {p}?"),
        ("audit", "share of permits audited each month", "text", "", lambda r: f"{r.choice([5, 8, 10, 12, 15, 20])}%",
         "What fraction of permits gets audited every month according to the {p}?")]),
    ("Gifts and Hospitality Policy", [
        ("gift_cap", "gift value that must be declared", "money", "", _money(20, 120, 5),
         "Above what value must a gift be declared under the {p}?"),
        ("meal_cap", "hospitality meal limit per person", "money", "", _money(35, 150, 5),
         "What is the per-person limit for a business meal under the {p}?"),
        ("register", "days allowed to record a gift in the register", "int", "days", _ints(3, 5, 7, 10, 14),
         "Within how many days must a gift go into the register under the {p}?")]),
    ("Working at Height Standard", [
        ("harness", "height above which a harness is mandatory", "dec", "m", _dec(15, 30),
         "From what height up does the {p} require a harness?"),
        ("ladder", "maximum ladder working time", "int", "minutes", _ints(15, 20, 30, 45),
         "How long may someone work from a ladder at a time under the {p}?"),
        ("inspect", "harness inspection interval", "int", "months", _ints(3, 6, 12),
         "How often does the {p} require harnesses to be inspected?")]),
    ("Sick Leave and Absence Policy", [
        ("self_cert", "self-certified sick days", "int", "days", _ints(2, 3, 4, 5, 7),
         "For how many days can staff sign themselves off sick under the {p}?"),
        ("call_by", "time by which absence must be reported", "text", "", lambda r: f"{r.choice(['07:30', '08:00', '08:15', '09:00', '09:30'])}",
         "By what time of day must an absence be phoned in under the {p}?"),
        ("rtw", "absences triggering a return-to-work review", "int", "", _ints(2, 3, 4, 5),
         "After how many separate absences does the {p} call for a return-to-work review?")]),
    ("Data Retention Schedule", [
        ("cctv", "CCTV footage retention", "int", "days", _ints(21, 28, 30, 31, 45, 60),
         "How long is CCTV recording kept under the {p}?"),
        ("payroll", "payroll records retention", "int", "years", _ints(6, 7, 8, 10),
         "For how many years does the {p} keep payroll records?"),
        ("visitor", "visitor log retention", "int", "months", _ints(6, 12, 18, 24),
         "How long are visitor sign-in logs kept according to the {p}?")]),
    ("Hot Work Procedure", [
        ("firewatch", "fire watch after hot work", "int", "minutes", _ints(30, 45, 60, 90, 120),
         "How long must a fire watch continue once hot work ends under the {p}?"),
        ("clear", "clearance radius for combustibles", "dec", "m", _dec(50, 150),
         "How far must combustible material be cleared from hot work under the {p}?"),
        ("extinguishers", "minimum extinguishers at the work point", "int", "", _ints(1, 2, 3),
         "How many extinguishers must be at hand during hot work according to the {p}?")]),
    ("Learning and Development Policy", [
        ("budget", "annual training budget per employee", "money", "", _money(300, 2400, 50),
         "How much training money does each employee get per year under the {p}?"),
        ("days", "study leave days per year", "int", "days", _ints(2, 3, 4, 5, 6, 8, 10),
         "How many days of study leave a year does the {p} give?"),
        ("clawback", "repayment period after funded courses", "int", "months", _ints(6, 12, 18, 24, 36),
         "Within what period after a funded course must fees be repaid on leaving, under the {p}?")]),
]
