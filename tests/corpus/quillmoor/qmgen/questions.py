"""Evaluation questions and training rows, all derived from the facts and the documents that state them.

Question types:
  verbatim     -- the training phrasing (memorisation; only meaningful for a model trained on these rows)
  paraphrase   -- re-worded, never used for training (does the knowledge carry over to new wording?)
  superseded   -- a fact that changed: the newest value is right, the outdated one (still in older files) is wrong
  multihop     -- needs two facts from two files (asset -> its site -> the site's manager, ...)
  reverse      -- find the entity from a value (which asset has serial SN...?)
  unanswerable -- the documents do not say (an attribute nobody wrote down, or an entity that does not exist): the
                  right answer is "the documents don't say"
Every answerable question carries ``evidence``: the files and the exact strings that prove the answer, so a retrieval
hit can be checked mechanically. Correctness of an ANSWER is decided by the judge, never by string matching.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from .model import Doc
from .specs import spec_for
from .world import Fact, World

SCAN_EXT = ".pdf(scan)"


def doc_format(d: Doc) -> str:
    return SCAN_EXT if d.meta.get("scan") else d.ext


def fspec(f: Fact):
    return spec_for(f.etype, f.attr, f.ctx.get("label", ""), f.ctx.get("para", ""), f.ctx.get("code", ""))


def fill(tpl: str, f: Fact, v: str = "") -> str:
    q = tpl.format(k=f.ekey, v=v, t=f.ctx.get("t", ""), s=f.ctx.get("s", ""))
    return " ".join(q.split())


class QuestionMaker:
    def __init__(self, w: World, docs: list[Doc]) -> None:
        self.w, self.docs = w, docs
        self.by_id = {d.doc_id: d for d in docs}
        self.r = random.Random(f"questions-{w.seed}-{w.tier}")
        self.marks: dict[str, list[tuple[str, str, bool]]] = defaultdict(list)   # fid -> (doc_id, written, superseded)
        for d in docs:
            for m in d.facts:
                self.marks[m.fact_id].append((d.doc_id, m.written, m.superseded))

    def evidence(self, f: Fact) -> list[dict[str, str]]:
        return [{"fact": f.fid, "doc": self.by_id[did].relpath, "format": doc_format(self.by_id[did]), "written": wr}
                for did, wr, old in self.marks[f.fid] if not old]

    def stale(self, f: Fact) -> list[dict[str, str]]:
        """Files that still state an outdated value of ``f`` (older policy version, the directory before a desk move)."""
        return [{"fact": f.fid, "doc": self.by_id[did].relpath, "written": wr} for did, wr, old in self.marks[f.fid] if old]

    def stated(self, f: Fact) -> bool:
        return any(not old for _, _, old in self.marks[f.fid])

    def q(self, qid: str, qtype: str, question: str, facts: list[Fact], answer: str, accept: list[str],
          wrong: list[str] | None = None, **extra: Any) -> dict[str, Any]:
        ev = [e for f in facts for e in self.evidence(f)]
        return {"id": qid, "type": qtype, "question": question, "answer": answer, "accept": accept, "wrong": wrong or [],
                "facts": [f.fid for f in facts], "etype": facts[0].etype if facts else extra.pop("etype", ""),
                "attr": facts[-1].attr if facts else "", "evidence": ev,
                "formats": sorted({e["format"] for e in ev}), "stale": [x for f in facts for x in self.stale(f)], **extra}

    # ── per-fact questions ──
    def single(self, f: Fact, qtype: str, n: int) -> dict[str, Any]:
        sp = fspec(f)
        tpl = self.r.choice(sp.ask if qtype == "verbatim" else sp.para)
        wrong = [f.canonical(o) for o in f.old]
        return self.q(f"{qtype}-{n:05d}", "superseded" if (f.old and qtype == "paraphrase") else qtype, fill(tpl, f), [f],
                      f.canonical(), f.accepted(), wrong)

    def training_rows(self) -> list[dict[str, Any]]:
        """Q&A rows a user would train an adapter on: the direct phrasing + the answer sentence, current values only."""
        rows = []
        for f in self.w.facts.values():
            if not self.stated(f):
                continue
            sp = fspec(f)
            for tpl in sp.ask:
                rows.append({"messages": [{"role": "user", "content": fill(tpl, f)},
                                          {"role": "assistant", "content": fill(sp.answer, f, f.canonical())}],
                             "fact": f.fid})
        return rows

    # ── the eval set ──
    def build(self, per_format: int = 12, n_verbatim: int = 150, n_multihop: int = 50, n_reverse: int = 40,
              n_unanswerable: int = 60) -> list[dict[str, Any]]:
        w, r = self.w, self.r
        facts = [f for f in w.facts.values() if self.stated(f)]
        # single-source facts, bucketed by the one format that states them: per-format accuracy is attributable
        buckets: dict[str, list[Fact]] = defaultdict(list)
        for f in facts:
            ev = self.evidence(f)
            fmts = {e["format"] for e in ev}
            if len(fmts) == 1 and not f.old:
                buckets[next(iter(fmts))].append(f)
        chosen: list[Fact] = []
        for fmt in sorted(buckets):
            pool = buckets[fmt][:]
            r.shuffle(pool)
            # spread over entity types and attributes inside a format
            seen: dict[tuple[str, str], int] = defaultdict(int)
            pool.sort(key=lambda f: seen.setdefault((f.etype, f.attr), 0))
            picked: list[Fact] = []
            by_attr: dict[tuple[str, str], list[Fact]] = defaultdict(list)
            for f in pool:
                by_attr[(f.etype, f.attr)].append(f)
            while len(picked) < per_format and any(by_attr.values()):
                for key in sorted(by_attr):
                    if by_attr[key] and len(picked) < per_format:
                        picked.append(by_attr[key].pop())
            chosen += picked
        out: list[dict[str, Any]] = []
        for i, f in enumerate(chosen):
            out.append(self.single(f, "paraphrase", i))
        verb = r.sample(chosen, k=min(n_verbatim, len(chosen)))
        for i, f in enumerate(verb):
            out.append(self.single(f, "verbatim", i))
        for i, f in enumerate([f for f in facts if f.old]):
            out.append(self.single(f, "paraphrase", 50000 + i))
        out += self.multihop(n_multihop)
        out += self.reverse(n_reverse)
        out += self.unanswerable(n_unanswerable)
        return out

    def multihop(self, n: int) -> list[dict[str, Any]]:
        w, r = self.w, self.r
        makers = []
        for a in w.e["asset"]:
            makers.append(lambda a=a: (f"Who manages the site where the {w.get('asset', a.key, 'model').ctx['t']} {a.key} is installed?",
                                       [w.get("asset", a.key, "site"), w.get("site", a.a["site"], "manager")]))
        for inc in w.e["incident"]:
            makers.append(lambda inc=inc: (f"What is the service interval of the asset that was involved in incident {inc.key}?",
                                           [w.get("incident", inc.key, "asset"), w.get("asset", inc.a["asset"], "service_interval")]))
        for p in w.e["part"]:
            makers.append(lambda p=p: (f"What payment terms apply to the supplier of stock item {p.key}?",
                                       [w.get("part", p.key, "supplier"), w.get("supplier", p.a["supplier"], "payment_terms")]))
        for pr in w.e["project"]:
            pm = w.get("project", pr.key, "pm").raw
            makers.append(lambda pr=pr, pm=pm: (f"What is the employee ID of the project manager of {pr.key}?",
                                                [w.get("project", pr.key, "pm"), w.get("person", pm, "employee_id")]))
        for c in w.e["customer"]:
            am = w.get("customer", c.key, "account_manager").raw
            makers.append(lambda c=c, am=am: (f"At which site does the account manager of {c.key} work?",
                                              [w.get("customer", c.key, "account_manager"), w.get("person", am, "site")]))
        r.shuffle(makers)
        out = []
        for mk in makers:
            if len(out) >= n:
                break
            question, chain = mk()
            if not all(self.stated(f) for f in chain) or chain[-1].old:
                continue
            last = chain[-1]
            out.append(self.q(f"multihop-{len(out):05d}", "multihop", question, chain, last.canonical(), last.accepted()))
        return out

    def reverse(self, n: int) -> list[dict[str, Any]]:
        w, r = self.w, self.r
        makers = []
        for a in w.e["asset"]:
            f = w.get("asset", a.key, "serial")
            makers.append((f"Which asset carries the manufacturer serial number {f.raw}?", f, a.key))
        for c in w.e["customer"]:
            f = w.get("customer", c.key, "account_no")
            makers.append((f"Which customer is billed under account number {f.raw}?", f, c.key))
        for v in w.e["vessel"]:
            f = w.get("vessel", v.key, "call_sign")
            makers.append((f"Which vessel uses the call sign {f.raw}?", f, v.key))
        for s in w.e["supplier"]:
            f = w.get("supplier", s.key, "vendor_no")
            makers.append((f"Which supplier has vendor number {f.raw}?", f, s.key))
        for p in w.e["person"]:
            f = w.get("person", p.key, "employee_id")
            makers.append((f"Whose employee ID is {f.raw}?", f, p.key))
        r.shuffle(makers)
        out = []
        for question, f, ans in makers:
            if len(out) >= n:
                break
            if self.stated(f):
                out.append(self.q(f"reverse-{len(out):05d}", "reverse", question, [f], ans, [ans]))
        return out

    def unanswerable(self, n: int) -> list[dict[str, Any]]:
        w, r = self.w, self.r
        keys = {(e.etype, e.key) for es in w.e.values() for e in es}
        missing_attr = [
            ("asset", "What is the warranty period of {k}?"), ("person", "What is the home address of {k}?"),
            ("customer", "What is the credit limit of {k}?"), ("service", "What is the IP address of the server that runs {k}?"),
            ("site", "How many visitors did {k} receive last year?"), ("supplier", "What is the fax number of {k}?"),
            ("contract", "Which court has jurisdiction over disputes under contract {k}?"), ("project", "How many risks are open on {k}?"),
            ("vessel", "What is the fuel consumption of {k} per hour?"), ("part", "What is the shelf life of {k}?"),
            ("chemical", "What is the CAS number of {k}?"), ("incident", "What was the weather like during {k}?"),
        ]
        fakes = []
        for _ in range(n * 3):
            t = r.choice(["PMP", "TRB", "VLV", "TRF"])
            fakes.append(("asset", f"{t}-{r.randint(1000, 9999)}-Q", "What is the service interval of {k}?"))
            fakes.append(("contract", f"QC-2017-{r.randint(1, 999):03d}", "When does contract {k} end?"))
            fakes.append(("part", f"QP-{r.randint(100000, 999999)}", "What is the unit price of {k}?"))
            fakes.append(("incident", f"INC-2022-{r.randint(1, 9999):04d}", "What was the root cause of incident {k}?"))
        out = []
        r.shuffle(fakes)
        for et, key, tpl in fakes:
            if len(out) >= n // 2:
                break
            if (et, key) in keys:
                continue
            out.append({"id": f"unanswerable-{len(out):05d}", "type": "unanswerable", "question": tpl.format(k=key),
                        "answer": "", "accept": [], "wrong": [], "facts": [], "etype": et, "attr": "nonexistent_entity",
                        "evidence": [], "formats": []})
        while len(out) < n:
            et, tpl = r.choice(missing_attr)
            e = r.choice(w.e[et])
            out.append({"id": f"unanswerable-{len(out):05d}", "type": "unanswerable", "question": tpl.format(k=e.key),
                        "answer": "", "accept": [], "wrong": [], "facts": [], "etype": et, "attr": "missing_attribute",
                        "evidence": [], "formats": []})
        return out
