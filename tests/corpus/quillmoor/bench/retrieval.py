"""Retrieval metrics against a Smart Brain search endpoint (an exported package's /search, or the app's own search).

A question is "found at rank r" when a passage at rank r comes from a file that states the answer AND holds the exact
value string the generator wrote there; a multi-hop question needs every hop found (its rank is the worst hop's).
These are retrieval facts, not answer verdicts: whether an answer is right is the judge's call (judge.py).
"""
from __future__ import annotations

import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePath
from typing import Any

import httpx
from common import norm

KS = (1, 3, 5, 10)
ANSWERABLE = ("paraphrase", "verbatim", "superseded", "reverse", "multihop")


def package_search(base: str, top_k: int = 10) -> Callable[[str], list[dict]]:
    def fn(q: str) -> list[dict]:
        r = httpx.post(f"{base}/search", json={"query": q, "top_k": top_k, "dense_extra": 0}, timeout=120)
        r.raise_for_status()
        return r.json()["hits"]
    return fn


def rank_of(q: dict, hits: list[dict]) -> tuple[int | None, bool]:
    """(rank the question is fully answered at, stale-first) — rank None = not within the hits."""
    by_fact: dict[str, list[dict]] = defaultdict(list)
    for e in q["evidence"]:
        by_fact[e["fact"]].append(e)
    normed = [(PurePath(h.get("filename") or "").name, norm(h.get("text") or "")) for h in hits]
    worst = 0
    for evs in by_fact.values():
        best = None
        for r, (fn, txt) in enumerate(normed, 1):
            if any(PurePath(e["doc"]).name == fn and norm(e["written"]) in txt for e in evs):
                best = r
                break
        if best is None:
            return None, False
        worst = max(worst, best)
    stale_first = False
    if q.get("stale"):
        for r, (fn, txt) in enumerate(normed, 1):
            if r >= worst:
                break
            if any(PurePath(s["doc"]).name == fn and norm(s["written"]) in txt for s in q["stale"]):
                stale_first = True
                break
    return worst, stale_first


def evaluate(questions: list[dict], search: Callable[[str], list[dict]], *, workers: int = 4) -> dict[str, Any]:
    qs = [q for q in questions if q["type"] in ANSWERABLE]
    lat: list[float] = []

    def one(q: dict) -> dict:
        t0 = time.time()
        hits = search(q["question"])
        dt = time.time() - t0
        r, stale = rank_of(q, hits)
        return {"id": q["id"], "type": q["type"], "formats": q["formats"], "etype": q["etype"], "rank": r, "stale_first": stale,
                "latency": dt}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(one, qs))
    lat = [x["latency"] for x in rows]

    def agg(sel: list[dict]) -> dict[str, Any]:
        n = len(sel)
        out: dict[str, Any] = {"n": n}
        for k in KS:
            out[f"hit@{k}"] = round(sum(1 for x in sel if x["rank"] is not None and x["rank"] <= k) / max(n, 1), 4)
        out["mrr@10"] = round(sum(1 / x["rank"] for x in sel if x["rank"] is not None and x["rank"] <= 10) / max(n, 1), 4)
        return out

    by_type = {t: agg([x for x in rows if x["type"] == t]) for t in ANSWERABLE if any(x["type"] == t for x in rows)}
    single = [x for x in rows if x["type"] in ("paraphrase", "verbatim") and len(x["formats"]) == 1]
    by_fmt = {f: agg([x for x in single if x["formats"][0] == f]) for f in sorted({x["formats"][0] for x in single})}
    sup = [x for x in rows if x["type"] == "superseded"]
    return {"overall": agg([x for x in rows if x["type"] != "verbatim"]), "by_type": by_type, "by_format": by_fmt,
            "stale_first": round(sum(x["stale_first"] for x in sup) / max(len(sup), 1), 4),
            "latency_ms": {"p50": round(1000 * statistics.median(lat), 1),
                           "p95": round(1000 * sorted(lat)[int(0.95 * (len(lat) - 1))], 1)} if lat else {},
            "rows": rows}
