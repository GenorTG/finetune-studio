"""Integrity audit of a saved test run: are the stored verdicts backed by judgements?

Nothing is re-scored here (there is no scripted scorer to recompute with): the audit checks that the numbers on
screen come from somewhere — every verdict has a judgement row from a human, an AI judge or the official exact
matcher — and reports what is still awaiting judgement.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

from finetune_studio.db import judgements as jdb


def audit_cases(bid: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise verdict provenance for a run's cases."""
    by_case: dict[str, list[dict[str, Any]]] = {}
    for j in jdb.list_judgements(benchmark_id=bid):
        by_case.setdefault(j["case_id"], []).append(j)

    malformed: list[str] = []
    unbacked: list[str] = []
    disagreements: list[dict[str, Any]] = []
    for case in cases:
        cid = str(case.get("id", ""))
        transcript = case.get("transcript", [])
        if isinstance(transcript, str):
            try:
                transcript = json.loads(transcript)
            except json.JSONDecodeError:
                malformed.append(cid)
        winner = jdb.effective_judgement(by_case.get(cid, []))
        stored = str(case.get("verdict") or "")
        if stored and (winner is None or winner["verdict"] != stored):
            unbacked.append(cid)
        ai = [j for j in by_case.get(cid, []) if j["kind"] == "ai" and j["verdict"]]
        if len({j["verdict"] for j in ai[-2:]}) > 1:  # the last two AI opinions differ
            disagreements.append({"id": cid, "ai": [{"model": j["judge_model"], "verdict": j["verdict"]} for j in ai[-2:]]})

    return {
        "case_count": len(cases),
        "malformed_transcripts": malformed,
        "verdicts_without_judgement": unbacked,
        "verdict_counts": dict(Counter(str(c.get("verdict") or "awaiting") for c in cases)),
        "verdict_sources": dict(Counter(str(c.get("judge") or "none") for c in cases if c.get("verdict"))),
        "awaiting": sum(1 for c in cases if not c.get("verdict")),
        "judge_disagreements": disagreements,
        "agreement_with_human": jdb.agreement(bid),
        "passed": not malformed and not unbacked,
    }
