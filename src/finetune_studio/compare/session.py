"""A compare group read back as one side-by-side view.

A group is the set of test runs (``kind == "compare"``) that one compare request created, one per model, all asked
the same questions. This module only reads: it never writes a verdict and never scores an answer.
"""
from __future__ import annotations

from typing import Any

from finetune_studio import db
from finetune_studio.db import judgements as jdb

COMPARE_KIND = "compare"
_ACTIVE_STATUSES = ("queued", "running")
_VERDICT_ORDER = ("pass", "partial", "fail")


def group_config(row: dict[str, Any]) -> dict[str, Any]:
    cfg = (row.get("config") or {}).get("compare")
    return cfg if isinstance(cfg, dict) else {}


def group_rows(project_id: str, group_id: str) -> list[dict[str, Any]]:
    """The group's runs in the order the models were requested ([] when there is no such group in the project)."""
    rows = [r for r in db.list_benchmarks_for_project(project_id, limit=5000)
            if r.get("kind") == COMPARE_KIND and group_config(r).get("group_id") == group_id]
    rows.sort(key=lambda r: int(group_config(r).get("index") or 0))
    return rows


def _rollup(statuses: list[str], *, active: tuple[str, ...]) -> str:
    """One status for a set of rows: still going > failed > cancelled > done ('' when nothing has started)."""
    if any(s in active for s in statuses):
        return "running"
    for s in ("failed", "cancelled"):
        if s in statuses:
            return s
    return "done" if statuses and all(s == "done" for s in statuses) else ""


def group_status(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Roll the per-run job states up to the group: ``run`` (answering) and ``judge`` (judging)."""
    return {
        "run": _rollup([str(r.get("status") or "") for r in rows], active=_ACTIVE_STATUSES) or "done",
        "judge": _rollup([str(r.get("judge_status") or "") for r in rows], active=("running",)),
    }


def list_groups(project_id: str) -> list[dict[str, Any]]:
    """Every compare group of a project, newest first, with the headline numbers of each model."""
    by_group: dict[str, list[dict[str, Any]]] = {}
    for r in db.list_benchmarks_for_project(project_id, limit=5000):
        if r.get("kind") == COMPARE_KIND and group_config(r).get("group_id"):
            by_group.setdefault(str(group_config(r)["group_id"]), []).append(r)
    out = []
    for gid, rows in by_group.items():
        rows.sort(key=lambda r: int(group_config(r).get("index") or 0))
        out.append({
            "group_id": gid, "suite_name": rows[0].get("suite_name", ""), "ran_at": min(r.get("ran_at") or 0 for r in rows),
            "status": group_status(rows), "models": [_model_summary(r) for r in rows],
        })
    out.sort(key=lambda g: g["ran_at"], reverse=True)
    return out


def _model_summary(row: dict[str, Any]) -> dict[str, Any]:
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    return {
        "benchmark_id": row["id"], "label": group_config(row).get("label") or row.get("model_path", ""),
        "model_path": row.get("model_path", ""), "status": row.get("status", ""), "error": row.get("error", ""),
        "progress_done": row.get("progress_done", 0), "progress_total": row.get("progress_total", 0),
        "judge_status": row.get("judge_status", ""), "judge_model": row.get("judge_model", ""),
        "judge_done": row.get("judge_done", 0), "judge_total": row.get("judge_total", 0),
        "judge_error": row.get("judge_error", ""), "time_ms": row.get("time_ms", 0),
        # pass_rate is None until something is judged: "awaiting judgement" is not 0 %.
        "scores": {k: scores.get(k) for k in ("total", "judged", "awaiting", "passed", "partial", "failed",
                                              "pass_rate", "weighted_score", "avg_time_ms", "by_judge")},
    }


def _case_keys(cases: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """``(case_name, n-th occurrence)`` per case, so a suite with a repeated name still lines up row by row."""
    seen: dict[str, int] = {}
    keys = []
    for c in cases:
        name = str(c.get("case_name") or "")
        keys.append((name, seen.get(name, 0)))
        seen[name] = seen.get(name, 0) + 1
    return keys


def side_by_side(project_id: str, group_id: str) -> dict[str, Any] | None:
    """The group as one table: per question, every model's saved answer next to the answer key and its verdicts.

    ``None`` when the group does not exist in the project. A model that has not answered a question yet (run still
    going, cancelled, failed) simply has no entry for it. ``verdict`` is the effective one (human > AI > exact);
    empty means awaiting judgement. ``disagree`` flags questions where the judged models got different verdicts.
    """
    rows = group_rows(project_id, group_id)
    if not rows:
        return None
    order: list[tuple[str, int]] = []
    questions: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        bid = row["id"]
        judgements: dict[str, list[dict[str, Any]]] = {}
        for j in jdb.list_judgements(benchmark_id=bid):
            judgements.setdefault(j["case_id"], []).append(j)
        cases = db.list_cases(bid)
        for key, c in zip(_case_keys(cases), cases, strict=True):
            if key not in questions:
                order.append(key)
                questions[key] = {
                    "case_name": c.get("case_name", ""), "category": c.get("category", ""),
                    "question": c.get("question", ""), "correct_answer": c.get("correct_answer", ""),
                    "expect_abstain": bool(c.get("expect_abstain")), "answers": {},
                }
            questions[key]["answers"][bid] = {
                "case_id": c["id"], "model_answer": c.get("model_answer", ""), "error": c.get("error", ""),
                "time_ms": c.get("time_ms", 0), "verdict": c.get("verdict", ""), "judge": c.get("judge", "none"),
                "judge_model": c.get("judge_model", ""), "judge_reasoning": c.get("judge_reasoning", ""),
                "judgements": [{k: j.get(k) for k in ("kind", "verdict", "reasoning", "confidence", "judge_model",
                                                      "provider_id", "error", "created_at")}
                               for j in judgements.get(c["id"], [])],
            }
    cases_out = []
    for i, key in enumerate(order):
        q = questions[key]
        verdicts = {a["verdict"] for a in q["answers"].values() if a["verdict"] in _VERDICT_ORDER}
        cases_out.append({"index": i, **q, "disagree": len(verdicts) > 1})
    return {
        "group_id": group_id, "suite_name": rows[0].get("suite_name", ""), "status": group_status(rows),
        "models": [_model_summary(r) for r in rows], "cases": cases_out,
        "agreement": {r["id"]: jdb.agreement(r["id"]) for r in rows},
    }
