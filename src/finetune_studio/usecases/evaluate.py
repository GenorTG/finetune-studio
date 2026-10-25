"""Base vs trained run on a use case's held-out questions: start, score, A/B, report.

An evaluation is an ordinary compare group (``webui.compare_jobs``): the untrained base the run was trained from and
the run's model answer the same held-out suite, same instruction, temperature 0. Scoring follows the template:

* extraction / classification — code scores each answer when the run is saved (:func:`apply_scoring`, called by the
  compare and test-run jobs for any use-case suite);
* distillation — the AI judge (auto-judge of the group) grades each answer against the teacher's answer;
* style — the blind A/B job (:func:`start_ab`, job kind ``usecase_ab``) compares base and trained answers.

The report pairs the two models per question: McNemar on pass / fail (``testing/paired.py``), or the A/B sign test.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from finetune_studio import db
from finetune_studio.usecases import metrics, store
from finetune_studio.usecases import style_ab as ab
from finetune_studio.usecases.templates import get_template

log = logging.getLogger(__name__)


class EvaluationConflict(RuntimeError):
    """The evaluation cannot start or continue in the current state (HTTP 409)."""


def _display(path: str) -> str:
    from finetune_studio.naming import display_for_path

    return display_for_path(path)


def start(pid: str, uid: str, run_id: str, *, judge_provider_id: str = "") -> dict[str, Any]:
    """Launch base + run on the held-out suite. Raises ``LookupError`` (404), :class:`EvaluationConflict` (409),
    ``TestingBusy`` (409: the GPU or another test job is busy)."""
    from finetune_studio.testing.suite import load_test_suite
    from finetune_studio.webui import compare_jobs
    from finetune_studio.webui.routes.benchmarks import _resolve_trained_target
    from finetune_studio.webui.testing_models import target_model

    manifest = store.get(pid, uid)
    run = db.get_run(run_id) if run_id else None
    if not run or run.get("project_id") != pid:
        raise LookupError("run not found")
    if run.get("status") != "done" or not str(run.get("output_path") or "").strip():
        raise EvaluationConflict(f"run {run_id} has no trained model (status: {run.get('status') or 'unknown'})")
    base = str(run.get("base_model") or "").strip()
    if not base:
        raise EvaluationConflict(f"run {run_id} does not record the base model it was trained from")
    try:
        base_path = target_model(pid, base)
        tuned_path = target_model(pid, _resolve_trained_target(run))
    except ValueError as exc:
        raise EvaluationConflict(str(exc)) from None
    if base_path == tuned_path:
        raise EvaluationConflict("the run's model and its base are the same path; nothing to compare")
    suite_path = manifest["suite"]["path"]
    try:
        cases = load_test_suite(suite_path)
    except (OSError, TypeError, ValueError) as exc:
        raise EvaluationConflict(f"the held-out suite cannot be read ({exc}); build the use case again") from None
    template = manifest["template"]
    plan = manifest.get("training") or {}
    spec = compare_jobs.CompareSpec(
        project_id=pid, suite_name=manifest["suite"]["name"], cases=cases,
        models=[compare_jobs.CompareModel(label=f"Untrained base · {_display(base_path)}", model_path=base_path),
                compare_jobs.CompareModel(label=_display(tuned_path), model_path=tuned_path)],
        max_tokens=int(plan.get("eval_max_tokens") or 512), temperature=float(plan.get("eval_temperature") or 0.0),
        config={"suite_path": suite_path, "usecase": {"id": uid, "template": template}},
        auto_judge=template == "distillation", judge_provider_id=judge_provider_id,
    )
    rows = compare_jobs.launch_compare(spec)
    record = {"group_id": rows[0]["config"]["compare"]["group_id"], "run_id": run_id, "base_bid": rows[0]["id"],
              "tuned_bid": rows[1]["id"], "base_model": base_path, "tuned_model": tuned_path,
              "started_at": time.time()}
    store.add_evaluation(pid, uid, record)
    return record


def apply_scoring(bid: str, config: dict[str, Any]) -> dict[str, Any] | None:
    """Code-score a saved run of an extraction / classification suite (no-op for any other suite). Never raises."""
    try:
        meta = store.suite_usecase(str((config or {}).get("suite_path") or ""))
        if not meta or meta.get("template") not in metrics.EXACT_TEMPLATES:
            return None
        return metrics.score_benchmark(bid, str(meta["template"]), meta)
    except Exception:
        log.exception("use-case scoring of run %s failed", bid)
        return None


def _evaluation(manifest: dict[str, Any], group_id: str) -> dict[str, Any]:
    for rec in manifest.get("evaluations") or []:
        if rec.get("group_id") == group_id:
            return rec
    raise LookupError("evaluation not found")


def _key(case: dict[str, Any]) -> str:
    return f"{str(case.get('case_name') or '').strip()}\x1f{str(case.get('question') or '').strip()}"


def _code_outcomes(template: str, cases: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, bool]:
    out = {}
    for c in cases:
        answer = c.get("model_answer") or ""
        if c.get("error") and not answer.strip():
            continue
        if template == "extraction":
            ok = metrics.score_extraction(answer, c.get("correct_answer") or "{}", meta.get("schema")).exact
        else:
            ok = metrics.score_classification(answer, c.get("correct_answer") or "",
                                              list(meta.get("labels") or [])).verdict == "pass"
        out[_key(c)] = ok
    return out


def _side(row: dict[str, Any] | None, cases: list[dict[str, Any]]) -> dict[str, Any]:
    scores = (row or {}).get("scores") if isinstance((row or {}).get("scores"), dict) else {}
    return {"benchmark_id": (row or {}).get("id"), "model_path": (row or {}).get("model_path", ""),
            "status": (row or {}).get("status", ""), "error": (row or {}).get("error", ""),
            "judge_status": (row or {}).get("judge_status", ""), "answered": len(cases),
            "scores": {k: scores.get(k) for k in ("total", "judged", "awaiting", "passed", "partial", "failed",
                                                  "pass_rate")}}


def report(pid: str, uid: str, group_id: str) -> dict[str, Any]:
    """The paired base-vs-run result of one evaluation. ``LookupError`` when it does not exist."""
    from finetune_studio.compare.session import group_rows, group_status
    from finetune_studio.testing.paired import case_outcomes, compare_outcomes

    manifest = store.get(pid, uid)
    rec = _evaluation(manifest, group_id)
    template = get_template(manifest["template"])
    rows = {r["id"]: r for r in group_rows(pid, group_id)}
    base_row, tuned_row = rows.get(rec["base_bid"]), rows.get(rec["tuned_bid"])
    base_cases = db.list_cases(rec["base_bid"]) if base_row else []
    tuned_cases = db.list_cases(rec["tuned_bid"]) if tuned_row else []
    status = group_status(list(rows.values())) if rows else {"run": "missing", "judge": ""}
    out: dict[str, Any] = {
        "usecase": {"id": uid, "name": manifest.get("name"), "template": template.id, "metric": template.metric},
        "evaluation": rec, "status": status, "base": _side(base_row, base_cases), "tuned": _side(tuned_row, tuned_cases),
        "paired": None, "ab": None, "notes": [],
    }
    answering = status.get("run") == "running"
    meta = store.suite_usecase(manifest["suite"]["path"]) or {"labels": manifest.get("labels"),
                                                                "schema": manifest.get("schema")}
    if template.id in metrics.EXACT_TEMPLATES:
        if base_cases:
            out["base"]["metrics"] = metrics.summarize(template.id, base_cases, meta)
        if tuned_cases:
            out["tuned"]["metrics"] = metrics.summarize(template.id, tuned_cases, meta)
        if not answering:
            out["paired"] = compare_outcomes(_code_outcomes(template.id, base_cases, meta),
                                             _code_outcomes(template.id, tuned_cases, meta)).as_dict()
    elif template.id == "distillation":
        a, b = case_outcomes(base_cases), case_outcomes(tuned_cases)
        if not answering and (a or b):
            out["paired"] = compare_outcomes(a, b).as_dict()
        awaiting = sum(1 for c in base_cases + tuned_cases if not c.get("verdict"))
        if awaiting:
            out["notes"].append(f"{awaiting} answer(s) are not judged yet: judge the group (Compare page or "
                                "POST /api/compare/projects/{pid}/groups/{gid}/judge) before reading the score")
    else:
        out["ab"] = store.read_ab(pid, uid, group_id)
        if out["ab"] is None and not answering:
            out["notes"].append("run the blind A/B judge to get the style result")
    paired = out["paired"]
    if paired and paired.get("verdict") == "could be noise":
        out["notes"].append("the gained / lost split could be chance (p >= 0.05): more held-out questions, or a "
                            "bigger effect, are needed before calling it a win")
    return out


# ── Style: blind A/B job ──────────────────────────────────────────────────────


def start_ab(pid: str, uid: str, group_id: str, provider_id: str = "") -> dict[str, Any]:
    """Launch the A/B judge job for a finished style evaluation. Raises like :func:`start`, plus ``ValueError``."""
    from finetune_studio.compare.session import group_rows
    from finetune_studio.testing.judge import default_judge_provider_id
    from finetune_studio.webui import testing_jobs

    manifest = store.get(pid, uid)
    if manifest["template"] != "style":
        raise ValueError("the blind A/B judge is the style template's metric; this use case is scored "
                         f"{'by code' if manifest['template'] in metrics.EXACT_TEMPLATES else 'by the AI judge'}")
    rec = _evaluation(manifest, group_id)
    rows = {r["id"]: r for r in group_rows(pid, group_id)}
    base_row, tuned_row = rows.get(rec["base_bid"]), rows.get(rec["tuned_bid"])
    if not base_row or not tuned_row:
        raise LookupError("evaluation runs not found")
    if base_row.get("status") != "done" or tuned_row.get("status") != "done":
        raise EvaluationConflict("both models must finish answering before the A/B judge runs")
    pairs = ab.pair_cases(db.list_cases(rec["base_bid"]), db.list_cases(rec["tuned_bid"]))
    if not pairs:
        raise EvaluationConflict("no question was answered by both models")
    provider = provider_id or default_judge_provider_id()
    if not provider:
        raise EvaluationConflict("no judge configured: pick a judge provider (Settings -> Testing) or pass one")
    local = testing_jobs._provider_is_local(provider)
    bids = [rec["base_bid"], rec["tuned_bid"]]
    testing_jobs._assert_free(bids, "judge", uses_gpu=local)
    state = {"status": "running", "provider_id": provider, "judge_model": "", "done": 0, "total": len(pairs),
             "started_at": time.time(), "prompt_version": ab.AB_PROMPT_VERSION}
    store.write_ab(pid, uid, group_id, state)
    try:
        testing_jobs._launch(testing_jobs.KIND_USECASE_AB, role="judge", bids=bids, uses_gpu=local, lease=local, project_id=pid,
                             title=f"style A/B {manifest.get('name') or uid}",
                             params={"pid": pid, "uid": uid, "group_id": group_id, "provider_id": provider,
                                     "style_guide": manifest.get("style_guide") or ""})
    except testing_jobs.TestingBusy as exc:
        store.write_ab(pid, uid, group_id, {**state, "status": "failed", "error": f"not started: {exc}"})
        raise
    return state


def ab_in_job(ctx: Any) -> dict[str, Any]:
    """The job side of the style A/B (``jobs.kinds.usecase_ab``): judge every pair, write the result file."""
    from finetune_studio.testing.judge import open_judge

    p = ctx.params
    pid, uid, group_id = p["pid"], p["uid"], p["group_id"]
    rec = _evaluation(store.get(pid, uid), group_id)
    state = store.read_ab(pid, uid, group_id) or {}
    pairs = ab.pair_cases(db.list_cases(rec["base_bid"]), db.list_cases(rec["tuned_bid"]))
    results: list[ab.ABResult] = []
    last = 0.0
    status, error = "done", ""
    try:
        with open_judge(str(p["provider_id"])) as judge:
            state["judge_model"] = judge.model

            def on_result(r: ab.ABResult) -> None:
                nonlocal last
                results.append(r)
                now = time.time()
                if now - last > 1.0 or len(results) == len(pairs):
                    last = now
                    store.write_ab(pid, uid, group_id, {**state, "done": len(results)})

            ab.run_ab(judge.chat, pairs, style_guide=str(p.get("style_guide") or ""), concurrent=judge.concurrent,
                      on_result=on_result, should_stop=ctx.stop.is_set)
        if ctx.stop.is_set():
            status = "cancelled"
    except Exception as exc:  # noqa: BLE001 - JudgeUnavailable and provider errors alike; written to the file
        log.warning("style A/B %s failed: %s", group_id, exc)
        status, error = "failed", str(exc)
    summary = ab.summarize(results)
    store.write_ab(pid, uid, group_id, {
        **state, "status": status, "error": error, "done": len(results), "finished_at": time.time(),
        "summary": summary.as_dict(),
        "results": [{"key": r.key, "outcome": r.outcome, "first": r.first, "second": r.second,
                     "reasoning": r.reasoning, "error": r.error} for r in results],
    })
    return {"status": {"done": "done", "cancelled": "stopped"}.get(status, "failed"), "group_id": group_id}
