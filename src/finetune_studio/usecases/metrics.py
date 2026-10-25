"""Deterministic metrics for the extraction and classification templates.

These two tasks have one right answer per question, so code scores them (no judge): an extraction passes when the
reply's JSON object equals the expected one, with field-F1 over the leaf values as the finer metric; a classification
passes when the reply names the right label. A reply that holds no JSON object / no label fails and is counted as
invalid, never repaired. ``score_benchmark`` writes the verdicts as ``exact`` judgements, so a later AI judge leaves
them alone and the Compare / McNemar views read them like any other verdict.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from finetune_studio.usecases import schema as schema_mod

EXACT_TEMPLATES = ("extraction", "classification")
NO_LABEL = "(no label)"
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)
_LABEL_PREFIX = re.compile(r"^(?:label|class|category|answer)\s*[:=-]\s*", re.IGNORECASE)
_STRIP = " \t\r\n\"'`*._:;,!()[]{}"


def strip_reasoning(text: str) -> str:
    return _THINK.sub("", text or "").strip()


def _json_spans(text: str) -> Iterator[str]:
    """Every balanced ``{...}`` span in ``text`` (string-aware)."""
    depth, start, in_str, esc = 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                yield text[start:i + 1]


def parse_json_reply(answer: str) -> tuple[dict[str, Any] | None, bool]:
    """(the reply's JSON object or None, whether the reply was that object and nothing else; a code fence is allowed)."""
    text = strip_reasoning(answer)
    fenced = _FENCE.match(text)
    body = fenced.group(1) if fenced else text
    try:
        obj = json.loads(body)
        if isinstance(obj, dict):
            return obj, True
    except ValueError:
        pass
    for span in _json_spans(text):
        try:
            obj = json.loads(span)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj, False
    return None, False


def _leaf(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip().casefold()
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return json.dumps(value, sort_keys=True)


def leaves(obj: Any, path: str = "") -> Counter[tuple[str, Any]]:
    """Multiset of ``(path, normalised value)`` leaves; list items share their list's path (order-free)."""
    out: Counter[tuple[str, Any]] = Counter()
    if isinstance(obj, dict):
        if not obj:
            out[(path, "{}")] += 1
        for key, val in obj.items():
            out.update(leaves(val, f"{path}.{key}" if path else str(key)))
    elif isinstance(obj, list):
        if not obj:
            out[(path, "[]")] += 1
        for item in obj:
            out.update(leaves(item, f"{path}[]"))
    else:
        out[(path, _leaf(obj))] += 1
    return out


def field_f1(predicted: Any, expected: Any) -> tuple[float, float, float]:
    """(precision, recall, F1) of the predicted leaves against the expected ones."""
    p, e = leaves(predicted), leaves(expected)
    hit = sum((p & e).values())
    n_p, n_e = sum(p.values()), sum(e.values())
    if n_p == 0 and n_e == 0:
        return 1.0, 1.0, 1.0
    precision = hit / n_p if n_p else 0.0
    recall = hit / n_e if n_e else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


@dataclass(frozen=True)
class ExtractionScore:
    verdict: str              # pass | partial | fail
    exact: bool
    json_valid: bool
    json_only: bool
    schema_valid: bool | None  # None when the use case has no schema
    precision: float
    recall: float
    f1: float
    reasoning: str


def score_extraction(answer: str, expected_json: str, schema: dict[str, Any] | None = None) -> ExtractionScore:
    expected = json.loads(expected_json)
    obj, only = parse_json_reply(answer)
    if obj is None:
        return ExtractionScore("fail", False, False, False, False if schema else None, 0.0, 0.0, 0.0,
                               "no JSON object in the reply")
    schema_ok = None if schema is None else not schema_mod.validate(obj, schema)
    precision, recall, f1 = field_f1(obj, expected)
    exact = obj == expected
    verdict = "pass" if exact else ("partial" if f1 > 0 else "fail")
    notes = [f"field-F1 {f1:.2f} (precision {precision:.2f}, recall {recall:.2f})"]
    if not exact:
        notes.insert(0, "object differs from the expected one")
    if not only:
        notes.append("the reply has text around the JSON")
    if schema_ok is False:
        notes.append("does not follow the schema")
    return ExtractionScore(verdict, exact, True, only, schema_ok, precision, recall, f1, "; ".join(notes))


@dataclass(frozen=True)
class ClassificationScore:
    verdict: str              # pass | fail
    predicted: str            # a label, or NO_LABEL
    label_only: bool          # the reply was the label and nothing else
    reasoning: str


def read_label(answer: str, labels: Iterable[str]) -> tuple[str, bool]:
    """(the label the reply names or NO_LABEL, whether the reply was just that label).

    A reply that is not just a label still counts when exactly one label occurs in it as a whole word; two or more
    different labels (or none) is no label."""
    labels = list(labels)
    by_fold = {lab.casefold(): lab for lab in labels}
    text = strip_reasoning(answer)
    bare = _LABEL_PREFIX.sub("", text.strip().strip(_STRIP)).strip(_STRIP).casefold()
    if bare in by_fold:
        return by_fold[bare], True
    found = {lab for lab in labels
             if re.search(r"(?<!\w)" + re.escape(lab.casefold()) + r"(?!\w)", text.casefold())}
    if len(found) == 1:
        return found.pop(), False
    return NO_LABEL, False


def score_classification(answer: str, expected: str, labels: list[str]) -> ClassificationScore:
    predicted, only = read_label(answer, labels)
    if predicted == NO_LABEL:
        return ClassificationScore("fail", NO_LABEL, False, f"no single label in the reply; expected {expected}")
    ok = predicted == expected
    note = "" if only else " (label found inside a longer reply)"
    return ClassificationScore("pass" if ok else "fail", predicted, only,
                               f"{predicted} {'=' if ok else '!='} expected {expected}{note}")


def confusion(pairs: Iterable[tuple[str, str]], labels: list[str]) -> dict[str, Any]:
    """Confusion matrix from ``(expected, predicted)`` pairs: rows = expected, columns = labels + NO_LABEL."""
    columns = [*labels, NO_LABEL]
    matrix = {lab: dict.fromkeys(columns, 0) for lab in labels}
    for exp, pred in pairs:
        if exp in matrix:
            matrix[exp][pred if pred in matrix[exp] else NO_LABEL] += 1
    per_label = {}
    for lab in labels:
        tp = matrix[lab][lab]
        support = sum(matrix[lab].values())
        predicted = sum(matrix[r][lab] for r in labels)
        per_label[lab] = {"support": support, "precision": tp / predicted if predicted else None,
                          "recall": tp / support if support else None}
    return {"labels": labels, "columns": columns, "matrix": matrix, "per_label": per_label}


def summarize(template_id: str, cases: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    """Aggregate metrics of one model's answered cases (``model_answer`` + ``correct_answer``)."""
    answered = [c for c in cases if not (c.get("error") and not (c.get("model_answer") or "").strip())]
    n = len(answered)
    if template_id == "extraction":
        scores = [score_extraction(c.get("model_answer") or "", c.get("correct_answer") or "{}", meta.get("schema"))
                  for c in answered]
        out: dict[str, Any] = {
            "answered": n, "exact_match": _share(sum(s.exact for s in scores), n),
            "field_f1": (sum(s.f1 for s in scores) / n) if n else None,
            "json_valid": _share(sum(s.json_valid for s in scores), n),
            "json_only": _share(sum(s.json_only for s in scores), n),
        }
        if meta.get("schema") is not None:
            out["schema_valid"] = _share(sum(bool(s.schema_valid) for s in scores), n)
        return out
    if template_id == "classification":
        labels = list(meta.get("labels") or [])
        scores = [(c.get("correct_answer") or "", score_classification(c.get("model_answer") or "",
                                                                       c.get("correct_answer") or "", labels))
                  for c in answered]
        return {
            "answered": n, "accuracy": _share(sum(s.verdict == "pass" for _, s in scores), n),
            "label_only": _share(sum(s.label_only for _, s in scores), n),
            "no_label": _share(sum(s.predicted == NO_LABEL for _, s in scores), n),
            "confusion": confusion([(exp, s.predicted) for exp, s in scores], labels),
        }
    raise ValueError(f"{template_id} is not scored by code")


def _share(k: int, n: int) -> float | None:
    return k / n if n else None


def score_benchmark(bid: str, template_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    """Write an ``exact`` verdict on every answered case of run ``bid`` and store the metric summary in its scores.

    Cases a human already decided keep the human verdict (it always wins). Returns the summary."""
    from finetune_studio import db
    from finetune_studio.db import judgements as jdb
    from finetune_studio.testing.scoring import rescore_benchmark

    if template_id not in EXACT_TEMPLATES:
        raise ValueError(f"{template_id} is not scored by code")
    cases = db.list_cases(bid)
    for case in cases:
        answer = case.get("model_answer") or ""
        if case.get("error") and not answer.strip():
            continue
        if template_id == "extraction":
            s = score_extraction(answer, case.get("correct_answer") or "{}", meta.get("schema"))
            verdict, reason = s.verdict, s.reasoning
        else:
            c = score_classification(answer, case.get("correct_answer") or "", list(meta.get("labels") or []))
            verdict, reason = c.verdict, c.reasoning
        jdb.add_judgement(case["id"], kind="exact", verdict=verdict, reasoning=reason,
                          judge_model=f"usecase:{template_id}")
    summary = summarize(template_id, db.list_cases(bid), meta)
    scores = rescore_benchmark(bid)
    db.update_benchmark_scores(bid, {**scores, "usecase": {"template": template_id, **summary}})
    return summary
