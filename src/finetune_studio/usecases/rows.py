"""Validate a use case's examples and shape them into SFT rows plus a held-out quiz (seeded split).

Every example is checked; a bad row refuses the build with the row numbers and reasons (the user fixes the data, the
app never guesses). Exact duplicates are merged; an input that appears with two different answers is dropped and
counted in the warnings, because it would teach and test both. The held-out questions are never in the training rows,
and every row and held-out question carries the same instruction as a system turn, so the base model is compared under
the instruction the trained model learned with.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from finetune_studio.usecases import schema as schema_mod
from finetune_studio.usecases.templates import (
    MIN_PER_LABEL,
    Template,
    TrainingPlan,
    get_template,
    heldout_size,
    recommend_settings,
)

MAX_ROWS = 20000
MAX_REPORTED_ERRORS = 20
SMALL_QUIZ = 50

_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "extraction": {"input": ("input", "text"), "target": ("output", "json", "target")},
    "classification": {"input": ("input", "text"), "target": ("label", "class", "category")},
    "style": {"input": ("prompt", "question", "input"), "target": ("answer", "response", "output")},
    "distillation": {"input": ("prompt", "question", "input"), "target": ("teacher_answer", "answer", "output")},
}


class RowsError(ValueError):
    """The examples cannot be built (message shown to the user; ``errors`` lists bad rows)."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


class TooFewRows(RowsError):
    """Valid examples, but fewer than the template needs."""


@dataclass
class BuildOptions:
    instruction: str = ""
    schema: dict[str, Any] | None = None
    labels: list[str] = field(default_factory=list)
    style_guide: str = ""
    teacher: str = ""
    seed: int = 42


@dataclass
class Prepared:
    template_id: str
    instruction: str
    train_rows: list[dict[str, Any]]
    heldout_cases: list[dict[str, Any]]
    labels: list[str]
    schema: dict[str, Any] | None
    style_guide: str
    teacher: str
    seed: int
    examples: int
    dropped_duplicates: int
    dropped_conflicts: int
    warnings: list[str]
    plan: TrainingPlan

    def summary(self) -> dict[str, Any]:
        return {
            "template": self.template_id, "examples": self.examples, "train_rows": len(self.train_rows),
            "heldout": len(self.heldout_cases), "labels": self.labels, "dropped_duplicates": self.dropped_duplicates,
            "dropped_conflicts": self.dropped_conflicts, "warnings": self.warnings, "instruction": self.instruction,
            "training": self.plan.as_dict(),
        }


def parse_upload(raw: bytes) -> list[Any]:
    """Rows from an uploaded file: a JSON list, ``{"rows": [...]}`` or one JSON object per line."""
    text = raw.decode("utf-8-sig", errors="replace").strip()
    if not text:
        raise RowsError("the file is empty")
    try:
        data: Any = json.loads(text)
    except ValueError:
        data = []
        for no, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    data.append(json.loads(line))
                except ValueError as e:
                    raise RowsError(f"line {no} is not valid JSON ({e})") from e
    if isinstance(data, dict):  # {"rows": [...]}, {"examples": [...]}, or a JSONL file of one example
        listed = data.get("rows", data.get("examples"))
        data = listed if isinstance(listed, list) else [data]
    if not isinstance(data, list) or not data:
        raise RowsError('expected a JSON list of examples, {"rows": [...]}, or one JSON object per line')
    return data


def _field(item: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in item and item[name] not in (None, ""):
            return item[name]
    return None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _extraction_target(value: Any, schema: dict[str, Any] | None) -> tuple[str, str]:
    """(canonical JSON text, '') or ('', error)."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return "", "output is not valid JSON"
    if not isinstance(value, dict):
        return "", "output must be a JSON object"
    if schema is not None:
        problems = schema_mod.validate(value, schema)
        if problems:
            return "", "output does not follow the schema: " + "; ".join(problems[:3])
    return json.dumps(value, ensure_ascii=False), ""


def _read_examples(template: Template, rows: list[Any], opts: BuildOptions) -> list[tuple[str, str]]:
    names = _ALIASES[template.id]
    out: list[tuple[str, str]] = []
    errors: list[str] = []
    for i, item in enumerate(rows, 1):
        if not isinstance(item, dict):
            errors.append(f"row {i}: not a JSON object")
            continue
        source = _field(item, names["input"])
        target = _field(item, names["target"])
        if not isinstance(source, str) or not source.strip():
            errors.append(f"row {i}: missing {' / '.join(names['input'])} text")
            continue
        if target is None:
            errors.append(f"row {i}: missing {' / '.join(names['target'])}")
            continue
        if template.id == "extraction":
            text, err = _extraction_target(target, opts.schema)
            if err:
                errors.append(f"row {i}: {err}")
                continue
        else:
            if not isinstance(target, (str, int, float)) or isinstance(target, bool):
                errors.append(f"row {i}: {names['target'][0]} must be text")
                continue
            text = str(target).strip()
            if not text:
                errors.append(f"row {i}: empty {names['target'][0]}")
                continue
        out.append((source.strip(), text))
    if errors:
        shown = errors[:MAX_REPORTED_ERRORS]
        more = f" (+{len(errors) - len(shown)} more)" if len(errors) > len(shown) else ""
        raise RowsError(f"{len(errors)} of {len(rows)} examples are not usable{more}", shown)
    return out


def _dedupe(examples: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], int, int]:
    """Merge exact duplicates; drop inputs that appear with different answers. Returns (rows, dupes, conflicts)."""
    by_input: dict[str, list[tuple[str, str]]] = {}
    for ex in examples:
        by_input.setdefault(_norm(ex[0]), []).append(ex)
    kept, dupes, conflicts = [], 0, 0
    for group in by_input.values():
        targets = {t for _, t in group}
        if len(targets) > 1:
            conflicts += len(group)
            continue
        kept.append(group[0])
        dupes += len(group) - 1
    return kept, dupes, conflicts


def _labels(examples: list[tuple[str, str]], given: list[str]) -> list[str]:
    counts = Counter(t for _, t in examples)
    labels = [str(lab).strip() for lab in given if str(lab).strip()] or sorted(counts)
    folded = Counter(lab.casefold() for lab in labels)
    clash = sorted(lab for lab in labels if folded[lab.casefold()] > 1)
    if clash:
        raise RowsError(f"labels differ only in upper/lower case: {', '.join(clash)}; use one spelling")
    unknown = sorted(set(counts) - set(labels))
    if unknown:
        raise RowsError(f"examples use labels that are not in the label list: {', '.join(unknown)}")
    thin = [f"{lab} ({counts.get(lab, 0)})" for lab in labels if counts.get(lab, 0) < MIN_PER_LABEL]
    if thin:
        raise TooFewRows(f"every label needs at least {MIN_PER_LABEL} examples to be trained and tested; too few: "
                         + ", ".join(thin))
    if len(labels) < 2:
        raise RowsError("classification needs at least two labels")
    return labels


def _instruction(template: Template, opts: BuildOptions, labels: list[str],
                 examples: list[tuple[str, str]]) -> str:
    if opts.instruction.strip():
        text = opts.instruction.strip()
        return text.replace("{labels}", ", ".join(labels)) if labels else text
    if template.id == "classification":
        return template.default_instruction.format(labels=", ".join(labels))
    if template.id == "extraction":
        if opts.schema is not None:
            spec = "JSON schema of the object:\n" + json.dumps(opts.schema, ensure_ascii=False)
        else:
            keys = Counter(k for _, t in examples for k in json.loads(t))
            spec = "Fields: " + ", ".join(k for k, _ in keys.most_common())
        return f"{template.default_instruction}\n{spec}"
    if template.id == "style":
        return opts.style_guide.strip()
    return ""


def _split(template: Template, examples: list[tuple[str, str]], n_heldout: int,
           seed: int) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    rng = random.Random(seed)
    if template.id != "classification":
        order = list(examples)
        rng.shuffle(order)
        return order[n_heldout:], order[:n_heldout]
    by_label: dict[str, list[tuple[str, str]]] = {}
    for ex in examples:
        by_label.setdefault(ex[1], []).append(ex)
    n = len(examples)
    quotas = {lab: n_heldout * len(items) / n for lab, items in sorted(by_label.items())}
    take = {lab: max(1, int(q)) for lab, q in quotas.items()}
    leftover = sorted(quotas, key=lambda lab: quotas[lab] - int(quotas[lab]), reverse=True)
    i = 0
    while sum(take.values()) < n_heldout and leftover:
        take[leftover[i % len(leftover)]] += 1
        i += 1
    train, held = [], []
    for lab, items in sorted(by_label.items()):
        items = list(items)
        rng.shuffle(items)
        held.extend(items[:take[lab]])
        train.extend(items[take[lab]:])
    rng.shuffle(train)
    rng.shuffle(held)
    return train, held


def _messages(instruction: str, prompt: str, answer: str | None = None) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": instruction}] if instruction else []
    msgs.append({"role": "user", "content": prompt})
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


def prepare(template_id: str, rows: list[Any], opts: BuildOptions) -> Prepared:
    """Validate ``rows`` for the template and return the SFT rows, the held-out quiz and the training plan."""
    template = get_template(template_id)
    if len(rows) > MAX_ROWS:
        raise RowsError(f"at most {MAX_ROWS} examples per use case (got {len(rows)})")
    if template.id == "extraction" and opts.schema is not None:
        schema_mod.check_schema(opts.schema)
    examples = _read_examples(template, rows, opts)
    examples, dupes, conflicts = _dedupe(examples)
    labels = _labels(examples, opts.labels) if template.id == "classification" else []
    n = len(examples)
    n_heldout = heldout_size(n)
    needed = template.needs.minimum + n_heldout
    if n < needed:
        raise TooFewRows(f"{template.title} needs at least {needed} distinct examples ({template.needs.minimum} to "
                         f"train + {n_heldout} held-out questions); got {n}"
                         + (f" after merging {dupes} duplicate(s)" if dupes else "")
                         + (f" and dropping {conflicts} conflicting row(s)" if conflicts else ""))
    instruction = _instruction(template, opts, labels, examples)
    train, held = _split(template, examples, n_heldout, opts.seed)
    warnings = []
    lo, hi = template.needs.recommended
    if len(train) < lo:
        warnings.append(f"{len(train)} training examples: enough to start, but gains were measured with {lo}-{hi}")
    if len(held) < SMALL_QUIZ:
        warnings.append(f"{len(held)} held-out questions: a paired test this small only shows large differences")
    if conflicts:
        warnings.append(f"{conflicts} rows dropped: the same input appeared with different answers")
    train_rows = [{"messages": _messages(instruction, p, a), "usecase": template.id} for p, a in train]
    cases = [{"name": f"{template.id}-{i:04d}", "category": f"usecase:{template.id}", "question": p,
              "correct_answer": a, "system_prompt": instruction} for i, (p, a) in enumerate(held, 1)]
    plan = recommend_settings(template, [r["messages"] for r in train_rows])
    return Prepared(template_id=template.id, instruction=instruction, train_rows=train_rows, heldout_cases=cases,
                    labels=labels, schema=opts.schema, style_guide=opts.style_guide.strip(),
                    teacher=opts.teacher.strip(), seed=opts.seed, examples=n, dropped_duplicates=dupes,
                    dropped_conflicts=conflicts, warnings=warnings, plan=plan)
