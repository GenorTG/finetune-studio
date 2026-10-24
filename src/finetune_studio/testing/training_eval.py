"""Evaluate a project's training dataset for memorization / leakage checks.

Training-set evaluation is intentionally labeled as leakage-aware: high scores
indicate recall of seen examples, not held-out generalization.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from finetune_studio.db import datasets as datasets_db
from finetune_studio.testing.generate_suite import (
    _analyze_difficulty,
    _categorize,
    _slugify,
)
from finetune_studio.testing.suite import BenchmarkCase
from finetune_studio.training.data import clean_answer_for_training

EvalKind = Literal["training_leakage", "heldout"]

LEAKAGE_WARNING = (
    "Training-data evaluation: cases are drawn from the project's approved "
    "training dataset. High pass rates measure in-distribution recall / "
    "memorization risk, not generalization to unseen data."
)


@dataclass(frozen=True)
class TrainingEvalMeta:
    """Metadata attached to training-data evaluation responses."""

    eval_kind: EvalKind
    leakage_warning: str
    dataset_id: str
    dataset_name: str
    dataset_path: str
    dataset_source: str
    case_count: int
    skipped: int
    max_cases: int | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_qa(example: dict[str, Any]) -> tuple[str, str] | None:
    """Pull (question, answer) from ShareGPT conversations or messages formats."""
    conversations = example.get("conversations")
    if isinstance(conversations, list):
        user = next(
            (
                turn
                for turn in conversations
                if isinstance(turn, dict)
                and str(turn.get("from") or turn.get("role") or "").lower()
                in {"human", "user"}
            ),
            None,
        )
        assistant = next(
            (
                turn
                for turn in conversations
                if isinstance(turn, dict)
                and str(turn.get("from") or turn.get("role") or "").lower()
                in {"gpt", "assistant"}
            ),
            None,
        )
        q = str((user or {}).get("value") or (user or {}).get("content") or "").strip()
        a = str((assistant or {}).get("value") or (assistant or {}).get("content") or "").strip()
        if q and a:
            return q, clean_answer_for_training(a)

    messages = example.get("messages")
    if isinstance(messages, list):
        user = next(
            (
                m
                for m in messages
                if isinstance(m, dict) and m.get("role") in {"user", "human"}
            ),
            None,
        )
        asst = next(
            (
                m
                for m in messages
                if isinstance(m, dict) and m.get("role") in {"assistant", "gpt"}
            ),
            None,
        )
        if user and asst:
            q = str(user.get("content") or "").strip()
            a = str(asst.get("content") or "").strip()
            if q and a:
                return q, clean_answer_for_training(a)

    # Flat Q/A keys used by some exporters
    q = str(example.get("question") or example.get("prompt") or "").strip()
    a = str(example.get("answer") or example.get("response") or example.get("completion") or "").strip()
    if q and a:
        return q, clean_answer_for_training(a)
    return None


def cases_from_training_jsonl(
    data_path: str,
    *,
    max_cases: int | None = 200,
) -> tuple[list[BenchmarkCase], int]:
    """Convert a training JSONL file into BenchmarkCase rows.

    ``max_cases=None`` means full coverage (every row). Returns
    ``(cases, skipped)``.
    """
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"training data not found: {data_path}")

    cases: list[BenchmarkCase] = []
    skipped = 0
    seen_names: dict[str, int] = {}

    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if max_cases is not None and len(cases) >= max_cases:
                break
            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(example, dict):
                skipped += 1
                continue
            case = _case_from_example(example, seen_names, len(cases))
            if case is None:
                skipped += 1
                continue
            cases.append(case)

    return cases, skipped


def _case_from_example(example: dict[str, Any], seen_names: dict[str, int], index: int) -> BenchmarkCase | None:
    """One training row as a quiz case (unique name per suite), None when it holds no Q/A."""
    qa = _extract_qa(example)
    if qa is None:
        return None
    question, answer = qa
    category = _categorize(question, answer)
    _diff, judge_hint = _analyze_difficulty(question, answer)
    base = _slugify(question[:60]) or f"case_{index}"
    n = seen_names.get(base, 0)
    seen_names[base] = n + 1
    return BenchmarkCase(
        name=base if n == 0 else f"{base}_{n + 1}",
        question=question,
        correct_answer=answer,
        category=category,
        context=judge_hint,
        source_id=str(example.get("source_id") or ""),
        chunk_idx=int(example.get("chunk_idx") or 0),
    )


def resolve_project_dataset(
    project_id: str,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    """Pick a project dataset (explicit id or most recent with qa_count > 0)."""
    if dataset_id:
        ds = datasets_db.get_dataset(dataset_id)
        if not ds or ds.get("project_id") != project_id:
            raise LookupError(f"dataset not found for project: {dataset_id}")
        return ds

    rows = datasets_db.list_datasets(project_id)
    for ds in rows:
        if int(ds.get("qa_count") or 0) > 0 and Path(str(ds.get("data_path") or "")).is_file():
            return ds
    for ds in rows:
        path = str(ds.get("data_path") or "")
        if path and Path(path).is_file():
            return ds
    raise LookupError("no project training dataset registered — export approved data first")


def build_training_eval(
    project_id: str,
    *,
    dataset_id: str | None = None,
    max_cases: int = 200,
) -> tuple[list[BenchmarkCase], TrainingEvalMeta]:
    """Resolve dataset + build leakage-eval cases and metadata."""
    ds = resolve_project_dataset(project_id, dataset_id)
    path = str(ds["data_path"])
    cases, skipped = cases_from_training_jsonl(path, max_cases=max_cases)
    if not cases:
        raise ValueError(
            f"no Q&A pairs extracted from dataset {ds.get('name')!r} "
            f"(skipped={skipped})"
        )
    meta = TrainingEvalMeta(
        eval_kind="training_leakage",
        leakage_warning=LEAKAGE_WARNING,
        dataset_id=str(ds["id"]),
        dataset_name=str(ds.get("name") or ""),
        dataset_path=path,
        dataset_source=str(ds.get("source") or ""),
        case_count=len(cases),
        skipped=skipped,
        max_cases=max_cases,
    )
    return cases, meta


def build_heldout_eval(
    project_id: str,
    *,
    dataset_id: str | None = None,
    max_cases: int = 200,
) -> tuple[list[BenchmarkCase], TrainingEvalMeta]:
    """Build the deterministic 10% validation slice used by the trainer.

    This is the meaningful quality score. The full approved dataset remains
    available as the separately labelled leakage check, but it must not be
    presented as generalization evidence. The slice is the trainer's own
    (``training.data.sft_validation_rows``): rows of one fact stay together, so a
    held-out question is never answered by a trained rewording of itself.
    """
    from finetune_studio.training.data import sft_validation_rows

    ds = resolve_project_dataset(project_id, dataset_id)
    rows: list[dict[str, Any]] = []
    skipped = 0
    with open(str(ds["data_path"]), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                skipped += 1
    seen: dict[str, int] = {}
    cases = []
    for row in sft_validation_rows(rows):
        case = _case_from_example(row, seen, len(cases))
        if case is None:
            skipped += 1
        elif len(cases) < max_cases:
            cases.append(case)
    if not cases:
        raise ValueError("dataset has no held-out validation examples")
    meta = TrainingEvalMeta(
        eval_kind="heldout",
        leakage_warning=(
            "Held-out validation slice: deterministic 10% excluded from the "
            "training split. This is the relevant in-domain quality score."
        ),
        dataset_id=str(ds["id"]),
        dataset_name=str(ds.get("name") or ""),
        dataset_path=str(ds["data_path"]),
        dataset_source=str(ds.get("source") or ""),
        case_count=len(cases),
        skipped=skipped,
        max_cases=max_cases,
    )
    return cases, meta


def suite_label_for_training_eval(meta: TrainingEvalMeta) -> str:
    """Human-readable suite name for DB / UI."""
    return f"{meta.eval_kind}:{meta.dataset_name or meta.dataset_id}"
