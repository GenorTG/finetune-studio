"""Official HuggingFace industry benchmarks: MMLU, GSM8K, HellaSwag.

Loads official test/validation splits, formats prompts, and scores with
strict MCQ / normalized GSM8K finals (no substring credit). Unit tests must
inject a fake dataset loader — never download HF corpora in CI.
"""

from __future__ import annotations

import ast
import logging
import os
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

from finetune_studio.testing.strict_scoring import (
    score_multiple_choice,
    score_numeric,
)
from finetune_studio.testing.suite import BenchmarkCase

_log = logging.getLogger(__name__)

RealFamily = Literal["mmlu", "gsm8k", "hellaswag"]
SampleOrder = Literal["dataset", "seeded_shuffle"]

REAL_SUITE_SCHEME = "real://"
DEFAULT_SAMPLE_LIMIT = 50
MAX_BOUNDED_SAMPLES = 500
DEFAULT_SEED = 0

# Official split sizes (HuggingFace; verified against common HF revisions).
OFFICIAL_CASE_COUNTS: dict[RealFamily, int] = {
    "mmlu": 14042,  # cais/mmlu · all · test
    "gsm8k": 1319,  # openai/gsm8k · main · test
    "hellaswag": 10042,  # Rowan/hellaswag · validation
}

MCQ_PROMPT_PREFIX = (
    "Answer the following multiple choice question. "
    "Reply with ONLY the letter (A, B, C, or D)."
)
GSM8K_PROMPT_PREFIX = (
    'Solve this math problem step by step. '
    'Put the final numeric answer after "####".'
)

DatasetLoader = Callable[..., Any]


class InferenceLike(Protocol):
    """Minimal generate surface used by real benchmark evaluation."""

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str | dict[str, Any]:
        ...


@dataclass(frozen=True)
class RealBenchmarkSpec:
    """Static identity of an official industry benchmark suite."""

    family: RealFamily
    name: str
    title: str
    description: str
    dataset_id: str
    dataset_config: str | None
    split: str
    official_case_count: int
    prompt_protocol: str
    scoring_method: str
    source_url: str = ""
    revision: str | None = None


@dataclass
class BenchmarkMetadata:
    """Explicit provenance + run parameters for a real benchmark execution."""

    family: RealFamily
    dataset_id: str
    dataset_config: str | None
    split: str
    source_url: str
    revision: str | None
    official_case_count: int
    sample_count: int
    seed: int
    order: SampleOrder
    prompt_protocol: str
    scoring_method: str
    is_real_benchmark: bool = True
    full_run: bool = False
    loaded_split_size: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize for API / scores persistence."""
        return asdict(self)


@dataclass
class CaseEvalResult:
    """One scored item from a real benchmark run."""

    case_id: str
    question: str
    prediction: str
    expected: str
    correct: bool
    validity: str
    scoring_method: str
    reasoning: str
    time_ms: float = 0.0
    category: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RealBenchmarkResult:
    """Aggregate result for one real suite evaluation."""

    family: RealFamily
    metadata: BenchmarkMetadata
    total: int
    correct: int
    accuracy: float
    results: list[CaseEvalResult]
    subject_scores: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly payload (also safe for the inference endpoint)."""
        return {
            "benchmark": self.family,
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "pass_rate": self.accuracy,
            "subjects": self.subject_scores,
            "metadata": self.metadata.as_dict(),
            "is_real_benchmark": True,
            "results": [
                {
                    "id": r.case_id,
                    "question": r.question[:200],
                    "prediction": r.prediction[:500],
                    "expected": r.expected,
                    "correct": r.correct,
                    "validity": r.validity,
                    "scoring_method": r.scoring_method,
                    "reasoning": r.reasoning,
                    "time_ms": r.time_ms,
                    "category": r.category,
                    **({"subject": r.metadata["subject"]} if "subject" in r.metadata else {}),
                }
                for r in self.results
            ],
        }


REAL_SPECS: dict[RealFamily, RealBenchmarkSpec] = {
    "mmlu": RealBenchmarkSpec(
        family="mmlu",
        name="mmlu_real",
        title="MMLU (official)",
        description=(
            "Official cais/mmlu test split (all subjects). Strict MCQ letter "
            "scoring — industry benchmark, requires HuggingFace datasets cache."
        ),
        dataset_id="cais/mmlu",
        dataset_config="all",
        split="test",
        official_case_count=OFFICIAL_CASE_COUNTS["mmlu"],
        prompt_protocol="zero_shot_mcq_letter_only",
        scoring_method="strict_mcq",
        source_url="https://huggingface.co/datasets/cais/mmlu",
        revision=None,
    ),
    "gsm8k": RealBenchmarkSpec(
        family="gsm8k",
        name="gsm8k_real",
        title="GSM8K (official)",
        description=(
            "Official openai/gsm8k main test split. Exact normalized #### "
            "final-answer scoring — industry benchmark, requires HF cache."
        ),
        dataset_id="openai/gsm8k",
        dataset_config="main",
        split="test",
        official_case_count=OFFICIAL_CASE_COUNTS["gsm8k"],
        prompt_protocol="zero_shot_gsm8k_hash_final",
        scoring_method="strict_numeric",
        source_url="https://huggingface.co/datasets/openai/gsm8k",
        revision=None,
    ),
    "hellaswag": RealBenchmarkSpec(
        family="hellaswag",
        name="hellaswag_real",
        title="HellaSwag (official)",
        description=(
            "Official Rowan/hellaswag validation split (canonical eval split). "
            "Strict MCQ letter scoring — industry benchmark, requires HF cache."
        ),
        dataset_id="Rowan/hellaswag",
        dataset_config=None,
        split="validation",
        official_case_count=OFFICIAL_CASE_COUNTS["hellaswag"],
        prompt_protocol="zero_shot_mcq_letter_only",
        scoring_method="strict_mcq",
        source_url="https://huggingface.co/datasets/Rowan/hellaswag",
        revision=None,
    ),
}


def real_suite_path(family: RealFamily | str) -> str:
    """Virtual suite path used by discovery / WebUI selection."""
    return f"{REAL_SUITE_SCHEME}{family}"


def parse_real_suite_path(path: str) -> RealFamily | None:
    """Return family when ``path`` is a real:// suite, else None."""
    raw = (path or "").strip()
    if not raw.startswith(REAL_SUITE_SCHEME):
        return None
    family = raw[len(REAL_SUITE_SCHEME) :].strip().lower()
    if family in REAL_SPECS:
        return family  # type: ignore[return-value]
    return None


def is_real_suite_path(path: str) -> bool:
    """True when path references a built-in real HF suite."""
    return parse_real_suite_path(path) is not None


def list_real_families() -> list[RealFamily]:
    """Ordered list of supported real benchmark families."""
    return ["mmlu", "gsm8k", "hellaswag"]


def _coerce_response(raw: str | dict[str, Any]) -> str:
    if isinstance(raw, dict):
        return str(raw.get("response", raw.get("text", "")))
    return str(raw or "")


def _parse_listish(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    return []


def format_mmlu_prompt(question: str, choices: Sequence[str]) -> str:
    """Zero-shot MCQ prompt; expects a single letter reply."""
    lines = [
        MCQ_PROMPT_PREFIX,
        "",
        f"Question: {question}",
        f"A) {choices[0]}",
        f"B) {choices[1]}",
        f"C) {choices[2]}",
        f"D) {choices[3]}",
        "",
        "Answer:",
    ]
    return "\n".join(lines)


def format_hellaswag_prompt(ctx: str, endings: Sequence[str]) -> str:
    """Zero-shot completion MCQ prompt."""
    lines = [
        MCQ_PROMPT_PREFIX,
        "",
        f"Context: {ctx}",
        f"A) {endings[0]}",
        f"B) {endings[1]}",
        f"C) {endings[2]}",
        f"D) {endings[3]}",
        "",
        "What happens next?:",
    ]
    return "\n".join(lines)


def format_gsm8k_prompt(question: str) -> str:
    """Zero-shot GSM8K prompt with #### final-answer convention."""
    return f"{GSM8K_PROMPT_PREFIX}\n\nQuestion: {question}\n\nSolution:"


def extract_gsm8k_gold(answer_field: str) -> str:
    """Official GSM8K gold is the token after the last ####."""
    text = answer_field or ""
    if "####" in text:
        return text.split("####")[-1].strip().replace(",", "")
    return text.strip().replace(",", "")


def _default_load_dataset(
    path: str,
    name: str | None = None,
    *,
    split: str,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> Any:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split, "cache_dir": cache_dir}
    if revision:
        kwargs["revision"] = revision
    if name:
        return load_dataset(path, name, **kwargs)
    return load_dataset(path, **kwargs)


def resolve_sample_count(
    *,
    split_size: int,
    num_samples: int | None,
    full_run: bool,
) -> int:
    """Clamp requested sample count; ``full_run`` uses the entire split."""
    if full_run or num_samples is None:
        return split_size
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    capped = min(int(num_samples), MAX_BOUNDED_SAMPLES, split_size)
    return capped


def select_indices(
    split_size: int,
    sample_count: int,
    *,
    seed: int = DEFAULT_SEED,
    order: SampleOrder = "dataset",
) -> list[int]:
    """Deterministic index selection over ``[0, split_size)``."""
    if sample_count > split_size:
        raise ValueError("sample_count exceeds split_size")
    if order == "dataset":
        return list(range(sample_count))
    rng = random.Random(seed)
    indices = list(range(split_size))
    rng.shuffle(indices)
    return indices[:sample_count]


def _row_get(row: Mapping[str, Any] | Any, key: str) -> Any:
    if isinstance(row, Mapping):
        return row[key]
    return row[key]


def mmlu_row_to_case(row: Mapping[str, Any] | Any, index: int) -> BenchmarkCase:
    """Convert one cais/mmlu row into a BenchmarkCase."""
    question = str(_row_get(row, "question"))
    choices = _parse_listish(_row_get(row, "choices"))
    if len(choices) < 4:
        raise ValueError(f"mmlu row {index}: expected 4 choices, got {len(choices)}")
    answer_idx = int(_row_get(row, "answer"))
    letter = "ABCD"[answer_idx]
    try:
        subject = str(_row_get(row, "subject"))
    except (KeyError, TypeError):
        subject = "mmlu"
    prompt = format_mmlu_prompt(question, [str(c) for c in choices[:4]])
    return BenchmarkCase(
        name=f"mmlu_{index:05d}_{subject}",
        question=prompt,
        correct_answer=letter,
        category=subject,
    )


def hellaswag_row_to_case(row: Mapping[str, Any] | Any, index: int) -> BenchmarkCase:
    """Convert one Rowan/hellaswag row into a BenchmarkCase."""
    ctx = str(_row_get(row, "ctx"))
    endings = _parse_listish(_row_get(row, "endings"))
    if len(endings) < 4:
        raise ValueError(f"hellaswag row {index}: expected 4 endings")
    gold_idx = int(_row_get(row, "label"))
    letter = "ABCD"[gold_idx]
    try:
        activity = str(_row_get(row, "activity_label"))
    except (KeyError, TypeError):
        activity = "hellaswag"
    prompt = format_hellaswag_prompt(ctx, [str(e) for e in endings[:4]])
    return BenchmarkCase(
        name=f"hellaswag_{index:05d}",
        question=prompt,
        correct_answer=letter,
        category=activity,
    )


def gsm8k_row_to_case(row: Mapping[str, Any] | Any, index: int) -> BenchmarkCase:
    """Convert one openai/gsm8k row into a BenchmarkCase."""
    question = str(_row_get(row, "question"))
    gold = extract_gsm8k_gold(str(_row_get(row, "answer")))
    prompt = format_gsm8k_prompt(question)
    return BenchmarkCase(
        name=f"gsm8k_{index:05d}",
        question=prompt,
        correct_answer=gold,
        category="math",
    )


_ROW_CONVERTERS: dict[RealFamily, Callable[[Any, int], BenchmarkCase]] = {
    "mmlu": mmlu_row_to_case,
    "hellaswag": hellaswag_row_to_case,
    "gsm8k": gsm8k_row_to_case,
}


def build_metadata(
    family: RealFamily,
    *,
    sample_count: int,
    seed: int = DEFAULT_SEED,
    order: SampleOrder = "dataset",
    full_run: bool = False,
    loaded_split_size: int | None = None,
) -> BenchmarkMetadata:
    """Assemble explicit metadata for a real suite run."""
    spec = REAL_SPECS[family]
    return BenchmarkMetadata(
        family=family,
        dataset_id=spec.dataset_id,
        dataset_config=spec.dataset_config,
        split=spec.split,
        source_url=spec.source_url,
        revision=spec.revision,
        official_case_count=spec.official_case_count,
        sample_count=sample_count,
        seed=seed,
        order=order,
        prompt_protocol=spec.prompt_protocol,
        scoring_method=spec.scoring_method,
        is_real_benchmark=True,
        full_run=full_run,
        loaded_split_size=loaded_split_size,
    )


class RealBenchmarkSuite:
    """Load and evaluate official MMLU / GSM8K / HellaSwag splits."""

    def __init__(
        self,
        cache_dir: str = "data/benchmarks/hf_cache",
        *,
        dataset_loader: DatasetLoader | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self._dataset_loader = dataset_loader or _default_load_dataset
        os.makedirs(cache_dir, exist_ok=True)

    def list_available(self) -> dict[str, str]:
        """Human-readable catalog of supported real suites."""
        return {
            family: f"{spec.title} — {spec.dataset_id} {spec.split} "
            f"({spec.official_case_count} official cases)"
            for family, spec in REAL_SPECS.items()
        }

    def load_split(self, family: RealFamily) -> Any:
        """Load the official split (or inject via ``dataset_loader``)."""
        spec = REAL_SPECS[family]
        _log.info(
            "Loading real benchmark %s (%s config=%s split=%s)",
            family,
            spec.dataset_id,
            spec.dataset_config,
            spec.split,
        )
        return self._dataset_loader(
            spec.dataset_id,
            spec.dataset_config,
            split=spec.split,
            cache_dir=self.cache_dir,
            revision=spec.revision,
        )

    def load_cases(
        self,
        family: RealFamily,
        *,
        num_samples: int | None = DEFAULT_SAMPLE_LIMIT,
        full_run: bool = False,
        seed: int = DEFAULT_SEED,
        order: SampleOrder = "dataset",
    ) -> tuple[list[BenchmarkCase], BenchmarkMetadata]:
        """Materialize BenchmarkCases from the official split."""
        ds = self.load_split(family)
        split_size = len(ds)
        sample_count = resolve_sample_count(
            split_size=split_size,
            num_samples=num_samples,
            full_run=full_run,
        )
        indices = select_indices(
            split_size, sample_count, seed=seed, order=order
        )
        converter = _ROW_CONVERTERS[family]
        cases: list[BenchmarkCase] = []
        for idx in indices:
            cases.append(converter(ds[idx], idx))
        meta = build_metadata(
            family,
            sample_count=sample_count,
            seed=seed,
            order=order,
            full_run=full_run or num_samples is None,
            loaded_split_size=split_size,
        )
        return cases, meta

    def evaluate(
        self,
        family: RealFamily,
        inference_engine: InferenceLike,
        *,
        num_samples: int | None = DEFAULT_SAMPLE_LIMIT,
        full_run: bool = False,
        seed: int = DEFAULT_SEED,
        order: SampleOrder = "dataset",
        max_tokens: int | None = None,
    ) -> RealBenchmarkResult:
        """Run inference + strict scoring for one real suite."""
        import time

        cases, meta = self.load_cases(
            family,
            num_samples=num_samples,
            full_run=full_run,
            seed=seed,
            order=order,
        )
        spec = REAL_SPECS[family]
        if max_tokens is None:
            max_tokens = 256 if family == "gsm8k" else 16

        results: list[CaseEvalResult] = []
        subject_hits: dict[str, list[bool]] = {}
        correct = 0

        for case in cases:
            t0 = time.time()
            raw = inference_engine.generate(
                [{"role": "user", "content": case.question}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            pred = _coerce_response(raw)
            elapsed = (time.time() - t0) * 1000

            if family == "gsm8k":
                scored = score_numeric(
                    correct_answer=case.correct_answer,
                    model_answer=pred,
                )
            else:
                scored = score_multiple_choice(
                    correct_answer=case.correct_answer,
                    model_answer=pred,
                )

            is_correct = scored.verdict == "pass"
            if is_correct:
                correct += 1
            subject_hits.setdefault(case.category, []).append(is_correct)
            results.append(
                CaseEvalResult(
                    case_id=case.name,
                    question=case.question,
                    prediction=pred,
                    expected=case.correct_answer,
                    correct=is_correct,
                    validity=scored.validity,
                    scoring_method=scored.scoring_method or spec.scoring_method,
                    reasoning=scored.reasoning,
                    time_ms=round(elapsed, 1),
                    category=case.category,
                    metadata={"subject": case.category} if family == "mmlu" else {},
                )
            )

        total = len(results)
        subject_scores = {
            sub: round(sum(1 for x in hits if x) / max(len(hits), 1) * 100, 1)
            for sub, hits in subject_hits.items()
        }
        return RealBenchmarkResult(
            family=family,
            metadata=meta,
            total=total,
            correct=correct,
            accuracy=round(correct / max(total, 1) * 100, 1),
            results=results,
            subject_scores=subject_scores if family == "mmlu" else {},
        )

    def evaluate_mmlu(
        self,
        inference_engine: InferenceLike,
        num_samples: int | None = DEFAULT_SAMPLE_LIMIT,
        *,
        full_run: bool = False,
        seed: int = DEFAULT_SEED,
        order: SampleOrder = "dataset",
        max_tokens: int = 16,
        subjects: list[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate MMLU; optional subject filter applied after load."""
        if subjects:
            cases, meta = self.load_cases(
                "mmlu",
                num_samples=None if full_run else num_samples,
                full_run=full_run,
                seed=seed,
                order=order,
            )
            wanted = set(subjects)
            cases = [c for c in cases if c.category in wanted]
            meta.sample_count = len(cases)
            # Re-run via internal path with prebuilt cases
            return self._evaluate_cases(
                "mmlu", inference_engine, cases, meta, max_tokens=max_tokens
            ).as_dict()
        return self.evaluate(
            "mmlu",
            inference_engine,
            num_samples=num_samples,
            full_run=full_run,
            seed=seed,
            order=order,
            max_tokens=max_tokens,
        ).as_dict()

    def evaluate_gsm8k(
        self,
        inference_engine: InferenceLike,
        num_samples: int | None = DEFAULT_SAMPLE_LIMIT,
        *,
        full_run: bool = False,
        seed: int = DEFAULT_SEED,
        order: SampleOrder = "dataset",
        max_tokens: int = 256,
    ) -> dict[str, Any]:
        """Evaluate GSM8K with strict numeric scoring."""
        return self.evaluate(
            "gsm8k",
            inference_engine,
            num_samples=num_samples,
            full_run=full_run,
            seed=seed,
            order=order,
            max_tokens=max_tokens,
        ).as_dict()

    def evaluate_hellaswag(
        self,
        inference_engine: InferenceLike,
        num_samples: int | None = DEFAULT_SAMPLE_LIMIT,
        *,
        full_run: bool = False,
        seed: int = DEFAULT_SEED,
        order: SampleOrder = "dataset",
        max_tokens: int = 16,
    ) -> dict[str, Any]:
        """Evaluate HellaSwag with strict MCQ scoring."""
        return self.evaluate(
            "hellaswag",
            inference_engine,
            num_samples=num_samples,
            full_run=full_run,
            seed=seed,
            order=order,
            max_tokens=max_tokens,
        ).as_dict()

    def _evaluate_cases(
        self,
        family: RealFamily,
        inference_engine: InferenceLike,
        cases: list[BenchmarkCase],
        meta: BenchmarkMetadata,
        *,
        max_tokens: int,
    ) -> RealBenchmarkResult:
        import time

        spec = REAL_SPECS[family]
        results: list[CaseEvalResult] = []
        subject_hits: dict[str, list[bool]] = {}
        correct = 0
        for case in cases:
            t0 = time.time()
            raw = inference_engine.generate(
                [{"role": "user", "content": case.question}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            pred = _coerce_response(raw)
            elapsed = (time.time() - t0) * 1000
            if family == "gsm8k":
                scored = score_numeric(
                    correct_answer=case.correct_answer, model_answer=pred
                )
            else:
                scored = score_multiple_choice(
                    correct_answer=case.correct_answer, model_answer=pred
                )
            is_correct = scored.verdict == "pass"
            if is_correct:
                correct += 1
            subject_hits.setdefault(case.category, []).append(is_correct)
            results.append(
                CaseEvalResult(
                    case_id=case.name,
                    question=case.question,
                    prediction=pred,
                    expected=case.correct_answer,
                    correct=is_correct,
                    validity=scored.validity,
                    scoring_method=scored.scoring_method or spec.scoring_method,
                    reasoning=scored.reasoning,
                    time_ms=round(elapsed, 1),
                    category=case.category,
                    metadata={"subject": case.category} if family == "mmlu" else {},
                )
            )
        total = len(results)
        subject_scores = {
            sub: round(sum(1 for x in hits if x) / max(len(hits), 1) * 100, 1)
            for sub, hits in subject_hits.items()
        }
        return RealBenchmarkResult(
            family=family,
            metadata=meta,
            total=total,
            correct=correct,
            accuracy=round(correct / max(total, 1) * 100, 1),
            results=results,
            subject_scores=subject_scores if family == "mmlu" else {},
        )

    def run_all(
        self,
        inference_engine: InferenceLike,
        num_samples: int | None = DEFAULT_SAMPLE_LIMIT,
        benchmarks: list[str] | None = None,
        *,
        full_run: bool = False,
        seed: int = DEFAULT_SEED,
        order: SampleOrder = "dataset",
    ) -> dict[str, Any]:
        """Run one or more real suites; always returns nested summary (not scalars)."""
        if benchmarks is None:
            benchmarks = list(list_real_families())

        all_results: dict[str, Any] = {}
        for name in benchmarks:
            family = name.strip().lower()
            if family not in REAL_SPECS:
                all_results[name] = {
                    "error": f"unsupported real benchmark: {name}",
                }
                continue
            try:
                result = self.evaluate(
                    family,  # type: ignore[arg-type]
                    inference_engine,
                    num_samples=num_samples,
                    full_run=full_run,
                    seed=seed,
                    order=order,
                )
                all_results[family] = result.as_dict()
            except Exception as exc:  # noqa: BLE001
                _log.exception("real benchmark %s failed", family)
                all_results[family] = {"error": str(exc)}

        total_correct = sum(
            int(r["correct"])
            for r in all_results.values()
            if isinstance(r, dict) and "correct" in r
        )
        total_questions = sum(
            int(r["total"])
            for r in all_results.values()
            if isinstance(r, dict) and "total" in r
        )
        return {
            "benchmarks": all_results,
            "summary": {
                "total_correct": total_correct,
                "total_questions": total_questions,
                "overall_accuracy": round(
                    total_correct / max(total_questions, 1) * 100, 1
                ),
                "is_real_benchmark": True,
            },
        }


def overall_accuracy_from_run_all(payload: Mapping[str, Any]) -> float:
    """Extract a scalar overall accuracy from ``run_all`` output.

    The broken inference endpoint used ``sum(results.values())`` on the nested
    dict; callers must use this helper (or ``summary.overall_accuracy``).
    """
    summary = payload.get("summary")
    if isinstance(summary, Mapping):
        val = summary.get("overall_accuracy")
        if isinstance(val, (int, float)):
            return float(val)
    benches = payload.get("benchmarks")
    if not isinstance(benches, Mapping):
        return 0.0
    correct = 0
    total = 0
    for item in benches.values():
        if not isinstance(item, Mapping):
            continue
        if "correct" in item and "total" in item:
            correct += int(item["correct"])
            total += int(item["total"])
    return round(correct / max(total, 1) * 100, 1)
