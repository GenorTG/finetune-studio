"""Preference-pair authoring (DPO) from a project's approved Q&A pairs.

Rows follow the TRL conversational preference contract that ``training.data.format_for_preference``
validates (this module never forks it)::

    {"prompt": [{"role": "user", "content": Q}],
     "chosen": [{"role": "assistant", "content": A_good}],
     "rejected": [{"role": "assistant", "content": A_bad}],
     "meta": {"source_id", "chunk_idx", "kind", "origin"}}

Two kinds, both anchored in an approved pair + its stored source chunk:

* ``hallucination`` — chosen = the approved, source-faithful answer; rejected = what a model says to the
  bare question WITHOUT the source. Kept only when the rejected answer is measurably less supported by the
  chunk than the chosen one (``qa_validate`` token overlap) — a rejected answer that is just as grounded
  teaches nothing and is dropped.
* ``abstain`` — a plausible question about the same subject that the corpus does NOT answer; chosen = a short
  honest refusal, rejected = a confident answer a model fabricated for it.

A rejected answer always comes from a model call (``generate``); with none, nothing is built — a rejected
answer is never invented by a template. Everything else here is pure and deterministic for a seed.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from finetune_studio.data import project_filesystem as pfs
from finetune_studio.data.prep.dataset_build import (
    PersistedDataset,
    register_dataset_file,
)
from finetune_studio.data.prep.export import deduplicate_qa_pairs
from finetune_studio.data.prep.qa_validate import (
    content_tokens,
    is_refusal_or_meta,
    normalize_question,
    token_overlap_ratio,
)
from finetune_studio.training.data import (
    clean_answer_for_training,
    format_for_preference,
    split_data,
)

KINDS = ("hallucination", "abstain")
DEFAULT_MAX_PAIRS = 100
MAX_PAIRS_LIMIT = 2000

# A rejected answer must be at least this much LESS supported by the source chunk than the chosen one
# (fraction of content tokens found in the chunk) — otherwise the pair does not separate good from bad.
MIN_SUPPORT_GAP = 0.15
# …and must not be a near-copy of the chosen answer (token Jaccard).
MAX_ANSWER_SIMILARITY = 0.8
_MIN_QUESTION_LEN = 8
_MAX_QUESTION_LEN = 300
# Model calls per wanted pair are bounded: at most this many candidates are tried per pair wanted.
ATTEMPTS_PER_PAIR = 3
_MAX_CONSECUTIVE_ERRORS = 3
# DPO learns length as a shortcut when chosen is systematically longer than rejected.
LENGTH_RATIO_WARN = 1.3
_VAL_FRACTION = 0.1

ABSTAIN_ANSWERS = (
    "That isn't covered in the provided documents.",
    "The provided documents don't say, so I can't answer that.",
    "I don't have that information in the documents I was given.",
    "That isn't something the provided documents cover.",
)

_QUESTION_PROMPT = (
    "Below is a passage from a document. Write ONE natural question about the same specific subject "
    "(same names, products or entities) that this passage does NOT answer — it should ask for a detail "
    "the passage never states, such as a price, date, number, location or person. "
    "Reply with only the question.\n\nPassage:\n{chunk}"
)
_VERIFY_PROMPT = (
    "Passage:\n{chunk}\n\nQuestion: {question}\n\n"
    "Does the passage contain the information needed to answer the question? Reply with only YES or NO."
)
_CONFIDENT_PROMPT = (
    "Answer the question in one or two confident sentences with specific details, "
    "as if you knew the answer for certain. Do not hedge and do not ask for more context.\n\n"
    "Question: {question}"
)
# Hedges that mean the model did not fabricate (extends qa_validate's refusal phrases).
_HEDGES = ("not sure", "don't know", "do not know", "no information", "isn't covered", "not covered",
           "unable to", "cannot answer", "can't answer", "no way to know", "not publicly")
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

GenerateFn = Callable[[str], str]
"""``generate(prompt) -> text`` — one plain user turn in, the model's reply out (blocking)."""
ProgressFn = Callable[[dict[str, Any]], None]


class PreferenceBuildError(Exception):
    """Base for honest, user-facing build failures."""


class NoApprovedPairs(PreferenceBuildError):
    """The project has no approved pair with a stored source chunk to anchor preference pairs on."""


class NoGenerator(PreferenceBuildError):
    """No model is available to write rejected answers (never fabricated by template)."""


class NoUsablePairs(PreferenceBuildError):
    """Every candidate was dropped by the quality gates; ``report`` says why."""

    def __init__(self, message: str, report: PreferenceReport) -> None:
        super().__init__(message)
        self.report = report


class GenerationFailed(PreferenceBuildError):
    """The model failed repeatedly; stop instead of shipping a partial, silently thinner dataset."""


@dataclass
class LengthStats:
    mean_chosen_chars: float = 0.0
    mean_rejected_chars: float = 0.0
    ratio: float = 0.0  # chosen / rejected
    warning: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"mean_chosen_chars": round(self.mean_chosen_chars, 1),
                "mean_rejected_chars": round(self.mean_rejected_chars, 1),
                "ratio": round(self.ratio, 2), "warning": self.warning}


@dataclass
class PreferenceReport:
    """What a build kept, dropped and why — shown by the API, CLI and UI."""

    by_kind: dict[str, int] = field(default_factory=dict)
    attempted: dict[str, int] = field(default_factory=dict)
    dropped: dict[str, Counter[str]] = field(default_factory=dict)
    length: LengthStats = field(default_factory=LengthStats)
    length_by_kind: dict[str, LengthStats] = field(default_factory=dict)
    candidates: int = 0  # approved pairs with a stored chunk
    skipped_no_chunk: int = 0
    deduped_prompts: int = 0
    split: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.by_kind.values())

    def as_dict(self) -> dict[str, Any]:
        return {"pairs": self.total, "by_kind": dict(self.by_kind), "attempted": dict(self.attempted),
                "dropped": {k: dict(v) for k, v in self.dropped.items()},
                "length": self.length.as_dict(),
                "length_by_kind": {k: v.as_dict() for k, v in self.length_by_kind.items()},
                "candidates": self.candidates, "skipped_no_chunk": self.skipped_no_chunk,
                "deduped_prompts": self.deduped_prompts, "split": dict(self.split)}


@dataclass(frozen=True)
class PreferenceResult:
    rows: list[dict[str, Any]]
    report: PreferenceReport


# ── pure helpers ────────────────────────────────────────────────────────

def _stable(seed: int, kind: str, item: dict[str, Any]) -> str:
    raw = f"{seed}|{kind}|{item.get('source_id', '')}|{item.get('chunk_idx', '')}|{item['question']}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _clean(text: str) -> str:
    return clean_answer_for_training(_THINK_BLOCK.sub("", text or "")).strip()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _support(text: str, chunk_tokens: set[str]) -> float:
    return token_overlap_ratio(content_tokens(text), chunk_tokens)


def _similarity(a: str, b: str) -> float:
    ta, tb = content_tokens(a), content_tokens(b)
    return len(ta & tb) / len(ta | tb) if ta | tb else 1.0


def _is_hedge(answer: str) -> bool:
    low = answer.lower()
    return is_refusal_or_meta(answer) or any(h in low for h in _HEDGES)


def _row(item: dict[str, Any], kind: str, question: str, chosen: str, rejected: str) -> dict[str, Any]:
    return {
        "prompt": [{"role": "user", "content": question}],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
        "meta": {"source_id": item.get("source_id", ""), "chunk_idx": item.get("chunk_idx", 0),
                 "kind": kind, "origin": "helper-model"},
    }


def hallucination_gate(chosen: str, rejected: str, chunk: str) -> str | None:
    """Reason a hallucination pair must be dropped, or ``None`` to keep it."""
    if not rejected:
        return "empty_rejected"
    if _norm(rejected) == _norm(chosen):
        return "identical"
    if _is_hedge(rejected):
        return "refusal_as_rejected"
    if _similarity(chosen, rejected) >= MAX_ANSWER_SIMILARITY:
        return "near_duplicate"
    chunk_tokens = content_tokens(chunk)
    if _support(chosen, chunk_tokens) - _support(rejected, chunk_tokens) < MIN_SUPPORT_GAP:
        return "not_less_grounded"
    return None


def _clean_question(raw: str) -> str:
    line = next((ln.strip() for ln in _clean(raw).splitlines() if ln.strip()), "")
    line = re.sub(r"^(?:[-*\d.)\s]+|question\s*:\s*)", "", line, flags=re.IGNORECASE).strip().strip('"\'`“”')
    return line


def abstain_question_gate(question: str, chunk: str, corpus_tokens: set[str],
                          known: set[str]) -> str | None:
    """Reason an out-of-scope question must be dropped, or ``None``.

    It must look like a question, share a subject with its own chunk, ask for at least one detail
    (content word) that appears NOWHERE in the corpus, and not repeat an existing question. (Topical
    overlap with the chunk cannot tell "same subject" from "answerable", so it is not the test.)
    """
    if not question.endswith("?") or not _MIN_QUESTION_LEN <= len(question) <= _MAX_QUESTION_LEN:
        return "malformed_question"
    q_tokens = content_tokens(question)
    if not q_tokens & content_tokens(chunk):
        return "off_subject"
    if normalize_question(question) in known:
        return "duplicate_question"
    if not q_tokens - corpus_tokens:
        return "no_novel_detail"
    return None


def length_stats(rows: list[dict[str, Any]]) -> LengthStats:
    """Mean chosen/rejected characters, their ratio, and a warning when chosen is systematically longer."""
    if not rows:
        return LengthStats()
    chosen = sum(len(r["chosen"][-1]["content"]) for r in rows) / len(rows)
    rejected = sum(len(r["rejected"][-1]["content"]) for r in rows) / len(rows)
    ratio = chosen / rejected if rejected else 0.0
    warning = None
    if ratio >= LENGTH_RATIO_WARN:
        warning = (f"Chosen answers average {ratio:.1f}× the length of rejected ones — DPO can learn "
                   "'longer is better' instead of 'faithful is better'. Consider rebuilding with fewer pairs "
                   "or a different seed, or review the pairs before training.")
    return LengthStats(chosen, rejected, ratio, warning)


def split_preference_rows(rows: list[dict[str, Any]], val_fraction: float = _VAL_FRACTION,
                          seed: int = 42) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic seeded (train, val) split — the same ``split_data`` the DPO engine applies.

    With the engine's defaults (``val_fraction=0.1``, seed 42) this is exactly the split training will use,
    so the preview shown to the user matches the held-out rows.
    """
    return split_data(rows, train_ratio=1.0 - val_fraction, seed=seed)


def _quotas(kinds: tuple[str, ...], max_pairs: int) -> dict[str, int]:
    base, extra = divmod(max_pairs, len(kinds))
    return {k: base + (1 if i < extra else 0) for i, k in enumerate(kinds)}


# ── the builder (pure given pairs + generate) ───────────────────────────

def build_preference_rows(items: list[dict[str, Any]], generate: GenerateFn, *,
                          kinds: tuple[str, ...] = KINDS, max_pairs: int = DEFAULT_MAX_PAIRS,
                          seed: int = 42, progress: ProgressFn | None = None) -> PreferenceResult:
    """Build gated preference rows from deduplicated approved pairs.

    ``items`` need ``question``, ``answer``, ``chunk_text`` (+ ``source_id``/``chunk_idx``). ``generate`` is the
    only source of rejected answers. Raises ``GenerationFailed`` after repeated model errors.
    """
    kinds = tuple(dict.fromkeys(kinds))
    unknown = [k for k in kinds if k not in KINDS]
    if unknown or not kinds:
        raise ValueError(f"kinds must be a non-empty subset of {', '.join(KINDS)}; got {list(kinds) or 'none'}")
    if max_pairs < 1:
        raise ValueError("max_pairs must be >= 1")

    report = PreferenceReport()
    candidates = [it for it in items if str(it.get("chunk_text") or "").strip()]
    report.candidates = len(candidates)
    report.skipped_no_chunk = len(items) - len(candidates)
    quotas = _quotas(kinds, max_pairs)
    planned = {k: min(len(candidates), quotas[k] * ATTEMPTS_PER_PAIR) for k in kinds}
    total_planned = sum(planned.values())
    corpus_tokens: set[str] = set().union(*(content_tokens(str(it["chunk_text"])) for it in candidates))
    known_questions = {normalize_question(str(it["question"])) for it in items}
    done_attempts = 0
    errors = 0

    def call(prompt: str) -> str | None:
        nonlocal errors
        try:
            out = generate(prompt)
        except Exception as exc:  # a model failure is counted, then surfaced after repeats
            errors += 1
            if errors >= _MAX_CONSECUTIVE_ERRORS:
                raise GenerationFailed(f"the model failed {errors} times in a row: {exc}") from exc
            return None
        errors = 0
        return _clean(str(out or ""))

    rows_by_kind: dict[str, list[dict[str, Any]]] = {k: [] for k in kinds}
    for kind in kinds:
        report.attempted[kind] = 0
        report.dropped[kind] = Counter()
        order = sorted(candidates, key=lambda it: _stable(seed, kind, it))
        for item in order[:planned[kind]]:
            if len(rows_by_kind[kind]) >= quotas[kind]:
                break
            report.attempted[kind] += 1
            done_attempts += 1
            row = _try_hallucination(item, call, report) if kind == "hallucination" else \
                _try_abstain(item, call, report, corpus_tokens, known_questions, seed)
            if row is not None:
                rows_by_kind[kind].append(row)
            if progress:
                progress({"kind": kind, "attempted": done_attempts, "planned": total_planned,
                          "kept": {k: len(v) for k, v in rows_by_kind.items()}})

    # Dedupe on prompt across kinds (first kind listed wins) so one question never appears twice.
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for kind in kinds:
        for row in rows_by_kind[kind]:
            key = normalize_question(row["prompt"][-1]["content"])
            if key in seen:
                report.deduped_prompts += 1
                continue
            seen.add(key)
            rows.append(row)
    rows = rows[:max_pairs]
    report.by_kind = dict(Counter(r["meta"]["kind"] for r in rows))
    report.length = length_stats(rows)
    report.length_by_kind = {k: length_stats([r for r in rows if r["meta"]["kind"] == k]) for k in report.by_kind}
    train, val = split_preference_rows(rows, seed=seed)
    report.split = {"train": len(train), "val": len(val)}
    return PreferenceResult(rows=rows, report=report)


def _try_hallucination(item: dict[str, Any], call: Callable[[str], str | None],
                       report: PreferenceReport) -> dict[str, Any] | None:
    drops = report.dropped["hallucination"]
    rejected = call(item["question"])
    if rejected is None:
        drops["generation_error"] += 1
        return None
    reason = hallucination_gate(item["answer"], rejected, str(item["chunk_text"]))
    if reason:
        drops[reason] += 1
        return None
    return _row(item, "hallucination", item["question"], item["answer"], rejected)


def _try_abstain(item: dict[str, Any], call: Callable[[str], str | None], report: PreferenceReport,
                 corpus_tokens: set[str], known: set[str], seed: int) -> dict[str, Any] | None:
    drops = report.dropped["abstain"]
    raw = call(_QUESTION_PROMPT.format(chunk=str(item["chunk_text"])[:3000]))
    if raw is None:
        drops["generation_error"] += 1
        return None
    question = _clean_question(raw)
    reason = abstain_question_gate(question, str(item["chunk_text"]), corpus_tokens, known)
    if reason:
        drops[reason] += 1
        return None
    # A refusal for a question the source DOES answer would teach over-refusal: ask the model to check.
    verdict = call(_VERIFY_PROMPT.format(chunk=str(item["chunk_text"])[:3000], question=question))
    if verdict is None:
        drops["generation_error"] += 1
        return None
    if not verdict.strip().lower().startswith("no"):
        drops["answerable_by_model_check"] += 1
        return None
    rejected = call(_CONFIDENT_PROMPT.format(question=question))
    if rejected is None:
        drops["generation_error"] += 1
        return None
    if not rejected:
        drops["empty_rejected"] += 1
        return None
    if _is_hedge(rejected):
        drops["refusal_as_rejected"] += 1
        return None
    known.add(normalize_question(question))
    pick = int(_stable(seed, "abstain-answer", {**item, "question": question})[:8], 16)
    return _row(item, "abstain", question, ABSTAIN_ANSWERS[pick % len(ABSTAIN_ANSWERS)], rejected)


# ── model access ────────────────────────────────────────────────────────

def helper_generate(*, max_tokens: int = 300, temperature: float = 0.7) -> GenerateFn | None:
    """``generate`` backed by the already-loaded data-prep helper, or ``None`` when it is not loaded."""
    from finetune_studio.data.prep.generator import resolve_generator
    chat = resolve_generator()
    if chat is None:
        return None

    def generate(prompt: str) -> str:
        return chat([{"role": "user", "content": prompt}], max_tokens=max_tokens,
                    temperature=temperature, top_p=0.9)

    return generate


@contextmanager
def helper_loaded_for_cli():
    """Yield a helper-backed ``generate``, loading the helper GGUF if no model is resident and unloading it after.

    CLI only: the server never loads models implicitly (the Pairs page loads the helper explicitly).
    """
    from finetune_studio.data.prep.generator import helper_resolution_error
    from finetune_studio.models.helper import (
        DEFAULT_HELPER_PROVIDER_ID,
        helper_missing_message,
        missing_gguf_for_provider,
    )
    from finetune_studio.models.llama_loader import unload_all_models
    from finetune_studio.models.manager import get_manager

    generate = helper_generate()
    loaded_here = False
    if generate is None:
        missing = missing_gguf_for_provider(DEFAULT_HELPER_PROVIDER_ID)
        if missing:
            raise NoGenerator(helper_missing_message(missing))
        get_manager().load(DEFAULT_HELPER_PROVIDER_ID)
        loaded_here = True
        generate = helper_generate()
    if generate is None:
        raise NoGenerator(helper_resolution_error())
    try:
        yield generate
    finally:
        if loaded_here:
            unload_all_models()


# ── project build: load → build → write → register ──────────────────────

@dataclass(frozen=True)
class BuiltPreference:
    persisted: PersistedDataset
    report: PreferenceReport


def build_preference_dataset(pid: str, kinds: tuple[str, ...] = KINDS, max_pairs: int = DEFAULT_MAX_PAIRS,
                             seed: int = 42, generate: GenerateFn | None = None,
                             progress: ProgressFn | None = None) -> BuiltPreference:
    """Author preference pairs for project ``pid`` and register ``<pid>-preference.jsonl``.

    Blocking (model calls): async callers run it with ``asyncio.to_thread``. Raises ``NoApprovedPairs``,
    ``NoGenerator``, ``NoUsablePairs``, ``GenerationFailed`` or ``ValueError`` (bad arguments).
    ``generate=None`` uses the loaded helper.
    """
    from finetune_studio import db
    items = deduplicate_qa_pairs(pfs.list_qa_pairs(pid, status="approved"))
    if not any(str(it.get("chunk_text") or "").strip() for it in items):
        raise NoApprovedPairs(
            "No approved Q&A pairs with a stored source passage to build preference pairs from. "
            "Review and approve pairs on the Pairs page first.")
    if generate is None:
        generate = helper_generate()
        if generate is None:
            from finetune_studio.data.prep.generator import helper_resolution_error
            raise NoGenerator(helper_resolution_error())
    result = build_preference_rows(items, generate, kinds=kinds, max_pairs=max_pairs, seed=seed,
                                   progress=progress)
    if not result.rows:
        raise NoUsablePairs("The model's answers did not produce a single usable preference pair "
                            "(every candidate failed a quality gate).", result.report)
    format_for_preference(result.rows)  # the exact validator the Training route applies — never ship a file it rejects
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in result.rows)
    project_name = (db.get_project(pid) or {}).get("name") or pid
    persisted = register_dataset_file(pid, f"{pid}-preference.jsonl", body,
                                      f"{project_name} · preference · {{rows}} pairs",
                                      source="data-prep-preference")
    return BuiltPreference(persisted=persisted, report=result.report)


__all__ = [
    "ABSTAIN_ANSWERS", "KINDS", "BuiltPreference", "GenerateFn", "GenerationFailed", "LengthStats",
    "NoApprovedPairs", "NoGenerator", "NoUsablePairs", "PreferenceBuildError", "PreferenceReport",
    "PreferenceResult", "abstain_question_gate", "build_preference_dataset", "build_preference_rows",
    "hallucination_gate", "helper_generate", "helper_loaded_for_cli", "length_stats", "split_preference_rows",
]
