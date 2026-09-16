"""Strict deterministic validation for auto-generated Q&A pairs.

Runs AFTER model-output parsing (parse_qa_json / coerce_pairs fallbacks).
Does not alter parsing — only accepts or rejects normalised {"q","a"} pairs
before they are written to disk.

Rejection reasons (stable string keys for counters):
  empty_question, empty_answer, malformed_question, malformed_answer,
  duplicate_question, unanswerable_from_chunk, ungrounded_answer,
  refusal_or_meta
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# Content-word tokenization (letters/digits; Unicode-aware via \w).
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "once", "here", "there", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "don", "should", "now", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "of",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "as", "it", "its", "how", "why", "where",
})

# Refusal / meta / non-passage answers (substring match on lowercased answer).
_REFUSAL_PHRASES: tuple[str, ...] = (
    "as an ai",
    "as a language model",
    "as an assistant",
    "i don't have",
    "i do not have",
    "i cannot",
    "i can't",
    "i am unable",
    "i'm unable",
    "based on my training",
    "based on my knowledge",
    "as an artificial intelligence",
    "i don't know from the",
    "the passage does not",
    "the text does not say",
    "not mentioned in the",
    "cannot be determined from",
    "no information in the passage",
    "outside the scope of",
    "i need more context",
)

_MIN_Q_LEN = 8
_MAX_Q_LEN = 400
_MIN_A_LEN = 8
_MAX_A_LEN = 4000
# Fraction of content tokens that must appear in the chunk.
_ANSWERABILITY_MIN_OVERLAP = 0.25
_GROUNDING_MIN_OVERLAP = 0.20
_MIN_CONTENT_TOKENS_FOR_OVERLAP = 2

VALIDATION_VERSION = "strict_v1"


@dataclass(frozen=True)
class QAPairInput:
    """Normalised Q&A candidate from the parser."""

    question: str
    answer: str

    @classmethod
    def from_mapping(cls, pair: Mapping[str, Any]) -> QAPairInput:
        q = str(pair.get("q") or pair.get("question") or "").strip()
        a = str(pair.get("a") or pair.get("answer") or "").strip()
        return cls(question=q, answer=a)


@dataclass(frozen=True)
class Provenance:
    """Traceability fields stamped onto accepted pairs."""

    source_id: str
    sha256: str
    filename: str
    chunk_idx: int
    generator: str = "data-prep-runner"
    validation: str = VALIDATION_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "sha256": self.sha256,
            "filename": self.filename,
            "chunk_idx": self.chunk_idx,
            "generator": self.generator,
            "validation": self.validation,
        }


@dataclass(frozen=True)
class PairValidation:
    """Result for one candidate pair."""

    accepted: bool
    question: str
    answer: str
    reasons: tuple[str, ...] = ()
    question_norm: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "question": self.question,
            "answer": self.answer,
            "reasons": list(self.reasons),
            "question_norm": self.question_norm,
        }


@dataclass
class RejectionCounters:
    """Per-reason rejection tallies (also tracks accepted / parsed totals)."""

    parsed: int = 0
    accepted: int = 0
    rejected: int = 0
    by_reason: Counter[str] = field(default_factory=Counter)

    def record(self, result: PairValidation) -> None:
        self.parsed += 1
        if result.accepted:
            self.accepted += 1
        else:
            self.rejected += 1
            for reason in result.reasons:
                self.by_reason[reason] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "parsed": self.parsed,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "by_reason": dict(sorted(self.by_reason.items())),
        }


@dataclass
class CoverageTracker:
    """Per-source / per-chunk acceptance coverage."""

    source_id: str
    chunks_total: int
    chunks_with_accepted: set[int] = field(default_factory=set)
    accepted_per_chunk: Counter[int] = field(default_factory=Counter)

    def mark_accepted(self, chunk_idx: int) -> None:
        self.chunks_with_accepted.add(chunk_idx)
        self.accepted_per_chunk[chunk_idx] += 1

    def uncovered_chunks(self, chunk_indices: Iterable[int]) -> list[int]:
        return sorted(i for i in chunk_indices if i not in self.chunks_with_accepted)

    def as_dict(self) -> dict[str, Any]:
        covered = len(self.chunks_with_accepted)
        total = max(0, self.chunks_total)
        return {
            "source_id": self.source_id,
            "chunks_total": total,
            "chunks_with_accepted": covered,
            "chunks_uncovered": max(0, total - covered),
            "coverage_ratio": round(covered / total, 4) if total else 0.0,
            "accepted_per_chunk": {
                str(k): v for k, v in sorted(self.accepted_per_chunk.items())
            },
        }


@dataclass
class BatchValidationResult:
    """Accepted pairs + rejection/coverage stats for one chunk batch."""

    accepted: list[PairValidation] = field(default_factory=list)
    rejected: list[PairValidation] = field(default_factory=list)
    counters: RejectionCounters = field(default_factory=RejectionCounters)

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted_count": len(self.accepted),
            "rejected_count": len(self.rejected),
            "counters": self.counters.as_dict(),
            "accepted": [p.as_dict() for p in self.accepted],
            "rejected": [p.as_dict() for p in self.rejected],
        }


def normalize_question(q: str) -> str:
    """Canonical form for duplicate detection."""
    s = q.strip().lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def content_tokens(text: str) -> set[str]:
    """Lowercased content tokens excluding stopwords and single-char noise."""
    out: set[str] = set()
    for tok in _TOKEN_RE.findall(text.lower()):
        if len(tok) < 2 or tok in _STOPWORDS:
            continue
        out.add(tok)
    return out


def token_overlap_ratio(needle: set[str], haystack: set[str]) -> float:
    if not needle:
        return 0.0
    return len(needle & haystack) / max(len(needle), 1)


def is_refusal_or_meta(answer: str) -> bool:
    low = answer.lower().strip()
    if not low:
        return False
    return any(ph in low for ph in _REFUSAL_PHRASES)


def validate_qa_pair(
    question: str,
    answer: str,
    chunk: str,
    *,
    seen_questions: set[str] | None = None,
) -> PairValidation:
    """Validate one Q&A against its source chunk. Deterministic; no model calls."""
    q = (question or "").strip()
    a = (answer or "").strip()
    q_norm = normalize_question(q)
    reasons: list[str] = []

    if not q:
        reasons.append("empty_question")
    if not a:
        reasons.append("empty_answer")

    if q and not (_MIN_Q_LEN <= len(q) <= _MAX_Q_LEN) or q and len(content_tokens(q)) < 1:
        reasons.append("malformed_question")

    if a and not (_MIN_A_LEN <= len(a) <= _MAX_A_LEN):
        reasons.append("malformed_answer")

    if q_norm and seen_questions is not None and q_norm in seen_questions:
        reasons.append("duplicate_question")

    chunk_toks = content_tokens(chunk or "")
    q_toks = content_tokens(q)
    a_toks = content_tokens(a)

    if q and chunk and "empty_question" not in reasons and "malformed_question" not in reasons:
        if len(q_toks) >= _MIN_CONTENT_TOKENS_FOR_OVERLAP:
            if token_overlap_ratio(q_toks, chunk_toks) < _ANSWERABILITY_MIN_OVERLAP:
                reasons.append("unanswerable_from_chunk")
        elif q_toks and not (q_toks & chunk_toks):
            reasons.append("unanswerable_from_chunk")

    if a and chunk and "empty_answer" not in reasons and "malformed_answer" not in reasons:
        if is_refusal_or_meta(a):
            reasons.append("refusal_or_meta")
        elif len(a_toks) >= _MIN_CONTENT_TOKENS_FOR_OVERLAP:
            if token_overlap_ratio(a_toks, chunk_toks) < _GROUNDING_MIN_OVERLAP:
                reasons.append("ungrounded_answer")
        elif a_toks and not (a_toks & chunk_toks) or not a_toks:
            reasons.append("ungrounded_answer")

    accepted = not reasons
    return PairValidation(
        accepted=accepted,
        question=q,
        answer=a,
        reasons=tuple(reasons),
        question_norm=q_norm,
    )


def validate_qa_batch(
    pairs: Sequence[Mapping[str, Any]],
    chunk: str,
    *,
    seen_questions: set[str] | None = None,
) -> BatchValidationResult:
    """Validate a list of parser outputs against one chunk.

    Updates ``seen_questions`` in place for cross-chunk duplicate detection
    when a set is provided.
    """
    seen = seen_questions if seen_questions is not None else set()
    batch = BatchValidationResult()
    for raw in pairs:
        candidate = QAPairInput.from_mapping(raw)
        result = validate_qa_pair(
            candidate.question,
            candidate.answer,
            chunk,
            seen_questions=seen,
        )
        batch.counters.record(result)
        if result.accepted:
            if result.question_norm:
                seen.add(result.question_norm)
            batch.accepted.append(result)
        else:
            batch.rejected.append(result)
    return batch


def build_qa_record(
    *,
    qa_id: str,
    pair: PairValidation,
    provenance: Provenance,
    chunk_text: str,
    difficulty: str,
    style: str,
    score: float,
    created_at: float,
    status: str = "pending",
) -> dict[str, Any]:
    """Assemble the on-disk Q&A dict with provenance + validation stamp."""
    return {
        "id": qa_id,
        "source_id": provenance.source_id,
        "sha256": provenance.sha256,
        "chunk_idx": provenance.chunk_idx,
        "chunk_text": chunk_text,
        "question": pair.question,
        "answer": pair.answer,
        "difficulty": difficulty,
        "style": style,
        "score": score,
        "status": status,
        "created_at": created_at,
        "updated_at": created_at,
        "provenance": provenance.as_dict(),
        "validation": {
            "accepted": True,
            "version": VALIDATION_VERSION,
            "reasons": [],
        },
    }
