"""Coverage-fill miner — guarantee every parsed chunk lands in the training set.

THE PROBLEM THIS SOLVES
======================
LLM mining passes over chunked sources stochastically: a chunk whose model
output parses to zero accepted pairs is silently skipped, leaving parsed
information (OCR'd scans, spreadsheets, one-chunk sources) absent from the
training dataset while every counter still reads "done".

This module closes that hole with a second, deterministic pass:

  1. Compute every (source, chunk) that produced no accepted pair.
  2. For each uncovered chunk, emit extractive Q&A pairs — question
     templated from, and the answer quoted verbatim from, the chunk text.
  3. Re-run the same strict validation gate the model-generated pairs use,
     so no garbage enters storage. Items that genuinely cannot yield a
     pair (e.g. a chunk of pure boilerplate) are surfaced as uncovered —
     never silently dropped.

Questions must be self-contained and specific (`coverage_question`): a
distinctive subject plus the section/file scope. A chunk whose sentences yield
no such question is NOT given a vague one ("What does the source say about
\u201cIt\u201d?"); it is reported in ``chunks_still_uncovered`` with
``reason="no_specific_question"`` and counted in ``no_specific_question``, so
the runner marks the source ``generated_incomplete`` and the export gate blocks
it (``force=true`` overrides).

Extractive pairs are marked ``status="approved"`` with
``origin="coverage_fill"`` provenance: content is quoted, not invented, so
the triage bar the human reviewer applies to model guesses is not needed
for verbatim extraction.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from finetune_studio.data import project_filesystem as pfs
from finetune_studio.data.prep.coverage_question import (
    build_question,
    proper_words,
    scope_for,
    split_sections,
)
from finetune_studio.data.prep.ingest import load_existing_chunks
from finetune_studio.data.prep.qa_validate import (
    PairValidation,
    Provenance,
    build_qa_record,
    content_tokens,
    token_overlap_ratio,
)

log = logging.getLogger(__name__)

# Sentences shorter than this rarely carry a learnable fact.
_MIN_SENT_CHARS = 25
# How many extractive pairs to emit per uncovered chunk, max.
_MAX_PER_CHUNK = 3


@dataclass
class FillResult:
    pairs_created: int = 0
    chunks_filled: int = 0
    chunks_still_uncovered: list[dict[str, int]] = field(default_factory=list)
    skipped_no_content: int = 0
    # chunks with text but no question that is self-contained and specific
    no_specific_question: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "pairs_created": self.pairs_created,
            "chunks_filled": self.chunks_filled,
            "chunks_still_uncovered": self.chunks_still_uncovered,
            "skipped_no_content": self.skipped_no_content,
            "no_specific_question": self.no_specific_question,
        }


# ── project-wide entry ──────────────────────────────────────────────────

def _fill_sources(pid: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Fill the given qa source manifests; aggregate into one summary."""
    total = FillResult()
    uncovered: list[dict[str, Any]] = []
    for src in sources:
        sid = str(src.get("id") or "")
        sha = str(src.get("sha256") or "")
        filename = str(src.get("filename") or src.get("name") or "")
        declared_chunks = int(src.get("chunk_count") or 0)
        try:
            chunks = load_existing_chunks(pid, sha) if sha else []
            result = fill_coverage_gaps(
                pid, sid, sha,
                chunk_texts={i: c for i, c in enumerate(chunks, 1)} or None,
                filename=filename,
            )
        except Exception as exc:
            # a source whose parsed artifacts are gone must surface, not vanish
            log.exception("coverage fill failed for source %s", sid)
            n = declared_chunks if declared_chunks else 1
            uncovered.extend(
                {"source": sid, "filename": filename, "chunk_idx": i,
                 "error": f"fill failed: {exc}"}
                for i in range(1, n + 1)
            )
            continue
        total.pairs_created += result.pairs_created
        total.chunks_filled += result.chunks_filled
        total.skipped_no_content += result.skipped_no_content
        total.no_specific_question += result.no_specific_question
        for unc in result.chunks_still_uncovered:
            unc["source"] = sid
            unc["filename"] = filename
            uncovered.append(unc)
        # Parsed-artifact loss: the source declares chunks but none could be
        # loaded — a silent hole in the dataset unless surfaced here.
        if declared_chunks and not chunks:
            uncovered.extend(
                {"source": sid, "filename": filename, "chunk_idx": i,
                 "error": "parsed chunks missing"}
                for i in range(1, declared_chunks + 1)
            )
    summary = total.as_dict()
    summary["uncovered_chunks"] = uncovered
    return summary


def fill_all_project_gaps(pid: str) -> dict[str, Any]:
    """Run the fill pass for every qa source in the project.

    Returns an aggregate summary used by the export gate:
    ``{pairs_created, chunks_filled, skipped_no_content, uncovered_chunks}``. This is the function the export pipeline calls so a
    dataset can never ship with silently-unmined chunks.
    """
    return _fill_sources(pid, pfs.list_qa_sources(pid))


def fill_sources_gaps(pid: str, source_ids: list[str]) -> dict[str, Any]:
    """Coverage-fill ONLY the given sources (subset builds).

    Same semantics as `fill_all_project_gaps` but restricted to the picked
    source ids so a specialized subset dataset can still carry the hard
    no-skips guarantee for exactly the files it derives from. Unknown ids
    contribute nothing; callers validate counts against their own request.
    """
    wanted = {s for s in source_ids if s}
    return _fill_sources(
        pid, [s for s in pfs.list_qa_sources(pid) if str(s.get("id") or "") in wanted]
    )


# ── sentence splitting ───────────────────────────────────────────────────

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


# A period after these does not end a sentence. Without this "...given by Dr. Maren Voss." was cut into
# "...given by Dr." — an approved extractive training answer that taught the model to stop mid-name.
_ABBREVIATIONS = frozenset({
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc", "no", "fig", "inc", "ltd", "co", "cf",
    "approx", "dept", "mt", "gen", "col", "lt", "sgt", "capt", "rev", "hon", "e.g", "i.e",
})


def _ends_with_abbreviation(piece: str) -> bool:
    last = piece.rstrip().rsplit(" ", 1)[-1].rstrip(".").lower()
    return last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha() and piece.rstrip().endswith("."))   # "J. Smith"


def split_sentences(text: str) -> list[str]:
    """Deterministic sentence split (same convention as augment_dataset), abbreviation-aware."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    merged: list[str] = []
    for piece in _SENT_RE.split(text):
        if merged and _ends_with_abbreviation(merged[-1]):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return [s.strip() for s in merged if len(s.strip()) >= _MIN_SENT_CHARS]


def _looks_tabular(sentence: str) -> bool:
    """True for CSV/table residue (pipes, many commas, mostly digits/symbols)."""
    if "|" in sentence or "---" in sentence:
        return True
    words = sentence.split()
    if not words:
        return True
    if sentence.count(",") >= max(4, len(words) // 2):
        return True
    alpha = sum(1 for w in words if re.fullmatch(r"[A-Za-z][A-Za-z'\-]*[.,;:!?]?", w))
    return alpha / len(words) < 0.6


def _plan_pairs(
    chunk_text: str,
    *,
    seen_questions: set[str],
    max_pairs: int = _MAX_PER_CHUNK,
    filename: str = "",
) -> tuple[list[tuple[str, str]], int]:
    """Deterministic (question, answer) pairs for one chunk + sentences dropped.

    Picks the densest fact-bearing sentences (most content tokens) whose
    answer grounds itself in the chunk, then asks a *self-contained* question
    about each (`coverage_question.build_question`: distinctive subject plus
    section/file scope). A sentence with no such question is dropped, never
    asked vaguely; the second return value counts those drops.
    """
    candidates: list[tuple[float, int, str, str]] = []  # (density, order, sentence, scope)
    order = 0
    for heading, paragraph in split_sections(chunk_text):
        scope = scope_for(heading, filename, chunk_text)
        for sent in split_sentences(paragraph):
            toks = content_tokens(sent)
            if not toks or _looks_tabular(sent):
                continue
            # density = informative tokens per char; favor long factual lines
            candidates.append((len(toks) / max(1, len(sent)), order, sent, scope))
            order += 1
    candidates.sort(key=lambda c: (-c[0], c[1]))

    chunk_tokens = content_tokens(chunk_text)
    proper = proper_words(chunk_text)
    out: list[tuple[str, str]] = []
    asked = list(seen_questions)
    dropped = 0
    seen_answers: set[str] = set()
    for _, _, sent, scope in candidates:
        if len(out) >= max_pairs:
            break
        # evidence gate: answer must ground itself in the chunk verbatim-ish
        if token_overlap_ratio(content_tokens(sent), chunk_tokens) < 0.9:
            continue
        q = build_question(sent, scope=scope, variant=len(out), proper=proper, avoid=asked)
        if q is None:
            dropped += 1
            continue
        if norm_ans(sent) in seen_answers:
            continue
        seen_answers.add(norm_ans(sent))
        asked.append(q)
        out.append((q, sent))
    return out, dropped


def _make_pairs_from_chunk(
    chunk_text: str,
    *,
    seen_questions: set[str],
    max_pairs: int = _MAX_PER_CHUNK,
    filename: str = "",
) -> list[tuple[str, str]]:
    """Pairs only; see `_plan_pairs` for selection rules and the drop contract."""
    return _plan_pairs(
        chunk_text, seen_questions=seen_questions, max_pairs=max_pairs, filename=filename
    )[0]


def norm_ans(a: str) -> str:
    return re.sub(r"\s+", " ", a or "").strip().lower()


# ── main entry ──────────────────────────────────────────────────────────

def fill_coverage_gaps(
    pid: str,
    source_id: str,
    sha256: str = "",
    *,
    chunk_texts: dict[int, str] | None = None,
    filename: str = "",
) -> FillResult:
    """Create approved extractive pairs for every chunk with no accepted pair.

    ``chunk_texts`` optionally maps 1-based chunk index -> chunk text. When
    omitted (the normal case), chunks are loaded from the stored parsed
    artifacts via the source's sha256 — the same files the mining runner
    read, so the filler covers exactly what was parsed.

    ``sha256``/``filename`` are stamped onto each pair's provenance when the
    caller has them, so an extractive pair carries the same traceability as
    a model-mined one instead of leaving those fields blank.

    Reads current pairs from disk to decide what is uncovered, writes new
    pairs with ``origin="coverage_fill"``, and returns a FillResult.
    Raises nothing for content problems; those are reported as uncovered.
    """
    if chunk_texts is None:
        chunks = load_existing_chunks(pid, sha256)
        chunk_texts = {i: c for i, c in enumerate(chunks, 1)}
    existing = pfs.list_qa_pairs(pid, source_id=source_id)
    covered = {int(r.get("chunk_idx") or 0) for r in existing if r.get("status") == "approved"}
    gaps = sorted(i for i in chunk_texts if i not in covered)

    result = FillResult()
    now = time.time()
    for idx in gaps:
        text = chunk_texts[idx]
        if not text or not text.strip():
            # surfaced as uncovered: the export gate only reads chunks_still_uncovered
            result.skipped_no_content += 1
            result.chunks_still_uncovered.append(
                {"chunk_idx": idx, "chars": 0, "reason": "no_content"}
            )
            continue
        made, dropped = _plan_pairs(
            text,
            seen_questions={str(r.get("question", "")) for r in existing},
            filename=filename,
        )
        if not made:
            # A vague question would cap suite/held-out scores and hide the gap;
            # report the chunk (export gate blocks it) instead of faking coverage.
            reason = "no_specific_question" if dropped else "no_extractable_sentence"
            if dropped:
                result.no_specific_question += 1
            result.chunks_still_uncovered.append(
                {"chunk_idx": idx, "chars": len(text), "reason": reason}
            )
            continue
        for q, a in made:
            qa_id = uuid.uuid4().hex[:12]
            qa = build_qa_record(
                qa_id=qa_id,
                pair=PairValidation(accepted=True, question=q, answer=a),
                provenance=Provenance(source_id=source_id, sha256=sha256,
                                      filename=filename, chunk_idx=idx),
                chunk_text=text[:1500],
                difficulty="medium",
                style="extractive",
                score=1.0,
                created_at=now,
                status="approved",
            )
            qa["origin"] = "coverage_fill"
            pfs.write_qa_pair(pid, qa)
            result.pairs_created += 1
        result.chunks_filled += 1
    return result
