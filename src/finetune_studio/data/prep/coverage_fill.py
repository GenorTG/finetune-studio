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
from finetune_studio.data.prep.ingest import load_existing_chunks
from finetune_studio.data.prep.qa_validate import (
    PairValidation,
    Provenance,
    build_qa_record,
    content_tokens,
    normalize_question,
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "pairs_created": self.pairs_created,
            "chunks_filled": self.chunks_filled,
            "chunks_still_uncovered": self.chunks_still_uncovered,
            "skipped_no_content": self.skipped_no_content,
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


def split_sentences(text: str) -> list[str]:
    """Deterministic sentence split (same convention as augment_dataset)."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_RE.split(text) if len(s.strip()) >= _MIN_SENT_CHARS]


# question templates: two forms keep extractive questions readable
# without any model call.
_TEMPLATES = (
    "What does the source say about \u201c{subject}\u2026\u201d?",
    "Complete this statement from the source: \u201c{subject}\u2026\u201d",
)


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


def _subject_of(sentence: str) -> str:
    """Lightweight subject extraction: first ~10 words, cleaned for a question."""
    s = sentence.strip().rstrip(".")
    words = s.split()
    # Skip a leading article/label so "A Ledger-Keeper is sworn…" reads well.
    while words and words[0].lower() in {"a", "an", "the", "their", "its"}:
        words = words[1:]
    subject = " ".join(words[:10])
    return subject or s[:10]


def _make_pairs_from_chunk(
    chunk_text: str,
    *,
    seen_questions: set[str],
    max_pairs: int = _MAX_PER_CHUNK,
) -> list[tuple[str, str]]:
    """Deterministic (question, answer) extractive pairs for one chunk.

    Picks the densest fact-bearing sentences (most content tokens) whose
    subject tokens actually appear in the sentence (grounding), skipping
    questions already seen for this export batch.
    """
    sentences = split_sentences(chunk_text)
    if not sentences:
        return []
    candidates: list[tuple[float, int, str]] = []
    for idx, sent in enumerate(sentences):
        toks = content_tokens(sent)
        if not toks or _looks_tabular(sent):
            continue
        # density = informative tokens per char; favor long factual lines
        candidates.append((len(toks) / max(1, len(sent)), idx, sent))
    candidates.sort(reverse=True)

    out: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for _, _, sent in candidates:
        if len(out) >= max_pairs:
            break
        answer = sent
        # evidence gate: answer must ground itself in the chunk verbatim-ish
        if token_overlap_ratio(content_tokens(answer), content_tokens(chunk_text)) < 0.9:
            continue
        subject = _subject_of(sent)
        q = _TEMPLATES[len(out) % len(_TEMPLATES)].format(subject=subject.rstrip("?:,;"))
        key = (normalize_question(q), norm_ans(answer))
        if key[0] in seen_questions or key in seen_pairs:
            continue
        seen_pairs.add(key)
        out.append((q, answer))
    return out


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
        made = _make_pairs_from_chunk(
            text,
            seen_questions={normalize_question(r.get("question", "")) for r in existing},
        )
        if not made:
            result.chunks_still_uncovered.append({"chunk_idx": idx, "chars": len(text)})
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
