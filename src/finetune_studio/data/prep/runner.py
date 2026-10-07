"""DataPrepRunner — orchestrates the full pipeline for one uploaded file.

Stage order:
  storing → parsing → chunking → generating → done
                (or → error at any stage)

At each stage we:
  - emit a PrepProgress callback (for the SSE UI stream)
  - append an entry to logs/ingestions.jsonl
  - write the per-file artefacts to disk

Cancellation: thread-safe via threading.Event; checked between chunks during Q&A gen.
"""
from __future__ import annotations

import logging
import mimetypes
import os
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Any

from finetune_studio.data import project_filesystem as pfs
from finetune_studio.data.fs.metadata import _safe_filename
from finetune_studio.data.prep.parsers import parse_qa_json
from finetune_studio.data.prep.prompts import (
    QA_SYSTEM_PROMPT,
    QA_USER_TEMPLATE,
    style_hint,
)
from finetune_studio.data.prep.qa_validate import (
    CoverageTracker,
    Provenance,
    RejectionCounters,
    build_qa_record,
    normalize_question,
    validate_qa_batch,
    validate_qa_pair,
)
from finetune_studio.data.prep.scorer import heuristic_score

log = logging.getLogger(__name__)

# Exhaustive mining writes one pair per fact: a dense chunk needs far more than the sampled miner's 1,200 tokens.
EXHAUSTIVE_MAX_TOKENS = 4096


@dataclass
class PrepProgress:
    stage: str = "queued"  # queued|storing|parsing|chunking|generating|done|error
    pct: float = 0.0
    message: str = ""
    source_id: str = ""
    sha256: str = ""
    chunks_total: int = 0
    chunks_done: int = 0
    qa_total: int = 0


class DataPrepRunner:
    def __init__(self, pid: str, data: bytes, filename: str, *,
                 qa_per_chunk: int = 3, difficulty: str = "medium",
                 style: str = "socratic", max_chunks: int = 0,
                 uploaded_by: str = "", progress_cb=None, mode: str = "exhaustive"):
        self.pid = pid
        self.data = data
        self.filename = filename
        self.qa_per_chunk = max(1, min(10, qa_per_chunk))
        # exhaustive: loop until every fact-bearing statement is covered (qa_per_chunk ignored); sampled: legacy n per chunk
        self.mode = mode if mode in ("exhaustive", "sampled") else "exhaustive"
        self.difficulty = difficulty
        self.style = style
        self.max_chunks = max_chunks
        self.uploaded_by = uploaded_by
        self.cb = progress_cb or (lambda p: None)
        self.progress = PrepProgress()
        self.source_id = ""
        self.sha256 = ""
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def _emit(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self.progress, k, v)
        try:
            self.cb(self.progress)
        except Exception:
            # Progress UI callback must never abort prep; log and continue.
            log.exception("prep progress callback failed")

    def _api_workers(self) -> int:
        """Parallel model calls: an API helper serves requests concurrently, a local GGUF does not (one GPU)."""
        try:
            from finetune_studio.models.manager import get_manager

            if (get_manager().active() or {}).get("kind") == "openai_compat":
                return max(1, min(16, int(os.environ.get("FTS_API_CONCURRENCY", "6"))))
        except Exception:
            log.debug("could not read the active provider for concurrency", exc_info=True)
        return 1

    def _generate_exhaustive(self, chat, chunks, meta, coverage, rejection_counters, seen_questions, now):
        """Exhaustive mining (see ``exhaustive``): per chunk, loop until every fact-bearing statement is covered.

        Chunks are independent (each carries everything it needs), so with an API helper they are mined in parallel and
        written in chunk order; cross-chunk duplicate questions are dropped when written.
        """
        from concurrent.futures import ThreadPoolExecutor

        from finetune_studio.data.prep import exhaustive as ex
        from finetune_studio.data.prep.coverage_question import split_sections

        title = ex.document_title(chunks[0] if chunks else "", self.filename)
        stats: dict[str, Any] = {"statements": 0, "covered": 0, "model_pairs": 0, "gap_pairs": 0, "extractive_pairs": 0,
                                 "rounds": 0}
        total_qa = 0

        def call(messages):
            return chat(messages, max_tokens=EXHAUSTIVE_MAX_TOKENS, temperature=0.0, top_p=1.0)

        def mine(i: int):
            if self._cancel.is_set():
                return None
            chunk = chunks[i - 1]
            carried = ex.table_header_before(chunks, i - 1)
            grounding = f"{carried}\n{chunk}" if carried else chunk
            local_seen: set[str] = set()
            counts = RejectionCounters()

            def accept(pairs, _chunk):
                batch = validate_qa_batch(pairs, grounding, seen_questions=local_seen)
                counts.parsed += batch.counters.parsed
                counts.accepted += batch.counters.accepted
                counts.rejected += batch.counters.rejected
                counts.by_reason.update(batch.counters.by_reason)
                return [{"q": p.question, "a": p.answer} for p in batch.accepted]

            sections = split_sections(chunk)
            section = (sections[0][0] if sections else "") or ""
            outcome = ex.mine_chunk(chunk, chat=call, parse=lambda raw: parse_qa_json(raw, 400), accept=accept,
                                    title=title, section=section, carried_header=carried)
            return outcome, grounding, counts

        workers = self._api_workers()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, result in enumerate(pool.map(mine, range(1, len(chunks) + 1)), 1):
                if result is None or self._cancel.is_set():
                    pool.shutdown(wait=False, cancel_futures=True)
                    return total_qa, stats, True
                outcome, grounding, counts = result
                chunk = chunks[i - 1]
                rejection_counters.parsed += counts.parsed
                rejection_counters.accepted += counts.accepted
                rejection_counters.rejected += counts.rejected
                rejection_counters.by_reason.update(counts.by_reason)
                prov = Provenance(source_id=self.source_id, sha256=meta.sha256, filename=self.filename, chunk_idx=i)
                for pair, origin in outcome.pairs:
                    qn = normalize_question(pair["q"])
                    if qn in seen_questions:
                        continue                       # the same question was already written for an earlier chunk
                    seen_questions.add(qn)
                    validated = validate_qa_pair(pair["q"], pair["a"], grounding)
                    qa = build_qa_record(
                        qa_id=uuid.uuid4().hex[:12], pair=validated, provenance=prov, chunk_text=chunk[:1500],
                        difficulty=self.difficulty, style=self.style,
                        score=heuristic_score(pair["q"], pair["a"], chunk), created_at=now, origin=origin,
                    )
                    pfs.write_qa_pair(self.pid, qa)
                    coverage.mark_accepted(i)
                    total_qa += 1
                    key = {"model": "model_pairs", "model_gap": "gap_pairs", "extractive_gap": "extractive_pairs"}[origin]
                    stats[key] += 1
                stats["statements"] += outcome.statements
                stats["covered"] += outcome.covered
                stats["rounds"] += outcome.rounds
                self._emit(stage="generating", pct=30 + (i / max(1, len(chunks))) * 65, chunks_done=i, qa_total=total_qa,
                           message=f"Chunk {i}/{len(chunks)}: {stats['covered']}/{stats['statements']} statements covered")
        return total_qa, stats, False

    def run(self) -> dict:
        try:
            return self._run_inner()
        except Exception as e:
            log.exception("data prep failed")
            self._emit(stage="error", message=str(e))
            return {"ok": False, "error": str(e)}

    def _run_inner(self) -> dict:
        # 1) Store content-addressed
        self._emit(stage="storing", pct=3, message=f"Storing {self.filename}…")
        mime, _ = mimetypes.guess_type(self.filename)
        _, meta = pfs.store_file(
            self.pid, self.data,
            original_filename=self.filename, mime_type=mime or "",
            uploaded_by=self.uploaded_by, source_kind="upload",
        )
        self.sha256 = meta.sha256
        self.source_id = meta.sha256[:12]  # short sha as source id
        pfs.log_ingestion(self.pid, {
            "event": "upload", "sha256": meta.sha256, "filename": self.filename,
            "byte_count": meta.byte_count, "uploaded_by": self.uploaded_by,
        })
        self._emit(stage="parsing", pct=10, sha256=meta.sha256,
                   message="Parsing with dedicated parser script…")
        # 2–5) Shared parse + chunk (also used by promote-from-library)
        from finetune_studio.data.prep.ingest import parse_and_chunk

        ingest = parse_and_chunk(
            self.pid,
            self.data,
            self.filename,
            sha256=meta.sha256,
            max_chunks=self.max_chunks,
            mime_type=mime or "",
            reuse_if_parsed=True,
        )
        if not ingest.ok:
            if ingest.error == "empty parse":
                pfs.log_ingestion(self.pid, {
                    "event": "parse_empty", "sha256": meta.sha256,
                    "filename": self.filename, "parser": ingest.parser or "?",
                })
                self._emit(stage="error",
                           message="Parser returned empty text. Unsupported format?")
            else:
                self._emit(stage="error", message="No chunks produced.")
            return {"ok": False, "error": ingest.error or "parse failed",
                    "sha256": meta.sha256}

        chunks = ingest.chunks
        if ingest.reused:
            pfs.log_ingestion(self.pid, {
                "event": "parsed_reused", "sha256": meta.sha256,
                "filename": self.filename, "parser": ingest.parser or "?",
                "char_count": ingest.char_count,
            })
        else:
            pfs.log_ingestion(self.pid, {
                "event": "parsed", "sha256": meta.sha256, "filename": self.filename,
                "parser": ingest.parser or "?",
                "char_count": ingest.char_count, "warnings": ingest.warnings,
            })
        self._emit(stage="chunking", pct=20, message="Splitting into semantic chunks…")
        # Q&A source manifest (filesystem-side). Keep a resolvable data_path:
        # the content-addressed raw file is the canonical location, and prep
        # re-runs (source picker → Start prep) resolve the source through it.
        # Bug 2026-09-18: this rewrite used to DROP data_path, so every
        # re-run after a restart 404'd with "source file missing".
        raw_canonical = pfs.file_dir(self.pid, meta.sha256) / _safe_filename(self.filename)
        pfs.write_qa_source(self.pid, {
            "id": self.source_id,
            "sha256": meta.sha256,
            "filename": self.filename,
            "mime_type": mime or "",
            "char_count": ingest.char_count,
            "chunk_count": ingest.chunk_count,
            "parser": ingest.parser,
            "uploaded_at": meta.uploaded_at,
            "status": "ready",
            "data_path": str(raw_canonical),
            "path": str(raw_canonical),
        })
        pfs.log_ingestion(self.pid, {
            "event": "chunked", "sha256": meta.sha256, "filename": self.filename,
            "chunk_count": ingest.chunk_count,
        })
        # 6) Q&A generation per chunk
        from finetune_studio.data.prep.generator import (
            helper_resolution_error,
            resolve_generator,
        )

        chat = resolve_generator()
        if chat is None:
            err = helper_resolution_error()
            self._emit(stage="error", message=err)
            return {"ok": False, "error": err, "sha256": meta.sha256}
        self._emit(stage="generating", pct=30, source_id=self.source_id,
                   chunks_total=len(chunks), chunks_done=0, qa_total=0,
                   message=f"Model generating Q&A from {len(chunks)} chunks…")
        total_qa = 0
        now = time.time()
        rejection_counters = RejectionCounters()
        coverage = CoverageTracker(
            source_id=self.source_id, chunks_total=len(chunks),
        )
        seen_questions: set[str] = set()
        chunk_indices = list(range(1, len(chunks) + 1))
        fact_stats: dict[str, Any] = {}
        if self.mode == "exhaustive":
            total_qa, fact_stats, cancelled = self._generate_exhaustive(
                chat, chunks, meta, coverage, rejection_counters, seen_questions, now)
            if cancelled:
                self._emit(stage="error", message="Cancelled")
                return {"ok": False, "error": "cancelled", "sha256": meta.sha256}
        else:
            for i, chunk in enumerate(chunks, 1):
                if self._cancel.is_set():
                    self._emit(stage="error", message="Cancelled")
                    return {"ok": False, "error": "cancelled", "sha256": meta.sha256}
                prompt = QA_USER_TEMPLATE.format(chunk=chunk[:6000], n=self.qa_per_chunk,
                                                 difficulty=self.difficulty, style_hint=style_hint(self.style))
                try:
                    raw = chat(
                        [{"role": "system", "content": QA_SYSTEM_PROMPT},
                         {"role": "user", "content": prompt}],
                        max_tokens=1200, temperature=0.7, top_p=0.9,
                    )
                except Exception as e:
                    log.exception("model call failed on chunk %d", i)
                    pfs.log_ingestion(self.pid, {
                        "event": "qa_chunk_error", "sha256": meta.sha256, "chunk_index": i,
                        "error": str(e),
                    })
                    continue
                # Parsing fallbacks stay in parse_qa_json; validation is post-parse.
                pairs = parse_qa_json(raw, self.qa_per_chunk)
                if not pairs:
                    # The model replied but no {"q","a"} pair could be extracted
                    # (empty array, malformed JSON, refusal prose, etc). Record
                    # it — otherwise this chunk vanishes from the ingestion log
                    # with no trace until coverage_fill silently backfills it.
                    pfs.log_ingestion(self.pid, {
                        "event": "qa_chunk_unparsed", "sha256": meta.sha256,
                        "chunk_index": i, "raw_preview": raw[:200],
                    })
                    continue
                batch = validate_qa_batch(
                    pairs, chunk, seen_questions=seen_questions,
                )
                rejection_counters.parsed += batch.counters.parsed
                rejection_counters.accepted += batch.counters.accepted
                rejection_counters.rejected += batch.counters.rejected
                rejection_counters.by_reason.update(batch.counters.by_reason)
                prov = Provenance(
                    source_id=self.source_id,
                    sha256=meta.sha256,
                    filename=self.filename,
                    chunk_idx=i,
                )
                for accepted in batch.accepted:
                    qa_id = uuid.uuid4().hex[:12]
                    qa = build_qa_record(
                        qa_id=qa_id,
                        pair=accepted,
                        provenance=prov,
                        chunk_text=chunk[:1500],
                        difficulty=self.difficulty,
                        style=self.style,
                        score=heuristic_score(accepted.question, accepted.answer, chunk),
                        created_at=now,
                    )
                    pfs.write_qa_pair(self.pid, qa)
                    coverage.mark_accepted(i)
                    total_qa += 1
                if batch.rejected:
                    pfs.log_ingestion(self.pid, {
                        "event": "qa_chunk_rejected",
                        "sha256": meta.sha256,
                        "chunk_index": i,
                        "rejected": len(batch.rejected),
                        "accepted": len(batch.accepted),
                        "reasons": dict(Counter(
                            reason for p in batch.rejected for reason in p.reasons
                        )),
                    })
                pct = 30 + (i / max(1, len(chunks))) * 65
                self._emit(stage="generating", pct=pct, chunks_done=i, qa_total=total_qa)
        coverage_info = coverage.as_dict()
        if fact_stats:
            coverage_info["facts"] = fact_stats
        coverage_info["uncovered_chunk_indices"] = coverage.uncovered_chunks(chunk_indices)
        rejection_info = rejection_counters.as_dict()
        # Self-heal: chunks the model mining never converted get deterministic
        # extractive pairs now, so nothing parsed is left out of the next export.
        fill_summary: dict[str, Any] | None = None
        fill_pairs = 0
        try:
            from finetune_studio.data.prep.coverage_fill import (
                FillResult,
                fill_coverage_gaps,
            )

            # Exhaustive mining closes its own gaps (flagged extractive pairs, pending review). The legacy fill would add
            # AUTO-APPROVED extractive pairs on top of every chunk whose model pairs are still pending review.
            fill = FillResult() if self.mode == "exhaustive" else fill_coverage_gaps(
                self.pid, self.source_id, meta.sha256,
                chunk_texts={i: c for i, c in enumerate(chunks, 1)},
                filename=self.filename,
            )
            fill_summary = fill.as_dict()
            fill_pairs = fill.pairs_created
            if fill.pairs_created:
                # merge honestly: tracker counts model-accepted chunks, the fill
                # result says which gaps it closed. No disk re-read (pfs may be
                # test-mocked).
                still_open = {
                    int(u["chunk_idx"])
                    for u in fill_summary.get("chunks_still_uncovered", [])
                }
                filled_idx = set(range(1, len(chunks) + 1)) - still_open
                covered_now = (coverage.chunks_with_accepted | filled_idx) & set(chunk_indices)
                coverage_info["chunks_with_accepted"] = len(covered_now)
                coverage_info["chunks_uncovered"] = len(chunk_indices) - len(covered_now)
                coverage_info["coverage_ratio"] = round(
                    coverage_info["chunks_with_accepted"] / max(1, len(chunk_indices)), 4)
        except Exception as exc:
            log.exception("coverage fill failed for %s", self.source_id)
            fill_summary = {
                "error": str(exc),
                "chunks_still_uncovered": [
                    {"source": self.source_id, "chunk_idx": idx}
                    for idx in coverage.uncovered_chunks(chunk_indices)
                ],
            }
        uncovered_after_fill = (
            fill_summary.get("chunks_still_uncovered", [])
            if fill_summary
            else [
                {"source": self.source_id, "chunk_idx": idx}
                for idx in coverage.uncovered_chunks(chunk_indices)
            ]
        )
        if uncovered_after_fill:
            log.warning(
                "source %s: %d chunk(s) could not yield any pair even extractively: %s",
                self.source_id, len(uncovered_after_fill), uncovered_after_fill,
            )
        pfs.log_ingestion(self.pid, {
            "event": "qa_generated", "sha256": meta.sha256, "filename": self.filename,
            "qa_count": total_qa,
            "qa_coverage_fill": fill_pairs,
            "rejection_counters": rejection_info,
            "coverage": coverage_info,
            "coverage_fill": fill_summary,
        })
        fact_open = bool(fact_stats) and fact_stats["covered"] < fact_stats["statements"]
        source_manifest = pfs.read_qa_source(self.pid, self.source_id)
        if source_manifest:
            pfs.write_qa_source(self.pid, {
                **source_manifest,
                "status": "generated_incomplete" if (uncovered_after_fill or fact_open) else "generated",
                "generated_at": time.time(),
                "generation": {
                    "qa_model_accepted": total_qa,
                    "qa_coverage_fill": fill_pairs,
                    "rejection_counters": rejection_info,
                    "coverage": coverage_info,
                },
            })
        if uncovered_after_fill:
            vague = sum(1 for u in uncovered_after_fill if u.get("reason") == "no_specific_question")
            message = (
                f"Coverage incomplete: {len(uncovered_after_fill)} of {len(chunks)} "
                "chunk(s) still have no approved pair"
                + (f" ({vague} could not yield a specific question)" if vague else "")
                + ". Dataset export is blocked."
            )
            self._emit(
                stage="error", pct=100, chunks_done=len(chunks),
                qa_total=total_qa + fill_pairs, message=message,
            )
            return {
                "ok": False,
                "error": message,
                "source_id": self.source_id,
                "sha256": meta.sha256,
                "chunks": len(chunks),
                "qa": total_qa,
                "qa_coverage_fill": fill_pairs,
                "coverage": coverage_info,
                "coverage_fill": fill_summary,
            }
        self._emit(stage="done", pct=100, chunks_done=len(chunks), qa_total=total_qa,
                   message=(
                       f"Done. {total_qa} accepted Q&A pairs from {len(chunks)} chunks "
                       f"({rejection_counters.rejected} rejected; "
                       f"+{fill_pairs} coverage-fill extractive)."
                   ))
        return {
            "ok": True,
            "source_id": self.source_id,
            "sha256": meta.sha256,
            "chunks": len(chunks),
            "qa": total_qa,
            "qa_coverage_fill": fill_pairs,
            "filename": self.filename,
            "rejection_counters": rejection_info,
            "coverage": coverage_info,
            "coverage_fill": fill_summary,
        }
