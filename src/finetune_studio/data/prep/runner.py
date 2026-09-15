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
import threading
import time
import uuid
from dataclasses import dataclass

from finetune_studio.data import project_filesystem as pfs
from finetune_studio.data.prep.parsers import parse_qa_json
from finetune_studio.data.prep.prompts import (
    QA_SYSTEM_PROMPT,
    QA_USER_TEMPLATE,
    style_hint,
)
from finetune_studio.data.prep.scorer import heuristic_score

log = logging.getLogger(__name__)


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
                 uploaded_by: str = "", progress_cb=None):
        self.pid = pid
        self.data = data
        self.filename = filename
        self.qa_per_chunk = max(1, min(10, qa_per_chunk))
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
        # Q&A source manifest (filesystem-side)
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
            pairs = parse_qa_json(raw, self.qa_per_chunk)
            if not pairs:
                continue
            for j, p in enumerate(pairs):
                qa_id = uuid.uuid4().hex[:12]
                qa = {
                    "id": qa_id, "source_id": self.source_id,
                    "chunk_idx": i, "chunk_text": chunk[:1500],
                    "question": p["q"], "answer": p["a"],
                    "difficulty": self.difficulty, "style": self.style,
                    "score": heuristic_score(p["q"], p["a"], chunk),
                    "status": "pending", "created_at": now, "updated_at": now,
                }
                pfs.write_qa_pair(self.pid, qa)
                total_qa += 1
            pct = 30 + (i / max(1, len(chunks))) * 65
            self._emit(stage="generating", pct=pct, chunks_done=i, qa_total=total_qa)
        pfs.log_ingestion(self.pid, {
            "event": "qa_generated", "sha256": meta.sha256, "filename": self.filename,
            "qa_count": total_qa,
        })
        self._emit(stage="done", pct=100, chunks_done=len(chunks), qa_total=total_qa,
                   message=f"Done. {total_qa} Q&A pairs from {len(chunks)} chunks.")
        return {"ok": True, "source_id": self.source_id, "sha256": meta.sha256,
                "chunks": len(chunks), "qa": total_qa, "filename": self.filename}
