"""One mining job: one parsed file, fixed stages, every step stored.

Stages of a ``mine`` job::

    inventory  rule items + one helper call per chunk listing its data points + one whole-file call for facts that span parts
               of the document; ungrounded values dropped; the items are stored before any pair is written.
    mining     one helper call per chunk with that chunk's checklist; validated pairs are written as PENDING.
    checking   which items no pair states yet (``coverage``).
    fixing     one call per chunk that still has open items, naming exactly those items; repeated until nothing is open,
               ``MAX_FIX_ROUNDS`` is reached, or a round closes nothing (no progress).

A ``fill`` job runs only checking + fixing on a reviewed file, and shows the helper the pairs the reviewer rejected for that
chunk with their reasons. A job ends ``done`` (no open item) or ``gaps`` (open items left for the reviewer: write a pair or
exclude the item), ``failed``, or ``cancelled``. Pairs are never auto-approved: the reviewer decides.
"""
from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from finetune_studio.data.mining import prompts
from finetune_studio.data.mining.coverage import item_states
from finetune_studio.data.mining.inventory import (
    agent_items,
    chunk_of,
    ground,
    item_prompt_text,
    parse_inventory,
    parse_skips,
    rule_items,
)
from finetune_studio.data.mining.match import contains, loosely_contains, norm
from finetune_studio.data.mining.statements import (
    document_title,
    drop_redundant,
    keyed_chunk,
    strip_overlap,
    table_header_before,
)
from finetune_studio.data.prep.parsers import parse_qa_json
from finetune_studio.data.prep.qa_validate import (
    Provenance,
    build_qa_record,
    normalize_question,
    validate_qa_batch,
)
from finetune_studio.data.prep.scorer import heuristic_score
from finetune_studio.data.prep.tokens import distinctive_tokens
from finetune_studio.knowledge.sources import Source

log = logging.getLogger(__name__)

MAX_FIX_ROUNDS = 3
MAX_TOKENS = 4096
FILE_PASS_MAX_CHARS = 40_000   # the whole-file inventory call only for files that fit a helper's context comfortably

ChatFn = Callable[..., str]


class JobCancelled(Exception):
    pass


def document_identifiers(title: str, first_chunk: str) -> str:
    """The codes, numbers and names the document introduces up front ("RB-017", "KC-Hub 200", "Grades O1-O6").

    A pair from a later chunk may name its document by them; checking it against that chunk alone rejected it as invented."""
    toks = distinctive_tokens(f"{title}\n{first_chunk}")
    return " ".join(sorted(toks))


SPAN_PROMPT_CHARS = 7500   # the prompts cut a passage at 8000 characters
CONTEXT_BEFORE_CHARS = 300
CONTEXT_AFTER_CHARS = 200


def neighbour_text(texts: list[str], i: int) -> tuple[str, str]:
    """(tail of chunk ``i-1``, head of chunk ``i+1``), cut at word boundaries; empty where there is no neighbour."""
    before = after = ""
    if i > 0:
        tail = " ".join(texts[i - 1].split())[-CONTEXT_BEFORE_CHARS:]
        before = tail.split(" ", 1)[1] if " " in tail and len(tail) == CONTEXT_BEFORE_CHARS else tail
    if i + 1 < len(texts):
        head = " ".join(texts[i + 1].split())
        after = head[:CONTEXT_AFTER_CHARS].rsplit(" ", 1)[0] if len(head) > CONTEXT_AFTER_CHARS else head
    return before, after


# Which side of a correction is the old value. The whole-file pass words it differently every run, so each rule below
# names one wording and the first that finds values on both sides wins:
#   "changed from OLD to NEW" | "replaced by / corrected (on <date>) to / moved to NEW" (old before) |
#   "OLD was superseded and the current ... is NEW" | "... is NEW, not OLD" (in a correction) |
#   "NEW, replacing / superseding / instead of the earlier OLD".
# A correction whose direction is not stated this way gets no note: a note on a chunk that holds only the NEW value made the
# helper write "3 months' notice ... was later corrected" for the current rule (core mine v4a).
_FROM_TO = re.compile(r"\bchanged from\b(?P<old>.*?)\bto\b", re.IGNORECASE)
_NEW_AFTER = re.compile(r"\b(?:(?:replaced|superseded) by|(?:corrected|changed|moved|moves|updated|amended)\b[^.;]{0,25}?\bto)\b",
                        re.IGNORECASE)
# passive "superseded" only: "is replaced, after probation, by 1 month" is a rule, not a correction
_PASSIVE = re.compile(r"\b(?:is|was|were|are|has been)\s+superseded\b(?!\s+by\b)", re.IGNORECASE)
_NOT_OLD = re.compile(r",?\s+not\s+(?:the\s+)?(?=[^.;]*\d)", re.IGNORECASE)
_CORRECTION_WORD = re.compile(r"\b(?:correct(?:ed|ion)|supersed\w*|replac\w*)\b", re.IGNORECASE)
# "replaced" / "instead of" alone also describe rules ("175 % instead of the 200 % Sunday rate", "replaced, after probation, by
# clause 10.1"): they count only next to an old-value marker.
_OLD_AFTER = re.compile(r"\b(?:replac(?:ing|es|ed)|supersed(?:ing|es|ed)|instead of)\b(?=[^.;]{0,40}?\b(?:earlier|previous|old|"
                        r"original|stated|draft|figure|v\d+)\b)", re.IGNORECASE)
# Values too vague to place a note: "v3" names a version (every "v3" chunk got the mileage note), a bare "20" matches any
# "20 minutes".
_VAGUE_VALUE = re.compile(r"(?:v|rev\.?\s*)\d+|\d{1,2}", re.IGNORECASE)
MAX_CORRECTION_VALUES = 4   # a bridge of seven figures "corrected" spread its note over most chunks of the board pack


def correction_sides(text: str, values: list[str]) -> tuple[list[str], list[str]] | None:
    """(old values, new values) of a correction statement, or ``None`` when it does not state which is which."""
    low = text.lower()
    at = {v: i for v in values if (i := low.find(v.lower())) >= 0}

    def split(cut: int, end: int, *, old_first: bool, stop: int | None = None) -> tuple[list[str], list[str]]:
        before = [v for v, i in at.items() if i < cut]
        after = [v for v, i in at.items() if end <= i < (len(text) if stop is None else stop)]
        return (before, after) if old_first else (after, before)

    candidates = []
    if m := _FROM_TO.search(text):
        candidates.append(([v for v, i in at.items() if m.start("old") <= i < m.end("old")],
                           [v for v, i in at.items() if i >= m.end()]))
    if m := _NEW_AFTER.search(text):
        candidates.append(split(m.start(), m.end(), old_first=True))
    if m := _PASSIVE.search(text):
        candidates.append(split(m.start(), m.end(), old_first=True))
    if _CORRECTION_WORD.search(text) and (m := _NOT_OLD.search(text)):
        clause = max(text.rfind(";", 0, m.start()), text.rfind(". ", 0, m.start())) + 1
        end = min((i for i in (text.find(";", m.end()), text.find(". ", m.end())) if i >= 0), default=len(text))
        candidates.append(([v for v, i in at.items() if m.end() <= i < end],
                           [v for v, i in at.items() if clause <= i < m.start()]))
    if m := _OLD_AFTER.search(text):
        candidates.append(split(m.start(), m.end(), old_first=False))
    return next(((old, new) for old, new in candidates if old and new), None)


def correction_notes(items: list[dict[str, Any]], groundings: list[str]) -> dict[int, list[str]]:
    """Chunk index (1-based) -> whole-file corrections whose old value it holds without the new one.

    A correction is a whole-file item (``agent_file``) with two to ``MAX_CORRECTION_VALUES`` values carrying digits whose
    text says which value replaced which (``correction_sides``)."""
    out: dict[int, list[str]] = {}
    for it in items:
        if it.get("origin") != "agent_file" or it.get("excluded_reason"):
            continue
        numbers = [v for v in it.get("values") or [] if any(ch.isdigit() for ch in v) and not _VAGUE_VALUE.fullmatch(v.strip())]
        sides = correction_sides(it["text"], numbers) if 2 <= len(numbers) <= MAX_CORRECTION_VALUES else None
        if sides is None:
            continue
        old, new = sides
        if _refines(old, new) or not _stated_in_source(old + new, groundings):
            continue
        for i, text in enumerate(groundings, start=1):
            if any(_holds(v, text) for v in old) and not any(_holds(v, text) for v in new):
                out.setdefault(i, []).append(it["text"])
    return out


# The whole-file pass can invent the relation ("the audit of 14 to 18 October supersedes the earlier 14 October", ISO report,
# core mine v4c): a correction counts only when a sentence of the file holding one of its values says something changed.
_CHANGE_CUE = re.compile(r"\b(?:correct\w*|not|no longer|instead|replac\w*|supersed\w*|chang\w*|mov(?:e|ed|es|ing)|updat\w*|"
                         r"amend\w*|struck|wrong|actually|earlier|previous|former|originally|draft|v\d+|rev\.?\s?\d+|"
                         r"do not use|pulled forward|postponed|delayed)\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n\s*\n")     # not every newline: PDF text wraps mid-sentence
_NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")


def _stated_in_source(values: list[str], groundings: list[str]) -> bool:
    """A sentence holding one of ``values``, or the sentence next to it ("Priya has NOT cleared 14,500. She cleared 12,750
    EUR"), says something changed."""
    for text in groundings:
        sentences = _SENTENCE_END.split(text)
        for i, sentence in enumerate(sentences):
            if any(_holds(v, sentence) for v in values) and \
                    any(_CHANGE_CUE.search(s) for s in sentences[max(0, i - 1):i + 2]):
                return True
    return False


def _refines(old: list[str], new: list[str]) -> bool:
    """Every number of the old side is also in the new one: "14 October 2024" -> "14 to 18 October 2024" adds detail, it
    corrects nothing."""
    old_n = {n for v in old for n in _NUMBER.findall(v)}
    return bool(old_n) and old_n <= {n for v in new for n in _NUMBER.findall(v)}


def _holds(value: str, text: str) -> bool:
    """``value`` as a whole token run of ``text`` (normalised): ``contains`` also matches "2 months" inside "12 months"."""
    return re.search(rf"(?<![0-9a-z]){re.escape(norm(value))}(?![0-9a-z])", norm(text)) is not None


def _values_of(items: list[dict[str, Any]], ch: Chunk) -> list[str] | None:
    """The values a joined passage was asked for; ``None`` for an ordinary chunk (every valid pair is kept)."""
    return [v for it in items for v in it.get("values") or []] if ch.span else None


@dataclass
class Chunk:
    idx: int                 # 1-based
    text: str                # as parsed
    grounding: str           # header carried from an earlier chunk + text: what answers are checked against
    prompt: str              # the form shown to the helper (keyed table rows)
    span: bool = False       # joined from several chunks for whole-file items: ask only for its checklist
    context: str = ""        # neighbouring-text block (prompts.context_block): which record a split sentence belongs to
    neighbours: str = ""     # the same text, plain: answers may name a record label that only it holds ("M5")


@dataclass
class JobContext:
    pid: str
    job: dict[str, Any]
    chat: ChatFn
    limiter: threading.Semaphore
    model: str
    workers: int
    cancel: threading.Event = field(default_factory=threading.Event)
    vision: Any = None                # models.vision.Reader shared by the batch, or None (figures keep their OCR text)


class MiningJob:
    def __init__(self, ctx: JobContext) -> None:
        self.ctx = ctx
        self.pid = ctx.pid
        self.job_id = ctx.job["id"]
        self.source_id = ctx.job["source_id"]
        self.kind = ctx.job["kind"]
        self.stats: dict[str, Any] = {"calls": 0, "call_errors": 0, "pairs_written": 0, "fix_rounds": 0}
        self._write_lock = threading.Lock()
        self._first_error = ""
        self._round_meta: dict[str, tuple[str, float]] = {}   # round id -> (error, seconds) until the round is finished

    # ── public ──────────────────────────────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        from finetune_studio import db

        started = time.time()
        db.mining.update_job(self.job_id, status="running", started_at=started)
        try:
            self._run()
        except JobCancelled:
            db.mining.update_job(self.job_id, status="cancelled", stage="done", finished_at=time.time(),
                                 message="Cancelled.", stats_json=self._stats(started))
        except Exception as exc:
            log.exception("mining job %s failed", self.job_id)
            db.mining.update_job(self.job_id, status="failed", stage="done", finished_at=time.time(),
                                 error=str(exc)[:2000], message=f"Failed: {exc}", stats_json=self._stats(started))

    # ── pipeline ────────────────────────────────────────────────────────────────────────────────────────────────

    def _run(self) -> None:
        from finetune_studio import db
        from finetune_studio.knowledge import sources

        started = time.time()
        source = sources.get_source(self.pid, self.source_id)
        if source is None:
            raise RuntimeError(f"source {self.source_id} not found")
        source = sources.ensure_parsed(self.pid, source.id)
        if not source.ready:
            raise RuntimeError(f"{source.filename}: {source.error or 'the parser produced no text to mine'}")
        if self.kind == "mine":
            source = self._figures(source)
        texts = sources.passages(self.pid, source.sha256)
        if not texts:
            raise RuntimeError(f"{source.filename}: the parser produced no text to mine")
        texts = strip_overlap(self._neutral(self._hold_back_pictures(source, texts)))
        self.source = source
        self.title = document_title(texts[0], source.filename)
        self.doc_ids = document_identifiers(self.title, texts[0])
        self.chunks = []
        for i, text in enumerate(texts):
            carried = table_header_before(texts, i)
            before, after = neighbour_text(texts, i)
            self.chunks.append(Chunk(i + 1, text, f"{carried}\n{text}" if carried else text, keyed_chunk(text, carried),
                                     context=prompts.context_block(before, after), neighbours=f"{before}\n{after}"))

        if self.kind == "mine":
            self._stage("inventory", f"Listing the data points of {len(self.chunks)} chunk(s)…")
            items = self._inventory(texts)
            self.stats["items"] = len(items)
            self._add_corrections(items)
            self._stage("mining", f"Mining {len(items)} data points…")
            self._mine(items)
        items = db.mining.list_items(self.pid, self.source_id)
        if not items:
            raise RuntimeError("the file has no inventory: mine it first")
        if self.kind != "mine":
            self._add_corrections(items)
        open_ids = self._fix_loop(items)
        if self.stats["calls"] and self.stats["call_errors"] == self.stats["calls"]:
            raise RuntimeError(f"every helper call failed: {self._first_error}")
        status = "gaps" if open_ids else "done"
        n_prop = int(self.stats.get("proposed") or 0)
        msg = (f"{len(open_ids)} data point(s) still open" + (f", {n_prop} of them proposed for exclusion" if n_prop else "")
               + ": write a pair, exclude them or confirm the proposals." if open_ids
               else "Every data point is stated by a pair. Review the pending pairs.")
        self.stats["open"] = len(open_ids)
        if self.stats.get("figures_held_back"):
            msg += (f" {self.stats['figures_held_back']} picture(s) (charts, table images) were not mined: OCR alone"
                    " mixes up their labels and values. Set a Vision reader in Settings, then Reset and Mine the file.")
        db.mining.update_job(self.job_id, status=status, stage="done", finished_at=time.time(), message=msg,
                             stats_json=self._stats(started))

    def _figures(self, source: Source) -> Source:
        """Stage 0: the Vision reader replaces the OCR text of the file's figures (``mining/figures.py``)."""
        from finetune_studio.data.mining.figures import (
            pending_blocks,
            read_file_figures,
        )
        from finetune_studio.knowledge import sources

        pending = pending_blocks(sources.parsed_text(self.pid, source))
        if not pending:
            return source
        if self.ctx.vision is None:
            return source                                   # pictures are held back below; scanned pages keep OCR
        self._stage("figures", f"Reading {len(pending)} figure(s) with {self.ctx.vision.label}…")
        res = read_file_figures(self.pid, source, self.ctx.vision, self._progress)
        self.stats["figures_read"] = res.read
        if res.failed:
            self.stats["figure_errors"] = res.failed[:20]
        return (sources.get_source(self.pid, source.id) or source) if res.changed else source

    def _hold_back_pictures(self, source: Source, texts: list[str]) -> list[str]:
        """Pictures still holding only OCR text (no Vision reader, or its reading failed) are not mined: their OCR keeps
        the numbers but not which label each belongs to. The stored chunks stay as they are (the reader sees the OCR)."""
        from finetune_studio.data.mining.figures import hold_back_pictures, ocr_pictures
        from finetune_studio.knowledge import sources

        pictures = ocr_pictures(sources.parsed_text(self.pid, source))
        if not pictures:
            return texts
        self.stats["figures_held_back"] = len(pictures)
        return [hold_back_pictures(t, pictures)[0] for t in texts]

    @staticmethod
    def _neutral(texts: list[str]) -> list[str]:
        from finetune_studio.data.mining.figures import neutral_markers
        return [neutral_markers(t) for t in texts]

    def _inventory(self, texts: list[str]) -> list[dict[str, Any]]:
        from finetune_studio import db

        rules = rule_items(texts)
        rules_by_chunk: dict[int, list] = {}
        for r in rules:
            rules_by_chunk.setdefault(r.chunk_idx, []).append(r)

        def per_chunk(ch: Chunk) -> list:
            reply, rid = self._call("inventory", ch.idx, prompts.inventory_messages(self.title, ch.prompt, ch.context))
            got = agent_items(ground(parse_inventory(reply), ch.grounding), ch.idx, rules_by_chunk.get(ch.idx, []))
            db.mining.finish_round(rid, reply=reply, error=self._round_error(rid), seconds=self._round_seconds(rid),
                                    parsed=len(parse_inventory(reply)), items_added=len(got))
            return got

        agent = [it for got in self._parallel(per_chunk, self.chunks, "Inventory") for it in got]
        whole = "\n\n".join(texts)
        file_items: list = []
        if len(whole) <= FILE_PASS_MAX_CHARS:
            reply, rid = self._call("inventory_file", 0, prompts.file_inventory_messages(self.title, whole))
            entries = ground(parse_inventory(reply), whole)
            for e in entries:
                idx = chunk_of(e["values"], texts)
                file_items += agent_items([e], idx, rules_by_chunk.get(idx, []), origin="agent_file")
            db.mining.finish_round(rid, reply=reply, error=self._round_error(rid), seconds=self._round_seconds(rid),
                                   parsed=len(parse_inventory(reply)), items_added=len(file_items))
        all_items = [*rules, *agent, *file_items]
        self.stats.update(rule_items=len(rules), agent_items=len(agent), file_items=len(file_items))
        db.mining.insert_items(self.pid, self.source_id, self.job_id, [it.as_row() for it in all_items])
        return db.mining.list_items(self.pid, self.source_id)

    def _mine(self, items: list[dict[str, Any]]) -> None:
        seen = self._seen_questions()

        def per_unit(unit: tuple[Chunk, list[dict[str, Any]]]) -> None:
            ch, todo = unit
            checklist = [item_prompt_text(it) for it in todo]
            # a joined passage repeats whole chunks: the full mining prompt ("cover every fact") mined them all again
            messages = (prompts.fix_messages(self.title, ch.prompt, checklist, context=ch.context) if ch.span
                        else prompts.mine_messages(self.title, ch.prompt, checklist, ch.context))
            self._pairs_round("mine", ch, messages, [it["id"] for it in todo], seen, only_values=_values_of(todo, ch))

        self._parallel(per_unit, self._units(items), "Mining")

    def _fix_loop(self, items: list[dict[str, Any]]) -> list[str]:
        from finetune_studio.data.fs import qa as qa_fs

        seen = self._seen_questions()
        proposed = {it["id"] for it in items if it.get("proposed_reason")}
        open_ids = self._open_items(items)
        for n in range(1, MAX_FIX_ROUNDS + 1):
            todo_ids = [i for i in open_ids if i not in proposed]     # a proposed exclusion waits for the reviewer
            if not todo_ids:
                break
            self._stage("fixing", f"Fix round {n}: {len(todo_ids)} data point(s) not stated yet…")
            self.stats["fix_rounds"] = n
            open_items = [it for it in items if it["id"] in todo_ids]
            rejected = self._rejected_by_chunk(qa_fs.list_qa_pairs(self.pid, source_id=self.source_id, status="rejected"))

            def per_unit(unit: tuple[Chunk, list[dict[str, Any]]], rejected=rejected) -> None:
                ch, todo = unit
                messages = prompts.fix_messages(self.title, ch.prompt, [item_prompt_text(it) for it in todo],
                                                rejected.get(ch.idx, []), ch.context)
                skipped = self._pairs_round("fix", ch, messages, [it["id"] for it in todo], seen,
                                            only_values=_values_of(todo, ch))
                proposed.update(skipped)

            before = len(proposed)
            self._parallel(per_unit, self._units(open_items), f"Fix round {n}")
            still = self._open_items(items)
            if len(still) >= len(open_ids) and len(proposed) == before:
                open_ids = still
                break                                    # no progress: more rounds of the same prompt will not help
            open_ids = still
        self.stats["proposed"] = len(proposed & set(open_ids))
        self._stage("checking", "Checking the result…")
        return open_ids

    # ── helpers ─────────────────────────────────────────────────────────────────────────────────────────────────

    def _pairs_round(self, kind: str, ch: Chunk, messages: list[dict[str, str]], item_ids: list[str],
                     seen: set[str], only_values: list[str] | None = None) -> list[str]:
        """Write the reply's valid pairs; in a fix round also record its skips as proposals. Returns the proposed item ids.

        ``only_values``: keep only pairs whose answer states one of them (a joined passage: the helper also wrote pairs for
        every other fact it saw, ~10 per call, restating chunks that were mined already)."""
        from finetune_studio import db
        from finetune_studio.data.fs import qa as qa_fs

        reply, rid = self._call(kind, ch.idx, messages, item_ids)
        raw = parse_qa_json(reply, 400) if reply else []
        written = 0
        with self._write_lock:
            batch = validate_qa_batch(raw, f"{ch.grounding}\n{ch.neighbours}\n{self.doc_ids}", seen_questions=seen)
            existing = [{"q": p["question"], "a": p["answer"]} for p in qa_fs.list_qa_pairs(self.pid, source_id=self.source_id)
                        if p.get("status") != "rejected" and int(p.get("chunk_idx") or 0) == ch.idx]
            accepted = [p for p in batch.accepted
                        if only_values is None or any(loosely_contains(v, p.answer) for v in only_values)]
            keep = drop_redundant([{"q": p.question, "a": p.answer} for p in accepted], existing)
            wanted = {(k["q"], k["a"]) for k in keep}
            now = time.time()
            for pv in accepted:
                if (pv.question, pv.answer) not in wanted:
                    continue
                prov = Provenance(source_id=self.source_id, sha256=self.source.sha256,
                                  filename=self.source.filename, chunk_idx=ch.idx, generator="mining")
                qa_fs.write_qa_pair(self.pid, build_qa_record(
                    qa_id=uuid.uuid4().hex[:12], pair=pv, provenance=prov, chunk_text=ch.text[:1500],
                    score=heuristic_score(pv.question, pv.answer, ch.text), created_at=now, origin=kind,
                    extra={"job_id": self.job_id, "round_id": rid}))
                written += 1
            self.stats["pairs_written"] += written
        skips = {item_ids[i]: why for i, why in parse_skips(reply, len(item_ids)).items()} if kind == "fix" else {}
        if skips:
            db.mining.propose_exclusions(self.pid, skips)
        db.mining.finish_round(rid, reply=reply, error=self._round_error(rid), seconds=self._round_seconds(rid),
                               parsed=len(raw), accepted=written)
        return list(skips)

    def _call(self, kind: str, chunk_idx: int, messages: list[dict[str, str]],
              item_ids: list[str] | None = None) -> tuple[str, str]:
        """One helper call through the global limiter. Returns (reply, round_id); a failed call returns ``""``."""
        from finetune_studio import db

        if self.ctx.cancel.is_set():
            raise JobCancelled()
        rid = db.mining.add_round(project_id=self.pid, source_id=self.source_id, job_id=self.job_id,
                                  chunk_idx=chunk_idx, kind=kind, model=self.ctx.model, messages=messages,
                                  item_ids=item_ids)
        with self.ctx.limiter:
            if self.ctx.cancel.is_set():
                raise JobCancelled()
            t0 = time.time()
            try:
                reply = self.ctx.chat(messages, max_tokens=MAX_TOKENS, temperature=0.0, top_p=1.0)
                err = ""
            except Exception as exc:  # noqa: BLE001 - one failed call is a recorded round, the fix rounds retry its items
                reply, err = "", f"{type(exc).__name__}: {exc}"
            seconds = time.time() - t0
        with self._write_lock:
            self.stats["calls"] += 1
            if err:
                self.stats["call_errors"] += 1
                self._first_error = self._first_error or err
        self._round_meta[rid] = (err, seconds)
        return reply or "", rid

    def _round_error(self, rid: str) -> str:
        return self._round_meta.get(rid, ("", 0.0))[0]

    def _round_seconds(self, rid: str) -> float:
        return self._round_meta.get(rid, ("", 0.0))[1]

    def _add_corrections(self, items: list[dict[str, Any]]) -> None:
        """Give every chunk that holds an old value the whole-file correction of it (prompt context + grounding: an answer
        naming the new value must not be rejected as invented)."""
        notes = correction_notes(items, [c.grounding for c in self.chunks])
        for ch in self.chunks:
            if notes.get(ch.idx):
                ch.context += prompts.corrections_block(notes[ch.idx])
                ch.neighbours += "\n" + "\n".join(notes[ch.idx])
        self.stats["corrections"] = len({n for got in notes.values() for n in got})

    def _units(self, items: list[dict[str, Any]]) -> list[tuple[Chunk, list[dict[str, Any]]]]:
        """One helper call per (passage, items): chunk items go with their chunk; a whole-file item goes with a passage
        joined from every chunk that holds one of its values (shared by the items that need the same chunks).

        A whole-file fact ("CHG-2291 set 6000000 ms, CHG-2294 corrected it to 6000") spans chunks; shown and checked against
        its anchor chunk alone, its pair was rejected as invented in every round."""
        local: dict[int, list[dict[str, Any]]] = {}
        spans: dict[tuple[int, ...], list[dict[str, Any]]] = {}
        for it in items:
            anchor = int(it["chunk_idx"])
            if it.get("origin") != "agent_file":
                local.setdefault(anchor, []).append(it)
                continue
            idxs = {anchor} | {c.idx for c in self.chunks for v in it.get("values") or [] if contains(v, c.grounding)}
            spans.setdefault(tuple(sorted(i for i in idxs if 1 <= i <= len(self.chunks))) or (anchor,), []).append(it)
        units = [(c, local[c.idx]) for c in self.chunks if c.idx in local]
        spans = self._pack(spans)
        for idxs, todo in spans.items():
            if len(idxs) == 1:
                units.append((self.chunks[idxs[0] - 1], todo))
                continue
            parts = [self.chunks[i - 1] for i in idxs]
            units.append((Chunk(int(todo[0]["chunk_idx"]) or idxs[0], "\n\n".join(c.text for c in parts),
                                "\n\n".join(c.grounding for c in parts), "\n\n[...]\n\n".join(c.prompt for c in parts), span=True),
                          todo))
        return units

    def _pack(self, spans: dict[tuple[int, ...], list[dict[str, Any]]]) -> dict[tuple[int, ...], list[dict[str, Any]]]:
        """Merge spans greedily (in chunk order) while the joined prompt stays within ``SPAN_PROMPT_CHARS``: one call per
        item group instead of one per item (run 11: 138 joined-passage calls for 19 files)."""
        size = {c.idx: len(c.prompt) for c in self.chunks}
        out: dict[tuple[int, ...], list[dict[str, Any]]] = {}
        cur: set[int] = set()
        group: list[dict[str, Any]] = []
        for idxs, todo in sorted(spans.items()):
            merged = cur | set(idxs)
            if group and sum(size.get(i, 0) for i in merged) > SPAN_PROMPT_CHARS:
                out.setdefault(tuple(sorted(cur)), []).extend(group)
                merged, group = set(idxs), []
            cur = merged
            group.extend(todo)
        if group:
            out.setdefault(tuple(sorted(cur)), []).extend(group)
        return out

    def _parallel(self, fn: Callable[[Any], Any], chunks: list[Any], label: str) -> list[Any]:
        """Run ``fn`` over chunks or work units, ``workers`` at a time (calls are bounded by the global limiter anyway)."""
        out: list[Any] = []
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, self.ctx.workers)) as pool:
            for result in pool.map(fn, chunks):
                out.append(result)
                done += 1
                self._progress(f"{label}: {done}/{len(chunks)} chunk(s)")
        return out

    def _open_items(self, items: list[dict[str, Any]]) -> list[str]:
        from finetune_studio.data.fs import qa as qa_fs

        states = item_states(items, qa_fs.list_qa_pairs(self.pid, source_id=self.source_id))
        return [s.item_id for s in states if s.state == "open"]

    def _seen_questions(self) -> set[str]:
        from finetune_studio.data.fs import qa as qa_fs

        return {normalize_question(str(p.get("question", ""))) for p in qa_fs.list_qa_pairs(self.pid, source_id=self.source_id)}

    @staticmethod
    def _rejected_by_chunk(pairs: list[dict[str, Any]]) -> dict[int, list[tuple[str, str, str]]]:
        out: dict[int, list[tuple[str, str, str]]] = {}
        for p in pairs:
            out.setdefault(int(p.get("chunk_idx") or 0), []).append(
                (str(p.get("question", "")), str(p.get("answer", "")), str(p.get("note") or "")))
        return out

    def _stage(self, stage: str, message: str) -> None:
        from finetune_studio import db

        db.mining.update_job(self.job_id, stage=stage, message=message)

    def _progress(self, message: str) -> None:
        """Best-effort progress text: a busy database must not fail the job over a status line."""
        import sqlite3

        from finetune_studio import db

        try:
            db.mining.update_job(self.job_id, message=message)
        except sqlite3.OperationalError:
            log.warning("mining job %s: progress update skipped (database busy)", self.job_id)

    def _stats(self, started: float) -> dict[str, Any]:
        return {**self.stats, "seconds": round(time.time() - started, 1)}
