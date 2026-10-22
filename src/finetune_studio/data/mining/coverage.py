"""Which inventory items the pairs of a file state, and what that makes the file.

An item is checked against the file's pairs (any chunk: a reviewer may anchor a pair elsewhere):

- rule item (statement): ``statements.covering_pairs``; its distinctive tokens must appear in the answers of the pairs about it;
- agent item: every value is stated (``match.loosely_contains``: numbers exact, 80 % of the words) by the answers of the pairs
  about it: the pairs whose answer states at least half of the item's values (never more than two of them), or states one
  value and shares enough content words with the item (the test rule items use, ``statements.covering_pairs``). One fact
  per pair is the mining rule, so a fact that combines several values ("327 FTE = 118 + 61 + 49 + ...") is legitimately spread
  over several pairs; requiring half of eight values in one answer left such items open forever, and a lone "moved to
  Thu 03.10" answer is about the item when it shares its words.

Per item the state is ``approved`` (approved pairs alone state it), ``pending`` (only with unreviewed pairs), ``open``
(nothing states it) or ``excluded`` (a reviewer decided it is not a trainable fact). Rejected pairs never count. An open item
the helper judged not to be a fact carries ``proposed`` (its reason); it stays open until a reviewer confirms the exclusion.

File status (``status``): ``parsed`` (no inventory yet), ``mining`` (a job is working on it), ``gaps`` (open items),
``review`` (items stated only by unreviewed pairs), ``complete`` (every item approved or excluded). Only ``complete`` files are
training-ready and pass the export check.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from finetune_studio.data.mining.match import Haystack, haystack, loosely_contains
from finetune_studio.data.mining.statements import (
    PairTokens,
    covering_pairs,
    hard_tokens,
    pair_tokens,
)
from finetune_studio.data.prep.qa_validate import content_tokens


@dataclass
class ItemState:
    item_id: str
    state: str                                  # approved | pending | open | excluded
    pair_ids: list[str] = field(default_factory=list)
    proposed: str = ""                          # open items only: the helper's reason to exclude it


@dataclass
class FileCoverage:
    source_id: str
    items: list[ItemState]
    status: str
    pairs: dict[str, int]

    @property
    def counts(self) -> dict[str, int]:
        out = {"items": len(self.items), "approved": 0, "pending": 0, "open": 0, "excluded": 0, "proposed": 0}
        for it in self.items:
            out[it.state] += 1
            out["proposed"] += bool(it.proposed)
        return out

    def as_dict(self) -> dict[str, Any]:
        c = self.counts
        decided = c["approved"] + c["excluded"]
        return {"source_id": self.source_id, "status": self.status, "counts": c, "pairs": self.pairs,
                "coverage_pct": round(100.0 * decided / c["items"], 1) if c["items"] else 0.0,
                "training_ready": self.status == "complete"}


def _covering(item: dict[str, Any], prepared: Sequence[PairTokens], answers: Sequence[tuple[str, Haystack]],
              content: dict[str, frozenset[str]]) -> list[str]:
    if item["origin"] == "rule":
        # Tokens from the text, not the stored values: a tokenizer fix must apply to inventories mined before it.
        return covering_pairs(item["text"], hard_tokens(item["text"]), prepared)
    values = item["values"]
    need = min(2, (len(values) + 1) // 2)
    own = content_tokens(item["text"]) - content_tokens(" ".join(values))   # words the values share prove nothing
    need_overlap = max(1, min(3, round(0.4 * len(own))))
    hits = {}
    for pair_id, answer in answers:
        found = {v for v in values if loosely_contains(v, answer)}
        if len(found) >= need or (found and len(own & content.get(pair_id, frozenset())) >= need_overlap):
            hits[pair_id] = found
    stated = set().union(*hits.values()) if hits else set()
    return list(hits) if stated >= set(values) else []


def item_states(items: Sequence[dict[str, Any]], pairs: Sequence[dict[str, Any]]) -> list[ItemState]:
    approved = [p for p in pairs if p.get("status") == "approved"]
    live = [p for p in pairs if p.get("status") in ("approved", "pending")]
    prep_a = [pair_tokens(str(p["id"]), str(p.get("question", "")), str(p.get("answer", ""))) for p in approved]
    prep_l = [pair_tokens(str(p["id"]), str(p.get("question", "")), str(p.get("answer", ""))) for p in live]
    hay = {str(p["id"]): haystack(str(p.get("answer", ""))) for p in live}      # normalised once, checked per item
    ans_a = [(str(p["id"]), hay[str(p["id"])]) for p in approved]
    ans_l = [(str(p["id"]), hay[str(p["id"])]) for p in live]
    content_a = {p.pair_id: p.content for p in prep_a}
    content_l = {p.pair_id: p.content for p in prep_l}
    out: list[ItemState] = []
    for it in items:
        if it["origin"] == "rule" and not hard_tokens(it["text"]):
            continue                    # nothing checkable left (a clause number only, under an older tokenizer): no data point
        if it.get("excluded_reason"):
            out.append(ItemState(it["id"], "excluded"))
            continue
        by_approved = _covering(it, prep_a, ans_a, content_a)
        if by_approved:
            out.append(ItemState(it["id"], "approved", by_approved))
            continue
        by_live = _covering(it, prep_l, ans_l, content_l)
        out.append(ItemState(it["id"], "pending", by_live) if by_live
                   else ItemState(it["id"], "open", proposed=str(it.get("proposed_reason") or "")))
    return out


# Coverage is items x answers fuzzy matching (a 19-file project: ~900k checks, seconds), asked for by every page that lists
# files. The result depends only on the item and pair fields below, so it is kept per file until one of them changes.
_STATES_CACHE: OrderedDict[str, list[ItemState]] = OrderedDict()
_STATES_LOCK = threading.Lock()
_STATES_MAX = 512


def _fingerprint(items: Sequence[dict[str, Any]], pairs: Sequence[dict[str, Any]]) -> str:
    h = hashlib.blake2b(digest_size=16)
    for it in items:
        h.update(repr((it.get("id"), it.get("origin"), it.get("text"), it.get("values"), it.get("excluded_reason"),
                       it.get("proposed_reason"))).encode())
    h.update(b"\x00")
    for p in pairs:
        h.update(repr((p.get("id"), p.get("status"), p.get("question"), p.get("answer"))).encode())
    return h.hexdigest()


def cached_item_states(items: Sequence[dict[str, Any]], pairs: Sequence[dict[str, Any]]) -> list[ItemState]:
    """``item_states``, reused while the file's items and pairs are unchanged (copies: callers may not share the lists)."""
    key = _fingerprint(items, pairs)
    with _STATES_LOCK:
        hit = _STATES_CACHE.get(key)
        if hit is not None:
            _STATES_CACHE.move_to_end(key)
    if hit is None:
        hit = item_states(items, pairs)
        with _STATES_LOCK:
            _STATES_CACHE[key] = hit
            while len(_STATES_CACHE) > _STATES_MAX:
                _STATES_CACHE.popitem(last=False)
    return [ItemState(s.item_id, s.state, list(s.pair_ids), s.proposed) for s in hit]


def file_status(states: Sequence[ItemState], *, has_inventory: bool, mining: bool) -> str:
    if mining:
        return "mining"
    if not has_inventory:
        return "parsed"
    kinds = {s.state for s in states}
    if "open" in kinds:
        return "gaps"
    if "pending" in kinds:
        return "review"
    return "complete"


def file_coverage(pid: str, source_id: str, *, items: Sequence[dict[str, Any]] | None = None,
                  pairs: Sequence[dict[str, Any]] | None = None, mining: bool | None = None) -> FileCoverage:
    from finetune_studio import db
    from finetune_studio.data.fs import qa as qa_fs

    items = db.mining.list_items(pid, source_id) if items is None else items
    pairs = qa_fs.list_qa_pairs(pid, source_id=source_id) if pairs is None else pairs
    if mining is None:
        mining = bool(db.mining.list_jobs(pid, source_id=source_id, active=True, limit=1))
    states = cached_item_states(items, pairs)
    counts = {"total": len(pairs), "approved": 0, "pending": 0, "rejected": 0}
    for p in pairs:
        if p.get("status") in counts:
            counts[p["status"]] += 1
    return FileCoverage(source_id, states, file_status(states, has_inventory=bool(items), mining=mining), counts)


def project_coverage(pid: str, source_ids: Sequence[str] | None = None) -> list[FileCoverage]:
    """Coverage of every parsed file of the project (or of ``source_ids``), with one read of items, pairs and jobs."""
    from finetune_studio import db
    from finetune_studio.data.fs import qa as qa_fs
    from finetune_studio.knowledge.sources import list_sources

    wanted = set(source_ids) if source_ids is not None else None
    ids = [s.id for s in list_sources(pid) if wanted is None or s.id in wanted]
    items_by: dict[str, list[dict[str, Any]]] = {}
    for it in db.mining.list_items(pid):
        items_by.setdefault(it["source_id"], []).append(it)
    pairs_by: dict[str, list[dict[str, Any]]] = {}
    for p in qa_fs.list_qa_pairs(pid):
        pairs_by.setdefault(str(p.get("source_id")), []).append(p)
    busy = {j["source_id"] for j in db.mining.list_jobs(pid, active=True)}
    return [file_coverage(pid, sid, items=items_by.get(sid, []), pairs=pairs_by.get(sid, []), mining=sid in busy)
            for sid in ids]
