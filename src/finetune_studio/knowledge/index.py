"""The project's Smart Brain index, built from the same parse training uses (``knowledge.sources``).

Every ready source goes in as one document (id = source id) indexed by its **passages**, the same ones mining reads
(``files/<sha12>/chunks/``): one chunking for the whole app. Vision readings and hand edits included (``set_text``
rebuilds the passages), nothing parsed or chunked a second time. Measured 2026-10-14 on Korvane (102 quiz questions,
retrieval only): all expected values in the top 5 for 95/102 with the passages vs 89/102 with the index's own
400-token chunks; every passage fits the embedder's 512-token window. ``manifest.extra.built_from`` records the text
hash of every source it read, so ``state`` can say which files changed (or arrived) since the last build.

Fact cards (opt-in, ``fact_cards``): every *approved* Q&A pair is indexed as its own short row next to the passages
of its file ("Q: …\\nA: …"), so a question can land on a reviewed fact as well as on the passage it came from. The
setting is kept in the manifest (a rebuild keeps it) and ``facts_sha`` says when the reviewed facts changed.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
from dataclasses import dataclass, field
from typing import Any

from finetune_studio import db
from finetune_studio.data.fs.paths import brain_index_dir
from finetune_studio.knowledge import sources
from finetune_studio.knowledge.errors import IndexBuildError

log = logging.getLogger(__name__)

FACTS_SUFFIX = "~facts"          # document id of a file's fact cards: <source_id>~facts


@dataclass
class IndexState:
    built: bool
    indexed: list[str] = field(default_factory=list)     # source ids in the index with the text they have now
    stale: list[str] = field(default_factory=list)       # in the index, but their text changed since (edit, Vision, reparse)
    missing: list[str] = field(default_factory=list)     # ready sources the index has never seen
    removed: list[str] = field(default_factory=list)     # in the index, but the file is gone
    fact_cards: bool = False                             # the last build indexed the approved pairs as fact cards
    facts_changed: bool = False                          # ... and pairs were approved, edited or un-approved since

    @property
    def fresh(self) -> bool:
        return self.built and not (self.stale or self.missing or self.removed or self.facts_changed)

    def as_dict(self) -> dict[str, Any]:
        return {"built": self.built, "fresh": self.fresh, "indexed": self.indexed, "stale": self.stale,
                "missing": self.missing, "removed": self.removed, "fact_cards": self.fact_cards,
                "facts_changed": self.facts_changed}


def _approved_facts(pid: str) -> dict[str, list[tuple[str, str, str]]]:
    """``source_id -> [(pair_id, question, answer)]`` of the approved pairs, in a stable order."""
    from finetune_studio.data.fs import qa as qa_fs

    out: dict[str, list[tuple[str, str, str]]] = {}
    for p in sorted(qa_fs.list_qa_pairs(pid, status="approved"), key=lambda p: str(p.get("id"))):
        q, a = str(p.get("question") or "").strip(), str(p.get("answer") or "").strip()
        if q and a:
            out.setdefault(str(p.get("source_id") or ""), []).append((str(p.get("id")), q, a))
    return out


def facts_sha(pid: str) -> str:
    """One hash over every approved pair (id, question, answer): changes when a fact is approved, edited or dropped."""
    h = hashlib.sha256()
    for sid, facts in sorted(_approved_facts(pid).items()):
        for fid, q, a in facts:
            h.update(f"{sid}\x00{fid}\x00{q}\x00{a}\x01".encode())
    return h.hexdigest()


def documents(pid: str, *, fact_cards: bool = False) -> list[Any]:
    """Every ready source of the project as an index document (+ its approved pairs as fact cards when asked)."""
    from finetune_studio.smart_brain.schema import IndexDocument

    facts = _approved_facts(pid) if fact_cards else {}
    out = []
    for s in sources.list_sources(pid):
        if not s.ready:
            continue
        text = sources.parsed_text(pid, s)
        path = str(sources.source_dir(pid, s) / "parsed.txt")
        passages = tuple(p for p in sources.passages(pid, s.sha256) if p.strip())
        if passages:
            out.append(IndexDocument(s.id, s.filename, path, text, chunks=passages))
        cards = tuple(f"Q: {q}\nA: {a}" for _fid, q, a in facts.get(s.id, ()))
        if cards:
            out.append(IndexDocument(s.id + FACTS_SUFFIX, f"{s.filename} · reviewed facts", path, "\n\n".join(cards),
                                     chunks=cards))
    return out


def _manifest_extra(pid: str) -> dict[str, Any] | None:
    from finetune_studio.smart_brain.io import read_json

    manifest = brain_index_dir(pid) / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        return read_json(manifest).get("extra") or {}
    except (OSError, ValueError):
        log.warning("index manifest unreadable for %s", pid, exc_info=True)
        return None


def built_from(pid: str) -> dict[str, str] | None:
    """``source_id -> text hash`` the index was built from; None when there is no index."""
    extra = _manifest_extra(pid)
    if extra is None:
        return None
    return {str(k): str(v) for k, v in (extra.get("built_from") or {}).items()}


def state(pid: str) -> IndexState:
    extra = _manifest_extra(pid)
    if extra is None:
        return IndexState(built=False, missing=[s.id for s in sources.list_sources(pid) if s.ready])
    built = {str(k): str(v) for k, v in (extra.get("built_from") or {}).items()}
    st = IndexState(built=True, fact_cards=bool(extra.get("fact_cards")))
    ready = {s.id: s for s in sources.list_sources(pid) if s.ready}
    for sid, s in ready.items():
        if sid not in built:
            st.missing.append(sid)
        elif built[sid] != s.text_sha:
            st.stale.append(sid)
        else:
            st.indexed.append(sid)
    st.removed = [sid for sid in built if sid not in ready]
    if st.fact_cards:
        st.facts_changed = str(extra.get("facts_sha") or "") != facts_sha(pid)
    return st


def prepare(pid: str, *, fact_cards: bool | None = None) -> dict[str, Any]:
    """Check a build can run and record it (a ``queued`` build row). Raises IndexBuildError (404 / 400).

    Returns ``build_id``, ``queued_files`` (parsed files to index) and the resolved ``fact_cards`` choice
    (None keeps what the last build used; off for a first build). ``run`` does the work (a job process in the web
    app: ``webui.brain_build_jobs``).
    """
    project = db.get_project(pid)
    if not project:
        raise IndexBuildError(f"project {pid} not found", 404)
    if fact_cards is None:
        fact_cards = bool((_manifest_extra(pid) or {}).get("fact_cards"))
    docs = documents(pid, fact_cards=fact_cards)
    queued = sum(1 for d in docs if not d.document_id.endswith(FACTS_SUFFIX))
    if not queued:
        raise IndexBuildError("no parsed files in this project yet: upload files and let them parse first", 400)
    corpus = brain_index_dir(pid)
    rows = db.list_brain_indexes(pid)
    index_id = (rows[0] if rows else db.create_brain_index(pid, project["name"], store_path=str(corpus)))["id"]
    run_row = db.create_brain_build(pid, index_id)
    return {"build_id": run_row["id"], "queued_files": queued, "fact_cards": fact_cards}


def run(pid: str, build_id: str, *, embedder: str | None = None, reset: bool = False,
        fact_cards: bool = False) -> dict[str, Any]:
    """Embed every ready source into the project index for a ``prepare``d build row; the row ends done or error
    (with the real error, then re-raised)."""
    from finetune_studio.smart_brain import DEFAULT_EMBEDDER, BrainIndex

    db.mark_brain_build_running(build_id)
    try:
        project = db.get_project(pid)
        if not project:
            raise IndexBuildError(f"project {pid} not found", 404)
        docs = documents(pid, fact_cards=fact_cards)
        corpus = brain_index_dir(pid)
        if reset and corpus.exists():
            shutil.rmtree(corpus)
        hashes = {s.id: s.text_sha for s in sources.list_sources(pid) if s.ready}
        extra = {"built_from": {d.document_id: hashes.get(d.document_id, "") for d in docs
                                if not d.document_id.endswith(FACTS_SUFFIX)},
                 "fact_cards": fact_cards, "facts_sha": facts_sha(pid) if fact_cards else ""}
        result = BrainIndex(corpus).build_from_documents(
            docs, name=project["name"], embedder=embedder or DEFAULT_EMBEDDER, chunk_size=0, overlap=0,
            splitter="passages", extra=extra)
    except Exception as e:
        log.exception("index build failed for %s", pid)
        db.mark_brain_build_failed(build_id, error=str(e) or type(e).__name__)
        raise
    n_docs, n_chunks = int(result.get("documents") or 0), int(result.get("chunks") or 0)
    db.mark_brain_build_done(build_id, doc_count=n_docs, chunk_count=n_chunks)
    db.ensure_brain_index(pid, str(corpus), name=project["name"], doc_count=n_docs, chunk_count=n_chunks)
    return {**result, "corpus_dir": str(corpus), "build_id": build_id, "fact_cards": fact_cards,
            "fact_card_count": sum(len(d.chunks) for d in docs if d.document_id.endswith(FACTS_SUFFIX))}


def build(pid: str, *, embedder: str | None = None, reset: bool = False,
          fact_cards: bool | None = None) -> dict[str, Any]:
    """(Re)build the project index from every ready source, here and now (the CLI). Tracked as a build row.

    ``fact_cards`` None keeps what the last build used (off for a first build).
    """
    plan = prepare(pid, fact_cards=fact_cards)
    return run(pid, plan["build_id"], embedder=embedder, reset=reset, fact_cards=plan["fact_cards"])
