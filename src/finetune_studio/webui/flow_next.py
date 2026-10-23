"""State-aware "Next" step for the project flow bar (base.html).

The bar's table is per page; on the pages whose next step depends on the project's state (Files, Pairs, Training) this
picks it from the project's counts instead, so an empty project is not told to "Review question-answer pairs" and a Pairs
page with nothing approved is not told to train (agent walkthrough 2026-10-17). ``href`` "" means the step is on the
current page: the bar shows it as text, not a link to itself. Costs ~25 ms on a 4,000-pair project.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

STATEFUL_PAGES = frozenset({"files", "data", "pairs", "data-prep", "training"})

# Each page's Next step once its own state-aware step (``next_for``) is done. One table for base.html and the API, so a
# bar refreshed in place (``GET /api/projects/{pid}/flow-next``) can fall back to it as well.
STATIC_NEXT: dict[str, tuple[str, str, str]] = {
    "files": ("/data-prep", "Review question-answer pairs", "For training, mine and approve examples from your files."),
    "pairs": ("/training", "Train an adapter", "Use the reviewed dataset to teach the model how to answer."),
    "training": ("/testing", "Test the answers", "Run a quiz to see what the adapter learned and when it declines."),
    "testing": ("/export", "Export or use the model", "Package a finished adapter, then try it in chat."),
    "export": ("/chat", "Try it in chat", "Test the exported model; add Smart Brain when answers need document facts."),
    "smart brain": ("/chat", "Ask with sources", "Chat uses the indexed files and shows where each answer came from."),
    "chat": ("/testing", "Measure answer quality", "Run a quiz to check accuracy and appropriate declines."),
    "compare": ("/testing", "Run a focused quiz", "Compare models on the same questions, then score the answers."),
    "benchmarks": ("/testing", "Test your own questions", "Public benchmarks are useful context; your quiz measures your task."),
    "models": ("/chat", "Open a model in chat", "Use a model in this project and optionally ground it with Smart Brain."),
    "settings": ("", "Return to project overview", "Pick up the next project step from its overview."),
}


@dataclass(frozen=True)
class ProjectCounts:
    files: int
    parsed: int
    pending: int
    approved: int
    datasets: int
    runs_done: int
    runs_active: int = 0


_RUN_ACTIVE = {"queued", "loading", "training", "running", "saving", "merging"}


def project_counts(pid: str) -> ProjectCounts:
    from finetune_studio import db
    from finetune_studio.data.fs import qa as qa_fs
    from finetune_studio.knowledge.sources import list_sources

    sources = list_sources(pid)
    pairs = qa_fs.list_qa_pairs(pid)
    runs = db.list_runs(pid)
    return ProjectCounts(
        files=len(sources),
        parsed=sum(1 for s in sources if s.ready),
        pending=sum(1 for p in pairs if p.get("status", "pending") == "pending"),
        approved=sum(1 for p in pairs if p.get("status") == "approved"),
        datasets=len(db.list_datasets(pid)),
        runs_done=sum(1 for r in runs if r.get("status") == "done"),
        runs_active=sum(1 for r in runs if r.get("status") in _RUN_ACTIVE),
    )


def next_for(page: str, c: ProjectCounts) -> tuple[str, str, str] | None:
    """(href suffix, title, why) for ``page`` given the counts; None keeps the bar's static entry."""
    if page in ("files", "data"):
        if not c.files:
            return ("", "Upload your files", "Everything starts from your documents: add them with Upload on this page.")
        if not c.parsed:
            return ("", "Wait for parsing", "Files are read in the background; the table shows when each one is ready.")
        if not (c.pending or c.approved):
            why = ("For training, the helper writes pairs from your files. To answer from the files without training, "
                   "build the Smart Brain instead.")
            return ("/data-prep", "Mine question-answer pairs", why)
        return None
    if page in ("pairs", "data-prep"):
        if not c.parsed:
            return ("/data", "Upload files first", "Pairs are mined from parsed files.")
        if not (c.pending or c.approved):
            return ("", "Mine pairs", "Pick files and start mining on this page.")
        if not c.approved:
            return ("", "Review the pending pairs", "Approve, fix or reject them below; only approved pairs train.")
        if not c.datasets:
            return ("", "Export the approved pairs", "Export turns approved pairs into the training dataset.")
        return None
    if page == "training":
        if not c.datasets:
            return ("/data-prep", "Build a dataset first", "Mine, approve and export pairs, or upload a JSONL dataset here.")
        if c.runs_active and not c.runs_done:
            return ("", "Wait for the run to finish", "Live status below follows it; test it on Testing when it is done.")
        if not c.runs_done:
            return ("", "Start a training run", "Pick the dataset and a preset below, then press Start.")
        return None
    return None


def flow_next(pid: str | None, page: str) -> tuple[str, str, str] | None:
    """Jinja global: the state-aware next step, or None (also on any error: the static entry stays)."""
    page = (page or "").strip()
    if not pid or page not in STATEFUL_PAGES:
        return None
    try:
        return next_for(page, project_counts(pid))
    except Exception:  # the bar must render even if a count fails
        log.warning("flow_next: counts failed for project %s", pid, exc_info=True)
        return None


def flow_next_step(pid: str | None, page: str) -> tuple[str, str, str] | None:
    """Jinja global + API: the state-aware step, else the page's static entry, else None (no Next on this page)."""
    page = (page or "").strip()
    return flow_next(pid, page) or STATIC_NEXT.get(page)
