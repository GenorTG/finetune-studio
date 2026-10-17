"""The flow bar's Next step follows the project's state on Files, Pairs and Training (walkthrough 2026-10-17)."""
from __future__ import annotations

import pytest

from finetune_studio.webui import flow_next as fn


def counts(**kw: int) -> fn.ProjectCounts:
    base = {"files": 0, "parsed": 0, "pending": 0, "approved": 0, "datasets": 0, "runs_done": 0}
    return fn.ProjectCounts(**{**base, **kw})


@pytest.mark.parametrize(("page", "state", "title", "href"), [
    ("files", counts(), "Upload your files", ""),
    ("files", counts(files=3, parsed=3), "Mine question-answer pairs", "/data-prep"),
    ("pairs", counts(), "Upload files first", "/data"),
    ("pairs", counts(files=3, parsed=3), "Mine pairs", ""),
    ("pairs", counts(files=3, parsed=3, pending=40), "Review the pending pairs", ""),
    ("pairs", counts(files=3, parsed=3, pending=40, approved=5), "Export the approved pairs", ""),
    ("training", counts(files=3, parsed=3, approved=5), "Build a dataset first", "/data-prep"),
    ("training", counts(files=3, parsed=3, approved=5, datasets=1), "Start a training run", ""),
])
def test_next_step_follows_the_state(page: str, state: fn.ProjectCounts, title: str, href: str) -> None:
    got = fn.next_for(page, state)
    assert got is not None and (got[0], got[1]) == (href, title)


@pytest.mark.parametrize(("page", "state"), [
    ("files", counts(files=3, parsed=3, pending=4)),
    ("pairs", counts(files=3, parsed=3, approved=5, datasets=1)),
    ("training", counts(datasets=1, runs_done=1)),
    ("testing", counts()),
])
def test_static_entry_stays_once_the_page_step_is_done(page: str, state: fn.ProjectCounts) -> None:
    assert fn.next_for(page, state) is None


def test_a_failing_count_keeps_the_static_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(pid: str) -> fn.ProjectCounts:
        raise RuntimeError("db down")
    monkeypatch.setattr(fn, "project_counts", boom)
    assert fn.flow_next("p1", "pairs") is None
    assert fn.flow_next("p1", "chat") is None and fn.flow_next(None, "pairs") is None


def test_base_template_uses_it_and_renders_same_page_steps_as_text() -> None:
    from pathlib import Path
    html = (Path(fn.__file__).parent / "templates" / "base.html").read_text(encoding="utf-8")
    assert "flow_next(pid, bc)" in html and 'class="next-step-here"' in html
