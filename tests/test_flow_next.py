"""The flow bar's Next step follows project state on Files, Smart Brain, Pairs and Training."""
from __future__ import annotations

import pytest

from finetune_studio.webui import flow_next as fn


def counts(**kw: int) -> fn.ProjectCounts:
    base = {"files": 0, "parsed": 0, "pending": 0, "approved": 0, "datasets": 0, "runs_done": 0}
    return fn.ProjectCounts(**{**base, **kw})


@pytest.mark.parametrize(("page", "state", "title", "href"), [
    ("files", counts(), "Upload your files", ""),
    ("files", counts(files=3, parsed=3), "Build the Smart Brain", "/brain"),
    ("files", counts(files=3, parsed=3, brain_built=True), "Ask your documents", "/brain"),
    ("smart brain", counts(), "Add your files", "/data"),
    ("smart brain", counts(files=3), "Wait for your files to finish reading", "/data"),
    ("smart brain", counts(files=3, parsed=3), "Build the Smart Brain", ""),
    ("pairs", counts(), "Upload files first", "/data"),
    ("pairs", counts(files=3, parsed=3), "Mine pairs", ""),
    ("pairs", counts(files=3, parsed=3, pending=40), "Review the pending pairs", ""),
    ("pairs", counts(files=3, parsed=3, pending=40, approved=5), "Export the approved pairs", ""),
    ("training", counts(files=3, parsed=3, approved=5), "Build a dataset first", "/data-prep"),
    ("training", counts(files=3, parsed=3, approved=5, datasets=1), "Start a training run", ""),
    ("training", counts(datasets=1, runs_active=1), "Wait for the run to finish", ""),
])
def test_next_step_follows_the_state(page: str, state: fn.ProjectCounts, title: str, href: str) -> None:
    got = fn.next_for(page, state)
    assert got is not None and (got[0], got[1]) == (href, title)


@pytest.mark.parametrize(("page", "state"), [
    ("pairs", counts(files=3, parsed=3, approved=5, datasets=1)),
    ("training", counts(datasets=1, runs_done=1)),
    ("testing", counts()),
    ("smart brain", counts(files=3, parsed=3, brain_built=True)),
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
    assert "flow_next_step(pid, bc)" in html and 'class="next-step-here"' in html


def test_built_smart_brain_falls_through_to_quiz_step(monkeypatch: pytest.MonkeyPatch) -> None:
    state = counts(files=3, parsed=3, brain_built=True)
    monkeypatch.setattr(fn, "project_counts", lambda _pid: state)
    assert fn.flow_next_step("project", "smart brain") == fn.STATIC_NEXT["smart brain"]


def test_flow_next_api_follows_the_counts(client, monkeypatch: pytest.MonkeyPatch) -> None:
    """The bar asks again after an upload parses: the API returns the same step the template would render."""
    pid = client.post("/api/projects", json={"name": "flow-next-api"}).json()["id"]
    counts = {"n": fn.ProjectCounts(files=0, parsed=0, pending=0, approved=0, datasets=0, runs_done=0)}
    monkeypatch.setattr(fn, "project_counts", lambda p: counts["n"])
    first = client.get(f"/api/projects/{pid}/flow-next?page=files").json()["next"]
    assert first == {"href": "", "title": "Upload your files", "why": first["why"]}
    counts["n"] = fn.ProjectCounts(files=6, parsed=6, pending=0, approved=0, datasets=0, runs_done=0)
    assert client.get(f"/api/projects/{pid}/flow-next?page=files").json()["next"]["href"] == "/brain"
    counts["n"] = fn.ProjectCounts(files=6, parsed=6, pending=0, approved=40, datasets=1, runs_done=0)
    # the page's own step is done: the static entry comes back (the in-place refresh must not keep "Export ...")
    assert client.get(f"/api/projects/{pid}/flow-next?page=pairs").json()["next"]["title"] == "Train an adapter"
    assert client.get(f"/api/projects/{pid}/flow-next?page=chat").json()["next"]["href"] == "/testing"
    counts["n"] = fn.ProjectCounts(files=6, parsed=6, pending=0, approved=40, datasets=1, runs_done=0)
    brain_next = client.get(f"/api/projects/{pid}/flow-next?page=smart%20brain").json()["next"]
    assert brain_next["title"] == "Build the Smart Brain" and brain_next["href"] == ""
    counts["n"] = fn.ProjectCounts(files=6, parsed=6, pending=0, approved=40, datasets=1, runs_done=0, brain_built=True)
    assert client.get(f"/api/projects/{pid}/flow-next?page=smart%20brain").json()["next"]["title"] == "Score its answers"
    assert client.get(f"/api/projects/{pid}/flow-next?page=nowhere").json() == {"next": None}
    assert client.get("/api/projects/nope/flow-next?page=files").status_code == 404
