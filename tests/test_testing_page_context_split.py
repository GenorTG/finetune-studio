"""The Testing page reports 'from memory' and 'with retrieved context' separately (same rule as the wizard)."""
from __future__ import annotations

from pathlib import Path

from finetune_studio.testing.run_store import is_grounded
from finetune_studio.testing.suite import CaseResult

HTML = (Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "templates"
        / "project_testing.html").read_text(encoding="utf-8")


def test_results_summary_splits_plain_and_grounded_rows() -> None:
    assert "c.judge_input && c.judge_input.grounded" in HTML
    assert "from memory (no context)" in HTML and "answering from retrieved context" in HTML


def test_a_case_asked_with_a_system_turn_is_flagged_grounded() -> None:
    plain = CaseResult("a", "g", "q", "k", "a", transcript=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
    grounded = CaseResult("b", "g", "q", "k", "a", transcript=[{"role": "system", "content": "CONTEXT: ..."}, {"role": "user", "content": "q"}])
    assert not is_grounded(plain) and is_grounded(grounded)


def test_run_store_saves_the_grounded_flag_with_the_case(temp_db) -> None:
    from finetune_studio import db
    from finetune_studio.testing import run_store

    pid = db.create_project(name="p")["id"]
    rid = run_store.resolve_owner_run(pid, "/m/merged")
    bid = run_store.start_run(rid, "quiz", model_path="/m/merged", kind="suite", config={}, total=1)["id"]
    run_store.save_case(bid, rid, CaseResult("b", "g", "q", "k", "a", transcript=[{"role": "system", "content": "CTX"}]))
    assert db.list_cases(bid)[0]["judge_input"]["grounded"] is True
