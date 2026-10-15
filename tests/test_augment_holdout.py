"""The held-out suite written by scripts/augment_dataset.py must be disjoint from training."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location("augment_dataset_under_test", REPO / "scripts" / "augment_dataset.py")
assert _SPEC and _SPEC.loader
augment = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(augment)

SOURCE = (
    "Risk RK-04 is owned by Elian Mertens. The Oriole Packaging agreement expires 2027-06-30.\n\n"
    "Helios may reject a shipment when more than 2 percent of cartons are damaged.\n\n"
    "Pavel Novak: Maintenance contact for Rotterdam. Policy owner: Nadiya Petrov.\n"
)


def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip().casefold()


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    proj = tmp_path / "projects" / "holdpid"
    (proj / "qa" / "pairs").mkdir(parents=True)
    src = proj / "files" / "000000000000"
    src.mkdir(parents=True)
    (src / "parsed.txt").write_text(SOURCE, encoding="utf-8")
    for i in range(40):
        (proj / "qa" / "pairs" / f"qa_orig_{i:03d}.json").write_text(
            json.dumps({"id": f"qa_orig_{i:03d}", "question": f"orig question {i}?",
                        "answer": f"orig answer {i}", "status": "approved",
                        "source_id": "000000000000"}),
            encoding="utf-8",
        )
    return tmp_path


def _run(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[set[str], set[str]]:
    monkeypatch.setattr(sys, "argv", ["augment_dataset.py", "--project-id", "holdpid", "--root", str(root)])
    assert augment.main() == 0
    proj = root / "projects" / "holdpid"
    held = json.loads((proj / "held-out.json").read_text(encoding="utf-8"))["cases"]
    ds = root / "data" / "projects" / "holdpid" / "datasets" / "holdpid-sharegpt-approved.jsonl"
    rows = [json.loads(x) for x in ds.read_text(encoding="utf-8").splitlines() if x.strip()]
    return (
        {_norm(c["question"]) for c in held},
        {_norm(r["conversations"][0]["value"]) for r in rows},
    )


def test_held_out_and_training_are_disjoint(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    held, train = _run(project, monkeypatch)
    assert held, "held-out suite must not be empty"
    assert train, "training set must not be empty"
    assert held.isdisjoint(train), f"leaked into training: {sorted(held & train)}"


def test_every_approved_question_lands_in_exactly_one_side(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    held, train = _run(project, monkeypatch)
    pairs_dir = project / "projects" / "holdpid" / "qa" / "pairs"
    approved = {_norm(json.loads(p.read_text(encoding="utf-8"))["question"]) for p in pairs_dir.glob("*.json")}
    assert held | train == approved
    # Generated additions must be part of the pool that gets split.
    assert any(not q.startswith("orig question") for q in approved)


def test_split_is_stable_across_reruns(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    held1, _ = _run(project, monkeypatch)
    held2, train2 = _run(project, monkeypatch)
    assert held1 == held2
    assert held2.isdisjoint(train2)
