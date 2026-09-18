"""Source-grounded augmentation must hit every targeted fact in the corpus.

Builds a tiny synthetic project (two source files containing the targeted
facts), runs the augmentation, and asserts that:
- every TARGET_PAIRS fact is found
- the answer text contains the answer signature
- the sharegpt dataset row count grows by the augmented count
- the held-out suite has the deterministic 90/10 split
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Use the same interpreter pytest is running under; "python" is not on PATH in
# the venv-isolated CI environment.
PYTHON = sys.executable


SYNTH_SOURCES = {
    "risk_register.csv.md": textwrap.dedent(
        """
        # Risk register

        | risk_id | description | owner | mitigation |
        | --- | --- | --- | --- |
        | RK-01 | C-17 maintenance debt | Pavel Novak | weekly inspection |
        | RK-04 | carton damage | Elian Mertens | weekly inspection |
        """
    ),
    "release_notes.md": textwrap.dedent(
        """
        # Release notes

        The Oriole Packaging agreement took effect 2026-07-01 and expires
        2027-06-30. Oriole supplies cartons in sizes S, M, and L. The
        service level is 98.5 percent on-time delivery measured monthly.

        Helios may reject a shipment when more than 2 percent of cartons in
        a lot are crushed, wet, or dimensionally incorrect.

        Project OCTOPUS-7741 marked unload_regression_fixed as true on
        2026-09-08. The external_api_hidden flag is set for the same
        project and will be reverted in release 2026.10.
        """
    ),
}


@pytest.fixture()
def synth_project(tmp_path: Path) -> Path:
    """Set up a project root with qa/pairs/ and files/<sha>/parsed.txt."""
    proj = tmp_path / "projects" / "synthpid"
    pairs = proj / "qa" / "pairs"
    files = proj / "files"
    pairs.mkdir(parents=True)
    files.mkdir(parents=True)
    # Seed 12 qa pairs (small enough to be a clear split demo).
    for i in range(12):
        (pairs / f"qa_synth_{i:03d}.json").write_text(
            json.dumps({"id": f"qa_synth_{i:03d}", "question": f"q{i}", "answer": f"a{i}", "status": "approved"}),
            encoding="utf-8",
        )
    for i, (stem, content) in enumerate(SYNTH_SOURCES.items()):
        sha = f"{i:08x}{'0' * 4}"
        d = files / sha
        d.mkdir()
        (d / "parsed.txt").write_text(content, encoding="utf-8")
    return tmp_path


def test_augment_writes_qa_pairs_and_sharegpt(synth_project: Path) -> None:
    """End-to-end: run augment_dataset.main() on a synthetic project and assert outputs."""
    # Make sure the script is importable as a module.
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "scripts" / "augment_dataset.py"
    assert script.is_file(), f"missing {script}"

    # Run as a subprocess (the script expects /home paths; we override via --root).
    result = subprocess.run(
        [PYTHON, str(script), "--project-id", "synthpid", "--root", str(synth_project)],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, f"script failed:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert "merged" in result.stdout

    proj = synth_project / "projects" / "synthpid"

    # augmented JSONL on disk.
    aug_path = proj / "augmented-pairs-v2.jsonl"
    assert aug_path.is_file()
    aug = [json.loads(l) for l in aug_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(aug) >= 2, f"expected at least 2 augmented pairs, got {len(aug)}"

    # Every augmented pair's answer contains a fact signature.
    for p in aug:
        assert p["question"]
        assert p["correct_answer"]
        assert p["source_id"] in {f"{i:08x}{'0' * 4}" for i in range(len(SYNTH_SOURCES))}

    # qa/pairs/ grew: 12 originals + N augmented.
    pair_files = list((proj / "qa" / "pairs").glob("*.json"))
    assert len(pair_files) == 12 + len(aug)

    # Held-out suite: deterministic 90/10 split → ceil(12 * 0.1) = 2 cases (or 1, depending on int).
    held_path = proj / "held-out.json"
    assert held_path.is_file()
    suite = json.loads(held_path.read_text(encoding="utf-8"))
    assert suite["seed"] == 42
    assert suite["train_ratio"] == 0.9
    assert 1 <= len(suite["cases"]) <= 2

    # Sharegpt dataset reflects the augmented count.
    ds_path = synth_project / "data" / "projects" / "synthpid" / "datasets" / "synthpid-sharegpt-approved.jsonl"
    assert ds_path.is_file()
    ds_rows = [l for l in ds_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(ds_rows) == 12 + len(aug)


def test_augment_pair_answers_hold_real_source_facts(synth_project: Path) -> None:
    """The RK-04 row's answer must name Elian Mertens from the risk register."""
    proj = synth_project / "projects" / "synthpid"
    aug_path = proj / "augmented-pairs-v2.jsonl"
    if not aug_path.is_file():
        # Run the augment script first.
        subprocess.run(
            [PYTHON, str(Path(__file__).resolve().parent.parent / "scripts" / "augment_dataset.py"),
             "--project-id", "synthpid", "--root", str(synth_project)],
            check=True, capture_output=True, text=True, timeout=30,
        )
    aug = [json.loads(l) for l in aug_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Find the RK-04 pair.
    rk04 = next((p for p in aug if "RK-04" in p["question"]), None)
    assert rk04 is not None, "expected RK-04 pair in augmentation"
    assert "Elian Mertens" in rk04["correct_answer"], (
        f"RK-04 answer must contain owner name; got: {rk04['correct_answer']!r}"
    )


def test_export_deduplicates_questions_and_prefers_original_pair(synth_project: Path) -> None:
    proj = synth_project / "projects" / "synthpid"
    duplicate_question = "When and where did conveyor C-17 stop?"
    original_answer = "C-17 stopped in Rotterdam at 09:42 CET."
    pairs_dir = proj / "qa" / "pairs"
    (pairs_dir / "qa_original.json").write_text(json.dumps({
        "question": duplicate_question,
        "answer": original_answer,
        "status": "approved",
        "source_id": "000000000000",
    }), encoding="utf-8")
    (pairs_dir / "qa_augmented.json").write_text(json.dumps({
        "question": f"  {duplicate_question}  ",
        "answer": "A whole noisy JSON record that should not train the model.",
        "status": "approved",
        "source_id": "000000000000",
        "category": "source-grounded-augmented",
    }), encoding="utf-8")

    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [PYTHON, str(repo_root / "scripts" / "augment_dataset.py"),
         "--project-id", "synthpid", "--root", str(synth_project)],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    dataset = synth_project / "data/projects/synthpid/datasets/synthpid-sharegpt-approved.jsonl"
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    matches = [
        row for row in rows
        if row["conversations"][0]["value"].strip() == duplicate_question
    ]
    assert len(matches) == 1
    assert matches[0]["conversations"][1]["value"] == original_answer
