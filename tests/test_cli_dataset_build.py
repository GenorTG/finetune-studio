"""`fts dataset build` + data/prep/dataset_build.py: the WebUI export pipeline, reachable without a server.

Hermetic: conftest's ``temp_db`` points the DB and FTS_ROOT at private temp dirs.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from finetune_studio import db
from finetune_studio.cli import _registry
from finetune_studio.data.fs import qa as qa_fs
from finetune_studio.data.prep import dataset_build as dsb
from finetune_studio.data.prep import grounding as g
from finetune_studio.db.datasets import list_datasets
from tests.knowledge_helpers import add_source


def run_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
            *argv: str) -> tuple[int, str, str]:
    monkeypatch.setattr(sys, "argv", ["fts", *argv])
    code = 0
    try:
        _registry.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def _pair(i: int) -> dict:
    return {
        "id": f"p{i}", "source_id": f"s{1 + i % 2}", "chunk_idx": i,
        "chunk_text": f"Chunk {i}: the fact number {i} is VALUE-{i}.",
        "question": f"What is fact {i}?", "answer": f"VALUE-{i}.",
        "category": "source-grounded", "status": "approved", "created_at": time.time(),
    }


@pytest.fixture
def project() -> dict:
    """A project with 10 approved pairs and no registered file: the completeness gate has nothing to block."""
    proj = db.create_project("cli-docs", base_model="x/test")
    for i in range(10):
        qa_fs.write_qa_pair(proj["id"], _pair(i))
    return proj


def _gappy_file(pid: str, filename: str = "broken.pdf") -> None:
    """A mined file with one data point no pair states: status ``gaps``."""
    sid = add_source(pid, filename, "The daily rate is EUR 41.75 for every booking.").id
    db.mining.insert_items(pid, sid, "job", [{"chunk_idx": 1, "origin": "agent", "kind": "fact",
                                               "text": "Rate", "values": ["EUR 41.75"]}])


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ── happy path ─────────────────────────────────────────────────────────

def test_build_by_name_writes_and_registers_dataset(project, monkeypatch, capsys) -> None:
    monkeypatch.setattr(g, "corpus_built", lambda pid: False)
    code, out, err = run_cli(monkeypatch, capsys, "dataset", "build", "--project", "cli-docs", "--json")
    assert code == 0, err
    s = json.loads(out)
    assert s["rows"] == 10 and s["grounded_rows"] == 0 and s["grounding"] is None
    assert s["completeness"] == {"files": 0, "complete": 0, "incomplete": []}
    path = Path(s["dataset"]["path"])
    assert path.name == f"{project['id']}-sharegpt-approved.jsonl"
    assert len(_rows(path)) == 10
    registered = list_datasets(project["id"])
    assert [d["data_path"] for d in registered] == [str(path)]
    assert registered[0]["name"] == "cli-docs · sharegpt · 10 rows"


def test_rebuild_updates_the_same_registry_row(project, monkeypatch, capsys) -> None:
    for _ in range(2):
        assert run_cli(monkeypatch, capsys, "dataset", "build", "--project", project["id"],
                       "--no-grounding")[0] == 0
    assert len(list_datasets(project["id"])) == 1


def test_auto_grounding_follows_corpus_and_reports_counts(project, monkeypatch, capsys) -> None:
    monkeypatch.setattr(g, "corpus_built", lambda pid: True)
    code, out, err = run_cli(monkeypatch, capsys, "dataset", "build", "--project", project["id"], "--json")
    assert code == 0, err
    s = json.loads(out)
    assert s["grounded_rows"] == 4  # 40% of 10
    assert s["grounding"]["grounded"] == 4
    assert sum(1 for r in _rows(Path(s["dataset"]["path"])) if r.get("grounded")) == 4
    assert "(4 with retrieved context)" in s["dataset"]["name"]


def test_explicit_share_seed_distractors_and_no_grounding(project, monkeypatch, capsys) -> None:
    monkeypatch.setattr(g, "corpus_built", lambda pid: True)
    base = ["dataset", "build", "--project", project["id"], "--json"]
    s1 = json.loads(run_cli(monkeypatch, capsys, *base, "--grounded-share", "1", "--distractors", "1")[1])
    assert s1["grounded_rows"] == 10 and s1["grounding"]["with_distractors"] == 10
    s0 = json.loads(run_cli(monkeypatch, capsys, *base, "--no-grounding")[1])
    assert s0["grounded_rows"] == 0
    a = run_cli(monkeypatch, capsys, *base, "--grounded-share", "0.5", "--seed", "1")[1]
    ids_a = [r["conversations"][-1]["value"] for r in _rows(Path(json.loads(a)["dataset"]["path"])) if r.get("grounded")]
    b = run_cli(monkeypatch, capsys, *base, "--grounded-share", "0.5", "--seed", "2")[1]
    ids_b = [r["conversations"][-1]["value"] for r in _rows(Path(json.loads(b)["dataset"]["path"])) if r.get("grounded")]
    assert len(ids_a) == len(ids_b) == 5 and ids_a != ids_b


def test_custom_name_fmt_and_text_summary(project, monkeypatch, capsys) -> None:
    code, out, err = run_cli(monkeypatch, capsys, "dataset", "build", "--project", project["id"],
                             "--name", "My set!", "--fmt", "openai", "--no-grounding")
    assert code == 0, err
    assert "Rows     : 10 (0 grounded" in out and "Complete : 0/0 file(s)" in out
    ds = list_datasets(project["id"])[0]
    assert ds["name"] == "My set!" and Path(ds["data_path"]).name == f"{project['id']}-My_set.jsonl"
    assert "messages" in _rows(Path(ds["data_path"]))[0]


# ── --out is fenced to the project ─────────────────────────────────────

def test_out_copy_inside_project_is_written(project, monkeypatch, capsys) -> None:
    from finetune_studio.data.fs.paths import project_roots
    target = project_roots(project["id"])[0] / "exports" / "plain.jsonl"
    code, out, err = run_cli(monkeypatch, capsys, "dataset", "build", "--project", project["id"],
                             "--no-grounding", "--out", str(target), "--json")
    assert code == 0, err
    assert json.loads(out)["copy"] == str(target) and len(_rows(target)) == 10


@pytest.mark.parametrize("bad", ["../escape.jsonl", "/etc/fts-escape.jsonl"])
def test_out_outside_project_is_refused_before_any_work(project, monkeypatch, capsys, bad: str) -> None:
    code, out, err = run_cli(monkeypatch, capsys, "dataset", "build", "--project", project["id"], "--out", bad)
    assert code == 1 and "--out" in err and "Traceback" not in err
    assert list_datasets(project["id"]) == [] and not Path("/etc/fts-escape.jsonl").exists()


# ── honest failures ────────────────────────────────────────────────────

def test_missing_project_is_a_clean_error(monkeypatch, capsys) -> None:
    code, _, err = run_cli(monkeypatch, capsys, "dataset", "build", "--project", "nope")
    assert code == 1 and "project not found: 'nope'" in err and "Traceback" not in err


def test_ambiguous_project_name_lists_ids(monkeypatch, capsys) -> None:
    a, b = db.create_project("twin"), db.create_project("twin")
    code, _, err = run_cli(monkeypatch, capsys, "dataset", "build", "--project", "twin")
    assert code == 1 and "ambiguous" in err and a["id"] in err and b["id"] in err


def test_bad_values_are_errors_not_clamped(project, monkeypatch, capsys) -> None:
    for extra in (["--grounded-share", "1.5"], ["--distractors", "5"]):
        code, _, err = run_cli(monkeypatch, capsys, "dataset", "build", "--project", project["id"], *extra)
        assert code == 1 and "Traceback" not in err
    code, _, err = run_cli(monkeypatch, capsys, "dataset", "build", "--project", project["id"],
                           "--grounded-share", "0.5", "--no-grounding")
    assert code == 2 and "not allowed with" in err  # argparse mutual exclusion


def test_blocked_export_gate_exits_nonzero_and_force_overrides(monkeypatch, capsys) -> None:
    proj = db.create_project("gappy")
    for i in range(3):
        qa_fs.write_qa_pair(proj["id"], _pair(i))
    _gappy_file(proj["id"])
    code, out, err = run_cli(monkeypatch, capsys, "dataset", "build", "--project", "gappy", "--no-grounding")
    assert code == 1 and out == ""
    assert "not every data point is approved yet in broken.pdf (gaps)" in err
    assert "broken.pdf: gaps (1 open, 0 awaiting review, 0 approved, 0 excluded of 1 data points)" in err and "--force" in err
    assert list_datasets(proj["id"]) == []
    code, out, err = run_cli(monkeypatch, capsys, "dataset", "build", "--project", "gappy",
                             "--no-grounding", "--force", "--json")
    assert code == 0, err
    assert json.loads(out)["completeness"]["incomplete"] == ["broken.pdf (gaps)"]


def test_real_gate_blocks_a_file_that_was_never_mined(monkeypatch, capsys) -> None:
    proj = db.create_project("lost-chunks")
    qa_fs.write_qa_pair(proj["id"], _pair(0))
    add_source(proj["id"], "gone.txt", "Two chunks of text nobody ever mined into pairs.")
    code, _, err = run_cli(monkeypatch, capsys, "dataset", "build", "--project", proj["id"],
                           "--no-grounding")
    assert code == 1 and "gone.txt" in err and list_datasets(proj["id"]) == []


def test_project_without_pairs_is_not_exported_silently(monkeypatch, capsys) -> None:
    proj = db.create_project("empty")
    code, _, err = run_cli(monkeypatch, capsys, "dataset", "build", "--project", proj["id"],
                           "--no-grounding")
    assert code == 1 and "no approved Q&A pairs to export" in err


def test_bare_dataset_command_prints_usage(monkeypatch, capsys) -> None:
    code, _, err = run_cli(monkeypatch, capsys, "dataset")
    assert code == 1 and "fts dataset build" in err


# ── module contract (what the route relies on) ─────────────────────────

def test_gate_force_logs_and_returns_summary() -> None:
    proj = db.create_project("gate")
    _gappy_file(proj["id"], "g.txt")
    with pytest.raises(dsb.ExportBlocked) as ei:
        dsb.completeness_gate(proj["id"])
    assert [f["filename"] for f in ei.value.files] == ["g.txt"] and ei.value.files[0]["status"] == "gaps"
    summary = dsb.completeness_gate(proj["id"], force=True)
    assert summary["files"] == 1 and summary["complete"] == 0 and summary["incomplete"][0]["filename"] == "g.txt"


def test_blocked_message_truncates_long_file_lists() -> None:
    err = dsb.ExportBlocked([{"filename": f"f{i}.txt", "status": "review"} for i in range(8)])
    assert "f0.txt (review)" in str(err) and "and 3 more" in str(err) and "f7.txt" not in str(err)
