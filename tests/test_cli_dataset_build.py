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
def project(monkeypatch: pytest.MonkeyPatch) -> dict:
    """A project with 10 approved pairs; coverage gate stubbed to 'everything covered'."""
    proj = db.create_project("cli-docs", base_model="x/test")
    for i in range(10):
        qa_fs.write_qa_pair(proj["id"], _pair(i))
    monkeypatch.setattr("finetune_studio.data.prep.coverage_fill.fill_all_project_gaps",
                        lambda pid, mode="pending": {"mode": mode, "pairs_created": 2, "chunks_filled": 1,
                                                 "uncovered_chunks": []})
    return proj


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ── happy path ─────────────────────────────────────────────────────────

def test_build_by_name_writes_and_registers_dataset(project, monkeypatch, capsys) -> None:
    monkeypatch.setattr(g, "corpus_built", lambda pid: False)
    code, out, err = run_cli(monkeypatch, capsys, "dataset", "build", "--project", "cli-docs", "--json")
    assert code == 0, err
    s = json.loads(out)
    assert s["rows"] == 10 and s["grounded_rows"] == 0 and s["grounding"] is None
    assert s["coverage_fill"] == {"mode": "pending", "pairs_created": 2, "pairs_promoted": 0, "chunks_filled": 1,
                                  "skipped_no_content": 0, "uncovered": 0}
    path = Path(s["dataset"]["path"])
    assert path.name == f"{project['id']}-sharegpt-approved.jsonl"
    assert len(_rows(path)) == 10
    registered = list_datasets(project["id"])
    assert [d["data_path"] for d in registered] == [str(path)]
    assert registered[0]["name"] == "cli-docs · sharegpt · 10 rows"


def test_rebuild_updates_the_same_registry_row(project, monkeypatch, capsys) -> None:
    for _ in range(2):
        assert run_cli(monkeypatch, capsys, "dataset", "build", "--project", project["id"],
                       "--no-rag-grounding")[0] == 0
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
    s0 = json.loads(run_cli(monkeypatch, capsys, *base, "--no-rag-grounding")[1])
    assert s0["grounded_rows"] == 0
    a = run_cli(monkeypatch, capsys, *base, "--grounded-share", "0.5", "--seed", "1")[1]
    ids_a = [r["conversations"][-1]["value"] for r in _rows(Path(json.loads(a)["dataset"]["path"])) if r.get("grounded")]
    b = run_cli(monkeypatch, capsys, *base, "--grounded-share", "0.5", "--seed", "2")[1]
    ids_b = [r["conversations"][-1]["value"] for r in _rows(Path(json.loads(b)["dataset"]["path"])) if r.get("grounded")]
    assert len(ids_a) == len(ids_b) == 5 and ids_a != ids_b


def test_custom_name_fmt_and_text_summary(project, monkeypatch, capsys) -> None:
    code, out, err = run_cli(monkeypatch, capsys, "dataset", "build", "--project", project["id"],
                             "--name", "My set!", "--fmt", "openai", "--no-rag-grounding")
    assert code == 0, err
    assert "Rows     : 10 (0 grounded" in out and "Coverage : 1 chunk(s) filled with 2 extractive pair(s) (left pending for review, NOT in this dataset); 0 still uncovered" in out
    ds = list_datasets(project["id"])[0]
    assert ds["name"] == "My set!" and Path(ds["data_path"]).name == f"{project['id']}-My_set.jsonl"
    assert "messages" in _rows(Path(ds["data_path"]))[0]


# ── --out is fenced to the project ─────────────────────────────────────

def test_out_copy_inside_project_is_written(project, monkeypatch, capsys) -> None:
    from finetune_studio.data.fs.paths import project_roots
    target = project_roots(project["id"])[0] / "exports" / "plain.jsonl"
    code, out, err = run_cli(monkeypatch, capsys, "dataset", "build", "--project", project["id"],
                             "--no-rag-grounding", "--out", str(target), "--json")
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
                           "--grounded-share", "0.5", "--no-rag-grounding")
    assert code == 2 and "not allowed with" in err  # argparse mutual exclusion


def test_blocked_export_gate_exits_nonzero_and_force_overrides(monkeypatch, capsys) -> None:
    proj = db.create_project("gappy")
    for i in range(3):
        qa_fs.write_qa_pair(proj["id"], _pair(i))
    gap = {"source": "s9", "filename": "broken.pdf", "chunk_idx": 3, "reason": "no_specific_question"}
    monkeypatch.setattr("finetune_studio.data.prep.coverage_fill.fill_all_project_gaps",
                        lambda pid, mode="pending": {"pairs_created": 0, "chunks_filled": 0, "uncovered_chunks": [gap]})
    code, out, err = run_cli(monkeypatch, capsys, "dataset", "build", "--project", "gappy",
                             "--no-rag-grounding")
    assert code == 1 and out == ""
    assert "no usable Q&A could be made from broken.pdf" in err
    assert "broken.pdf chunk 3: no_specific_question" in err and "--force" in err
    assert list_datasets(proj["id"]) == []
    code, out, err = run_cli(monkeypatch, capsys, "dataset", "build", "--project", "gappy",
                             "--no-rag-grounding", "--force", "--json")
    assert code == 0, err
    assert json.loads(out)["coverage_fill"]["uncovered"] == 1


def test_real_gate_blocks_a_source_whose_parsed_chunks_are_gone(monkeypatch, capsys) -> None:
    proj = db.create_project("lost-chunks")
    qa_fs.write_qa_pair(proj["id"], _pair(0))
    qa_fs.write_qa_source(proj["id"], {"id": "s1", "filename": "gone.txt", "sha256": "ab" * 32,
                                       "chunk_count": 2, "uploaded_at": time.time()})
    code, _, err = run_cli(monkeypatch, capsys, "dataset", "build", "--project", proj["id"],
                           "--no-rag-grounding")
    assert code == 1 and "gone.txt" in err and list_datasets(proj["id"]) == []


def test_coverage_check_crash_blocks_the_export(project, monkeypatch, capsys) -> None:
    def boom(pid: str) -> dict:
        raise RuntimeError("disk on fire")
    monkeypatch.setattr("finetune_studio.data.prep.coverage_fill.fill_all_project_gaps", boom)
    code, _, err = run_cli(monkeypatch, capsys, "dataset", "build", "--project", project["id"])
    assert code == 1 and "coverage verification failed: disk on fire" in err
    assert list_datasets(project["id"]) == []


def test_project_without_pairs_is_not_exported_silently(monkeypatch, capsys) -> None:
    proj = db.create_project("empty")
    monkeypatch.setattr("finetune_studio.data.prep.coverage_fill.fill_all_project_gaps",
                        lambda pid: {"uncovered_chunks": []})
    code, _, err = run_cli(monkeypatch, capsys, "dataset", "build", "--project", proj["id"],
                           "--no-rag-grounding")
    assert code == 1 and "no approved Q&A pairs to export" in err


def test_bare_dataset_command_prints_usage(monkeypatch, capsys) -> None:
    code, _, err = run_cli(monkeypatch, capsys, "dataset")
    assert code == 1 and "fts dataset build" in err


# ── module contract (what the route relies on) ─────────────────────────

def test_gate_force_logs_and_returns_summary(monkeypatch) -> None:
    monkeypatch.setattr("finetune_studio.data.prep.coverage_fill.fill_all_project_gaps",
                        lambda pid: {"uncovered_chunks": [{"source": "s"}]})
    with pytest.raises(dsb.ExportBlocked) as ei:
        dsb.coverage_gate("p")
    assert ei.value.files == ["s"]
    assert dsb.coverage_gate("p", force=True)["uncovered_chunks"] == [{"source": "s"}]


def test_blocked_message_truncates_long_file_lists() -> None:
    err = dsb.ExportBlocked([{"filename": f"f{i}.txt"} for i in range(8)])
    assert "f0.txt" in str(err) and "and more" in str(err) and "f7.txt" not in str(err)
