"""Read-only state tools on a fixture project (CPU only, no model)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from finetune_studio import db
from finetune_studio.data.fs import qa as qa_fs
from finetune_studio.guide import state
from finetune_studio.guide.tools import ToolContext, run_tool


@pytest.fixture
def fts_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "fts"
    (root / "projects").mkdir(parents=True)
    monkeypatch.setattr("finetune_studio.data.fs.paths._ROOT", root)
    monkeypatch.setattr("finetune_studio.data.fs.paths._PROJECTS", root / "projects")
    return root


def _jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(path)


def _qa_rows(n: int) -> list[dict]:
    return [{"messages": [
        {"role": "user", "content": f"What is the warranty period of product number {i}?"},
        {"role": "assistant", "content": f"Product number {i} has a warranty of {i + 1} years from the delivery date."},
    ]} for i in range(n)]


@pytest.fixture
def project(fts_root: Path, tmp_path: Path) -> dict:
    created = db.create_project(name="Guide fixture", base_model="/models/Qwen__Qwen3-4B")
    pid = created["id"] if isinstance(created, dict) else created
    qa_fs.write_qa_source(pid, {"id": "s1", "filename": "spec.csv", "chunk_count": 2, "status": "generated"})
    qa_fs.write_qa_source(pid, {"id": "s2", "filename": "empty.txt", "chunk_count": 0, "status": "registered"})
    for i, status in enumerate(("approved", "approved", "pending")):
        qa_fs.write_qa_pair(pid, {"id": f"qa{i}", "source_id": "s1", "question": f"Q{i}?", "answer": f"A{i}.",
                                  "chunk_idx": 1, "status": status, "chunk_text": f"A{i} text"})
    ds_path = _jsonl(tmp_path / "ds.jsonl", _qa_rows(60))
    ds = db.create_dataset(pid, "approved-export", ds_path, source="data-prep-export", qa_count=60)
    return {"pid": pid, "dataset": ds, "path": ds_path}


def test_project_overview_matches_readiness_summary(project: dict) -> None:
    out = state.project_overview(project["pid"])
    ready = state.readiness(project["pid"])
    assert out["summary"] == ready["summary"]
    assert out["sources"] == {"count": 2, "parsed": 1}
    assert out["qa_pairs"]["approved"] == 2 and out["qa_pairs"]["pending"] == 1
    assert out["datasets"][0]["name"] == "approved-export"
    assert out["runs"]["count"] == 0
    assert out["next_step"] == "Review pending Q&A pairs."


def test_project_overview_without_project_lists_projects(project: dict) -> None:
    out = state.project_overview(None)
    assert out["project"] is None
    assert any(p["name"] == "Guide fixture" for p in out["projects"])


def test_unknown_project_is_an_error_not_a_crash() -> None:
    assert "error" in state.project_overview("nope0000")
    assert "error" in state.list_runs("nope0000")
    assert "error" in state.list_datasets(None)


def test_list_datasets_and_runs(project: dict) -> None:
    ds = state.list_datasets(project["pid"])
    assert ds["count"] == 1 and ds["datasets"][0]["rows"] == 60
    run = db.create_run(project["pid"], "run-a", base_model="/m/Qwen__Qwen3-4B", settings_obj={"training_mode": "dpo", "num_epochs": 1})
    db.update_run(run["id"], status="done", final_loss=0.31, output_path="/out/run-a", metrics={"total_steps": 42})
    db.update_project(project["pid"], production_run=run["id"])
    runs = state.list_runs(project["pid"])
    row = runs["runs"][0]
    assert runs["production_run"] == run["id"]
    assert row["name"] == "run-a" and row["status"] == "done" and row["production"] is True
    assert row["training_mode"] == "dpo" and row["base_model"] == "Qwen__Qwen3-4B" and row["has_output"]


def test_dataset_health_wraps_the_page_check(project: dict) -> None:
    out = state.dataset_health(project["pid"])
    assert out["examples"] == 60 and out["trainable"] == 60
    assert out["holdout"] >= 1
    assert out["dataset"]["name"] == "approved-export"
    assert out["qa_audit"]["approved_pairs"] == 2
    assert state.dataset_health(project["pid"], training_mode="bogus")["error"].startswith("training_mode")
    assert "not found" in state.dataset_health(project["pid"], dataset_id="zzz")["error"]


def test_dataset_health_flags_unusable_rows_for_the_chosen_route(project: dict, tmp_path: Path) -> None:
    """SFT rows are not DPO rows: the route-aware checker reports it instead of passing them."""
    out = state.dataset_health(project["pid"], training_mode="dpo")
    assert out["trainable"] == 0
    assert out["issues"]


def test_dataset_health_without_dataset_points_to_export(fts_root: Path) -> None:
    created = db.create_project(name="No data")
    pid = created["id"] if isinstance(created, dict) else created
    out = state.dataset_health(pid)
    assert "export approved Q&A pairs" in out["error"]


def test_recommend_training_uses_dataset_size_and_flags_too_few_steps(project: dict) -> None:
    out = state.recommend_training(project["pid"], goal="memorize product facts")
    assert out["route"] == "sft" and out["tier"] == "balanced"
    assert out["dataset"]["rows"] == 60
    assert out["base_params_b"] == 4.0
    assert out["settings"]["num_epochs"] >= 1 and out["settings"]["lora_rank"] == 64
    assert out["steps_floor"] == 400
    assert isinstance(out["too_few_steps"], bool)
    assert {a["tier"] for a in out["alternatives"]} == {"smoke", "precision", "overkill"}
    assert "upper bound" in out["caveat"]
    # 60 pairs cannot reach the Precision floor even at the 60-epoch cap → flagged with the reason.
    precision = state.recommend_training(project["pid"], tier="precision")
    assert precision["too_few_steps"] is True
    assert any("too few" in w for w in precision["warnings"])


@pytest.mark.parametrize(
    ("goal", "route"),
    [
        ("make it prefer concise answers over rambling ones", "dpo"),
        ("teach it to call my API tools", "tool_sft"),
        ("adapt to my raw text corpus", "continued_pretraining"),
        ("distill reasoning traces from a teacher", "reasoning_distillation"),
        ("answers must cite sources and the docs change weekly", "rag"),
        ("", "sft"),
    ],
)
def test_recommend_training_picks_the_route_for_the_goal(project: dict, goal: str, route: str) -> None:
    out = state.recommend_training(project["pid"], goal=goal)
    assert out["route"] == route
    if route == "dpo":
        assert out["settings"]["learning_rate"] == "1e-6" and out["settings"]["num_epochs"] == 1
    if route == "rag":
        assert "settings" not in out and out["page"] == "rag"


def test_recommend_training_without_dataset_says_it_assumes_the_reference(fts_root: Path) -> None:
    created = db.create_project(name="Empty")
    pid = created["id"] if isinstance(created, dict) else created
    out = state.recommend_training(pid)
    assert out["dataset"] is None
    assert any("No dataset" in w for w in out["warnings"])


def test_system_status_is_compact_and_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "finetune_studio.webui.gpu_probe.vram_devices",
        lambda: [{"index": 0, "name": "RTX 3090", "used_gb": 1.2, "total_gb": 24.0, "pct": 5.0}],
    )
    out = state.system_status()
    assert out["gpus"] == [{"index": 0, "name": "RTX 3090", "used_gb": 1.2, "total_gb": 24.0}]
    assert {"accelerator", "ram", "helper", "disk"} <= set(out)
    assert out["disk"]["free_gb"] > 0
    assert "loaded_model" in out and "training" in out
    json.dumps(out)


def test_tools_dispatch_through_run_tool(project: dict) -> None:
    ctx = ToolContext(pid=project["pid"], page="training")
    assert run_tool(ctx, "project_overview", {})["project"]["id"] == project["pid"]
    assert run_tool(ctx, "list_datasets", {})["count"] == 1
    assert run_tool(ctx, "explain_setting", {"name": "epochs"})["setting"] == "num_epochs"
    assert run_tool(ctx, "app_help", {"query": "how do I export a gguf"})["results"][0]["entry"] == "export"
    assert run_tool(ctx, "app_help", {"query": ""})["error"] == "query required"
    assert run_tool(ctx, "app_help", {"query": "zxqv blorf"})["results"] == []
    assert "error" in run_tool(ctx, "nope", {})
    # project-scoped tools refuse to run with no active project
    assert "no active project" in run_tool(ToolContext(), "list_sources", {})["error"]
    assert "no active project" in run_tool(ToolContext(), "inspect_project_readiness", {})["error"]
