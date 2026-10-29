"""Guide actions: prepared server-side against the real project, run by the panel (ask or hands-free)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db
from finetune_studio.guide import actions
from finetune_studio.guide.tools import ToolContext, run_tool

WEB = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui"


def _project(base_model: str = "/models/Qwen3.5-9B") -> str:
    return db.create_project("guide-actions", base_model=base_model)["id"]


def _dataset(pid: str, tmp_path: Path, name: str = "pairs") -> dict:
    f = tmp_path / f"{name}.jsonl"
    f.write_text(json.dumps({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}) + "\n")
    return db.create_dataset(project_id=pid, name=name, data_path=str(f), source="test", qa_count=1, size_bytes=1)


def test_an_action_is_only_prepared_never_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pid = _project()
    ds = _dataset(pid, tmp_path)
    import finetune_studio.webui.app  # noqa: F401 - the route modules load in the app's order
    started = []
    monkeypatch.setattr("finetune_studio.webui.routes.training.start_training", lambda *a, **k: started.append(1))
    r = run_tool(ToolContext(pid=pid), "start_training", {"preset": "qlora", "num_epochs": 2})
    assert r["ok"] and r["status"] == "awaiting_confirmation" and r["mode"] == "ask"
    ev = r["ui_event"]
    assert ev["type"] == "action" and ev["action"] == "start_training" and ev["method"] == "POST"
    assert ev["url"] == "/api/training/start"
    assert ev["body"] == {"project_id": pid, "dataset_id": ds["id"], "model_path": "/models/Qwen3.5-9B",
                          "preset_id": "qlora", "overrides": {"num_epochs": 2}, "training_mode": "sft"}
    assert started == [] and db.list_runs(pid) == []           # nothing ran


@pytest.mark.parametrize("name,args,msg", [
    ("start_training", {}, "no dataset"),
    ("build_reworded_quiz", {}, "no dataset"),
    ("start_mining", {}, "no parsed file"),
    ("run_test", {"suite": "nope"}, "unknown suite"),
])
def test_actions_refuse_with_the_reason(name: str, args: dict, msg: str) -> None:
    pid = _project()
    r = run_tool(ToolContext(pid=pid), name, args)
    assert msg in r["error"] and "ui_event" not in r


def test_training_checks_dataset_preset_and_base_model(tmp_path: Path) -> None:
    pid = _project(base_model="")
    _dataset(pid, tmp_path)
    assert "no base model" in run_tool(ToolContext(pid=pid), "start_training", {})["error"]
    pid = _project()
    _dataset(pid, tmp_path)
    assert "unknown preset" in run_tool(ToolContext(pid=pid), "start_training", {"preset": "turbo"})["error"]
    assert "unknown dataset" in run_tool(ToolContext(pid=pid), "start_training", {"dataset_id": "zzz"})["error"]
    assert "no active project" in run_tool(ToolContext(), "build_dataset", {})["error"]


@pytest.mark.parametrize("arguments", [{"preset": "baseline"}, {"tier": "balanced"}, {"tier": "baseline"}])
def test_baseline_uses_the_adaptive_training_advisor(arguments: dict[str, str], tmp_path: Path) -> None:
    from finetune_studio.training.preset_advisor import propose

    pid = _project()
    ds = _dataset(pid, tmp_path)
    advice = propose(tier="balanced", base_model_ref="/models/Qwen3.5-9B", dataset_path=ds["data_path"])

    result = run_tool(ToolContext(pid=pid), "start_training", arguments)

    assert result.get("status") == "awaiting_confirmation"
    body = result["ui_event"]["body"]
    assert body["dataset_id"] == ds["id"]
    assert body["model_path"] == "/models/Qwen3.5-9B"
    assert "preset_id" not in body
    assert body["num_epochs"] == advice.num_epochs
    assert body["lora_rank"] == advice.lora_rank
    assert body["lora_alpha"] == advice.lora_alpha
    assert body["learning_rate"] == advice.learning_rate
    assert body["batch_size"] == advice.batch_size
    assert body["gradient_accumulation_steps"] == advice.gradient_accumulation_steps
    assert body["warmup_steps"] == advice.warmup_steps
    assert "Baseline is balanced" in result["ui_event"]["detail"]
    detail = result["ui_event"]["detail"]
    assert f"alpha {advice.lora_alpha}" in detail
    assert f"learning rate {advice.learning_rate}" in detail
    assert f"batch size {advice.batch_size}" in detail
    assert f"gradient accumulation {advice.gradient_accumulation_steps}" in detail
    assert f"warmup {advice.warmup_steps} steps" in detail
    assert "max sequence length 2048" in detail and "final model is merged" in detail


@pytest.mark.parametrize("requested", ["Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B"])
def test_training_resolves_project_model_alias_to_its_local_path(requested: str, tmp_path: Path) -> None:
    configured = "/srv/fts/hf_models/Qwen__Qwen3.5-0.8B"
    pid = _project(base_model=configured)
    _dataset(pid, tmp_path)

    result = run_tool(ToolContext(pid=pid), "start_training", {"base_model": requested, "tier": "smoke"})

    assert result.get("status") == "awaiting_confirmation"
    assert result["ui_event"]["body"]["model_path"] == configured


def test_training_keeps_a_distinct_explicit_model_override(tmp_path: Path) -> None:
    pid = _project(base_model="/models/Qwen3.5-0.8B")
    _dataset(pid, tmp_path)

    result = run_tool(ToolContext(pid=pid), "start_training", {
        "base_model": "org/OtherModel", "preset": "qlora",
    })

    assert result.get("status") == "awaiting_confirmation"
    assert result["ui_event"]["body"]["model_path"] == "org/OtherModel"


def test_each_action_targets_the_pages_own_endpoint(tmp_path: Path) -> None:
    pid = _project()
    ds = _dataset(pid, tmp_path)
    db.create_version(pid, label="first")
    ctx = ToolContext(pid=pid)
    ev = run_tool(ctx, "build_dataset", {"force": True})["ui_event"]
    assert (ev["method"], ev["url"]) == ("GET", f"/api/projects/{pid}/data-prep/export?force=true")
    ev = run_tool(ctx, "build_smart_brain", {})["ui_event"]
    assert (ev["method"], ev["url"]) == ("POST", f"/api/projects/{pid}/brain/quick")
    ev = run_tool(ctx, "build_reworded_quiz", {})["ui_event"]
    assert ev["url"] == f"/api/projects/{pid}/testing/reworded-quiz" and ev["body"] == {"dataset_id": ds["id"]}
    ev = run_tool(ctx, "pin_version", {"label": "after training"})["ui_event"]
    assert ev["url"] == f"/api/projects/{pid}/versions" and ev["body"]["label"] == "after training"
    assert ev["body"]["parent_version_id"] == db.list_versions(pid)[0]["id"]      # continues the line


def test_mining_uses_parsed_files_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    pid = _project()
    files = [SimpleNamespace(id="s1", filename="a.pdf", ready=True), SimpleNamespace(id="s2", filename="b.pdf", ready=False)]
    monkeypatch.setattr("finetune_studio.knowledge.sources.list_sources", lambda p: files)
    ev = run_tool(ToolContext(pid=pid), "start_mining", {})["ui_event"]
    assert ev["body"] == {"source_ids": ["s1"], "kind": "mine"} and "a.pdf" in ev["detail"]
    assert "not parsed" in run_tool(ToolContext(pid=pid), "start_mining", {"source_ids": ["s2"]})["error"]


def test_hands_free_mode_is_reported_to_the_model_and_the_panel(client: TestClient, tmp_path: Path) -> None:
    pid = _project()
    _dataset(pid, tmp_path)
    assert client.get("/api/settings/guide").json()["action_mode"] == "ask"
    assert client.put("/api/settings/guide", json={"action_mode": "yolo"}).status_code == 400
    assert client.put("/api/settings/guide", json={"action_mode": "auto"}).json()["action_mode"] == "auto"
    r = run_tool(ToolContext(pid=pid), "build_reworded_quiz", {})
    assert r["status"] == "running" and r["ui_event"]["mode"] == "auto" and "Hands-free" in r["note"]
    from finetune_studio.guide.prompt import build_system_prompt
    assert "Action mode: hands-free" in build_system_prompt(ToolContext(pid=pid))


def test_the_guide_loop_forwards_the_action_to_the_panel(tmp_path: Path) -> None:
    from finetune_studio.guide.loop import run_guide_loop

    pid = _project()
    _dataset(pid, tmp_path)
    replies = iter(['<tool_call>{"name":"build_reworded_quiz","arguments":{}}</tool_call>', "Press Run to build it."])
    events = list(run_guide_loop(ToolContext(pid=pid), [{"role": "user", "content": "make a quiz"}],
                                 lambda messages: next(replies)))
    ui = [e for e in events if e["type"] == "ui"]
    assert ui and ui[0]["event"]["type"] == "action" and events[-1]["reply"] == "Press Run to build it."


def test_the_panel_runs_only_prepared_app_requests_and_limits_hands_free() -> None:
    js = (WEB / "static" / "js" / "guide.js").read_text(encoding="utf-8")
    base = (WEB / "templates" / "base.html").read_text(encoding="utf-8")
    assert 'id="guide-mode"' in base and 'value="auto">Hands-free' in base and 'value="ask">Ask before each action' in base
    assert "if (!/^\\/api\\//.test(ev.url)" in js and 'credentials: "same-origin"' in js
    assert f"var MAX_AUTO_ACTIONS = {actions.MAX_AUTO_ACTIONS};" in js
    assert '"ACTION_RESULT "' in js and '"ACTION_SKIPPED "' in js
    skip_action = js.split("function skipAction(ev, card) {", 1)[1].split("\n  }", 1)[0]
    assert 'handBack("ACTION_SKIPPED "' in skip_action
    assert "'/api/settings/guide'" in js.replace('"', "'")
    assert "data-guide-run" not in js or "guideRun" in js
