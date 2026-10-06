"""App-wide guidance tools remain project-scoped and read-only."""
from __future__ import annotations


def test_app_guide_explains_supported_routes_and_preference_format() -> None:
    from finetune_studio.webui.routes.data_prep_chat import _run_tool

    result = _run_tool("unused", "get_app_guide", {})

    assert [item["step"] for item in result["workflow"]] == [1, 2, 3, 4, 5, 6]
    assert "chosen" in result["routes"]["dpo"]
    assert "tools" in result["routes"]["tool_sft"]
    assert "raw text" in result["routes"]["continued_pretraining"]
    assert "ORPO" in result["routes"]["not_supported_yet"]
    assert "RAG" in result["routes"]["rag"]
    # The guide now navigates/highlights/pre-fills through allow-listed tools, but never presses buttons.
    assert "never presses" in result["navigation_note"]
    assert {entry["id"] for entry in result["kb"]} >= {"training", "pairs", "rag"}


def test_readiness_tool_reports_project_specific_next_step(client) -> None:
    from finetune_studio import db
    from finetune_studio.webui.routes.data_prep_chat import _run_tool

    project = db.create_project(name="Readiness tool probe")
    pid = project["id"] if isinstance(project, dict) else project
    result = _run_tool(pid, "inspect_project_readiness", {})

    assert result["project"]["id"] == pid
    assert result["qa_pairs"] == {"total": 0, "pending": 0, "approved": 0, "rejected": 0}
    assert result["next_step"] == "Upload and parse source files."
