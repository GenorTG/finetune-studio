"""navigate / highlight / suggest_settings: allow-lists, validation and the UI events they emit."""
from __future__ import annotations

import pytest

from finetune_studio.guide import registry
from finetune_studio.guide.tools import ToolContext, run_tool

CTX = ToolContext(pid="abc12345", page="training")


@pytest.mark.parametrize(
    ("path", "page", "pid"),
    [
        ("/projects/abc12345/training", "training", "abc12345"),
        ("/projects/abc12345/data-prep?x=1#top", "pairs", "abc12345"),
        ("/projects/abc12345", "overview", "abc12345"),
        ("/projects", "projects", None),
        ("/", "dashboard", None),
        ("/settings", "settings", None),
        ("/models/explore", "models_library", None),
        ("/nonsense", None, None),
    ],
)
def test_context_from_path(path: str, page: str | None, pid: str | None) -> None:
    ctx = ToolContext.from_path(path)
    assert (ctx.page, ctx.pid) == (page, pid)
    assert ToolContext.from_path("/settings", pid="zz").pid == "zz"


def test_navigate_emits_a_ui_event_for_allow_listed_pages() -> None:
    out = run_tool(CTX, "navigate", {"page": "brain"})
    assert out["ok"] and out["url"] == "/projects/abc12345/brain"
    assert out["ui_event"] == {"type": "navigate", "page": "brain", "url": "/projects/abc12345/brain", "title": "Smart Brain"}


@pytest.mark.parametrize("page", ["admin", "../../etc/passwd", "http://evil.example", "", "TRAINING"])
def test_navigate_rejects_unknown_pages(page: str) -> None:
    out = run_tool(CTX, "navigate", {"page": page})
    assert "error" in out and "ui_event" not in out


def test_navigate_to_project_page_needs_a_project() -> None:
    out = run_tool(ToolContext(), "navigate", {"page": "training"})
    assert "needs a project" in out["error"] and "ui_event" not in out
    assert run_tool(ToolContext(), "navigate", {"page": "settings"})["url"] == "/settings"


def test_highlight_resolves_selector_from_the_registry_only() -> None:
    out = run_tool(CTX, "highlight", {"control_id": "pairs.mine_selected"})
    assert out["ok"] and out["opens_page_first"] is True
    assert out["ui_event"]["selector"] == "#dp-mine-selected"
    assert out["ui_event"]["url"] == "/projects/abc12345/data-prep"
    same = run_tool(CTX, "highlight", {"control_id": "training.start"})
    assert same["opens_page_first"] is False


@pytest.mark.parametrize("control", ["#evil", "document.body", "pairs.nonexistent", "", "training.start; drop"])
def test_highlight_rejects_unknown_controls(control: str) -> None:
    out = run_tool(CTX, "highlight", {"control_id": control})
    assert "error" in out and "ui_event" not in out
    assert "allowed" in out


def test_suggest_settings_prefills_validated_fields() -> None:
    out = run_tool(CTX, "suggest_settings", {
        "page": "training",
        "settings": {"num_epochs": "6", "lora_rank": 64, "learning_rate": "2e-4", "early_stopping": True},
    })
    assert out["ok"] and out["applied"] == {"num_epochs": 6, "lora_rank": 64, "learning_rate": "2e-4", "early_stopping": True}
    event = out["ui_event"]
    assert event["type"] == "prefill" and event["url"] == "/projects/abc12345/training"
    by_name = {f["name"]: f for f in event["fields"]}
    assert by_name["num_epochs"]["selector"] == '[name="num_epochs"]' and by_name["num_epochs"]["scope"] == "#train-form"
    assert "nothing was submitted" in out["note"]


def test_suggest_settings_rejects_unknown_fields_and_bad_values_but_keeps_good_ones() -> None:
    out = run_tool(CTX, "suggest_settings", {
        "page": "training",
        "settings": {"num_epochs": 6, "launch_missiles": 1, "lora_rank": 999999, "training_mode": "orpo"},
    })
    assert out["ok"] and list(out["applied"]) == ["num_epochs"]
    assert set(out["rejected"]) == {"launch_missiles", "lora_rank", "training_mode"}
    assert [f["name"] for f in out["ui_event"]["fields"]] == ["num_epochs"]


def test_suggest_settings_with_nothing_valid_emits_no_event() -> None:
    out = run_tool(CTX, "suggest_settings", {"page": "training", "settings": {"start": True}})
    assert out["ok"] is False and "ui_event" not in out and "start" in out["rejected"]
    assert "batch_size" in out["allowed"]


@pytest.mark.parametrize(
    "args",
    [
        {"page": "nope", "settings": {"a": 1}},
        {"page": "settings", "settings": {"a": 1}},       # a real page with no prefillable fields
        {"page": "training", "settings": {}},
        {"page": "training", "settings": "num_epochs=3"},
        {"page": "training"},
    ],
)
def test_suggest_settings_bad_requests_are_errors(args: dict) -> None:
    out = run_tool(CTX, "suggest_settings", args)
    assert "error" in out and "ui_event" not in out


def test_no_tool_approves_exports_or_deletes_and_starting_work_is_an_action() -> None:
    """Approve / reject / export / delete stay user-clicked. Work that starts something (mining, training, …) is an
    action tool: its call only PREPARES the request; the panel runs it after Run (or at once in hands-free mode)."""
    from finetune_studio.guide import actions
    from finetune_studio.guide.tools import TOOLS_CATALOG

    names = {t["name"] for t in TOOLS_CATALOG}
    assert not {n for n in names if any(w in n for w in ("approve", "export", "delete", "reject", "submit"))}
    starting = {n for n in names if n.startswith(("start_", "build_", "run_", "pin_"))}
    assert starting <= set(actions.ACTION_TOOLS)
    mutating = {"create_qa_pairs"}
    ui = {"navigate", "highlight", "suggest_settings"}
    read_only = names - mutating - ui - set(actions.ACTION_TOOLS)
    assert {"app_help", "project_overview", "list_datasets", "list_runs", "system_status",
            "recommend_training", "dataset_health", "explain_setting", "list_suites"} <= read_only

def test_registry_never_prefills_a_submit_control() -> None:
    for spec in registry.FIELDS.values():
        assert "btn" not in spec.selector and "start" not in spec.selector
