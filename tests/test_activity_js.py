"""Regression: activity drawer keeps expanded rows + filters across live re-render.

Live /api/activity snapshots (SSE, with silent poll fallback) used to wipe
body.innerHTML and reset every row to aria-expanded=false / no .open class.
Expansion must be keyed by a stable task identity (run_id / id /
kind+project+url+name), not array index, and filter selects must remain the
source of truth across re-renders.
"""
from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_ACTIVITY_JS = (
    _ROOT / "src" / "finetune_studio" / "webui" / "static" / "js" / "activity.js"
)


def _src() -> str:
    return _ACTIVITY_JS.read_text(encoding="utf-8")


def test_task_identity_helper_exists() -> None:
    src = _src()
    assert "function taskIdentity(" in src
    assert "run_id" in src
    # Must not key expansion off poll-unstable started_at for training.
    identity_block = src.split("function taskIdentity(", 1)[1].split(
        "function fmtTimeAgo(", 1
    )[0]
    # Strip comments before asserting — a "don't use started_at" note is fine.
    code_only = re.sub(r"//.*?$", "", identity_block, flags=re.MULTILINE)
    assert "started_at" not in code_only
    assert ".join(" in code_only


def test_expanded_set_restored_on_render() -> None:
    src = _src()
    assert "_expanded" in src
    assert 'classList.toggle("open")' in src
    assert "_expanded.add" in src
    assert "_expanded.delete" in src
    # Re-render must paint open class + aria-expanded from the set.
    assert re.search(r'isOpen\s*\?\s*"open"', src) or '${isOpen ? "open"' in src
    assert 'aria-expanded="${isOpen ? "true" : "false"}"' in src
    assert "data-task-key=" in src


def test_render_uses_filtered_not_raw_tasks_index() -> None:
    src = _src()
    render = src.split("function renderTasks(", 1)[1].split(
        "function updateBadge(", 1
    )[0]
    assert "tasks.filter(activityMatchesFilter)" in render
    assert "filtered" in render
    # Must map the filtered list, not the raw tasks array (index-keyed bug).
    assert "filtered" in render.split("bodyEl.innerHTML", 1)[1].split(".map(", 1)[0]
    assert "data-idx=" not in render


def test_filter_state_synced_from_dom_and_exported() -> None:
    src = _src()
    assert "function syncFilterFromDom(" in src
    assert "activity-filter-type" in src
    assert "activity-filter-project" in src
    assert "activity-filter-status" in src
    # Project select rebuild must restore prior selection.
    populate = src.split("function activityPopulateProjectFilter(", 1)[1].split(
        "function activityMatchesFilter(", 1
    )[0]
    assert "sel.value = current" in populate
    assert "window.activityApplyFilter" in src


def test_activity_prefers_sse_events() -> None:
    src = _src()
    assert "/api/activity/events" in src
    assert "setInterval(refresh, 2000)" not in src
