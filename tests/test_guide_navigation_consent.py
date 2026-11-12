"""The guide opens a page only when the user asked for that page (or said yes to its offer)."""
from __future__ import annotations

import json
from typing import Any

import pytest

from finetune_studio.guide import loop as guide_loop
from finetune_studio.guide.kb import load_kb
from finetune_studio.guide.loop import run_guide_loop
from finetune_studio.guide.navigation_consent import allowed_pages
from finetune_studio.guide.search import search_kb
from finetune_studio.guide.tools import ToolContext

OFFER = {"role": "assistant", "content": "Want me to open the Files page so you can upload your first document?"}

# (reply to the offer, pages it allows)
ACCEPTED = [
    ("Yes", {"files"}),
    ("Yes please", {"files"}),
    ("Sure!", {"files"}),
    ("ok", {"files"}),
    ("Please do", {"files"}),
    ("Do it", {"files"}),
    ("Go ahead", {"files"}),
    ("Yes, take me there", {"files"}),
    ("Yes — open it", {"files"}),
    ("take me there", {"files"}),
    ("Show me the Files page", {"files"}),
    ("Let's go to Files", {"files"}),
    ("I want to go to Files", {"files"}),
    ("Can you show me Smart Brain?", {"brain"}),
    ("Yes, open the Files page.", {"files"}),
    ("No thanks. Take me to Training instead.", set()),  # a refusal first: nothing moves
]

REFUSED = [
    "OK, open source models - which work best?",
    "Open-weight models: which one should I pick?",
    "Open files: what formats are supported?",
    "Can you open-source the model?",
    "Don't, please. Open nothing.",
    "No, don't open the Files page.",
    "What does the Testing page do?",
    "Show me how to export a gguf",
]


def _turn(text: str, *, offer: bool = True) -> list[dict]:
    return [*([OFFER] if offer else []), {"role": "user", "content": text}]


@pytest.mark.parametrize(("reply", "pages"), ACCEPTED)
def test_reply_to_an_offer_allows_exactly_the_page(reply: str, pages: set[str]) -> None:
    assert allowed_pages(_turn(reply)) == pages


@pytest.mark.parametrize("text", REFUSED)
def test_questions_and_refusals_never_allow_navigation(text: str) -> None:
    assert allowed_pages(_turn(text)) == set()
    assert allowed_pages(_turn(text, offer=False)) == set()


@pytest.mark.parametrize("reply", ["Yes", "Sure!", "Go ahead", "take me there", "Yes — open it"])
def test_a_bare_yes_without_an_offer_opens_nothing(reply: str) -> None:
    assert allowed_pages(_turn(reply, offer=False)) == set()


def test_an_explicit_request_needs_no_offer() -> None:
    assert allowed_pages(_turn("Open Settings.", offer=False)) == {"settings", "project_settings"}
    assert allowed_pages(_turn("take me to training", offer=False)) == {"training"}
    assert allowed_pages(_turn("Can you go to the project models page?", offer=False)) == {"project_models"}


def test_only_the_latest_offer_counts() -> None:
    messages = [
        {"role": "assistant", "content": "Shall I take you to Training?"},
        {"role": "user", "content": "Not yet."},
        {"role": "assistant", "content": "Do you want me to open Smart Brain instead?"},
        {"role": "user", "content": "Yes please"},
    ]
    assert allowed_pages(messages) == {"brain"}


# ── in the loop ──────────────────────────────────────────────────────────────

class Script:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)

    def __call__(self, _messages: list[dict]) -> str:
        return self.replies.pop(0) if self.replies else ""


def _call(page: str) -> str:
    return "<tool_call>" + json.dumps({"name": "navigate", "arguments": {"page": page}}) + "</tool_call>"


def _run(messages: list[dict], page: str, monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], dict[str, Any]]:
    executed: list[str] = []
    monkeypatch.setattr(guide_loop, "run_tool", lambda ctx, name, args: executed.append(args["page"]) or {})
    events = list(run_guide_loop(ToolContext(pid="project-1", page="project"), messages, Script(_call(page), "Done.")))
    return executed, next(e for e in events if e["type"] == "tool_result")["result"]


@pytest.mark.parametrize("reply", ["Yes", "Go ahead", "Yes — open it"])
def test_loop_navigates_to_the_offered_page_on_a_bare_yes(reply: str, monkeypatch: pytest.MonkeyPatch) -> None:
    executed, _ = _run(_turn(reply), "files", monkeypatch)
    assert executed == ["files"]


def test_loop_refuses_a_different_page_than_the_one_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    executed, result = _run(_turn("Yes"), "training", monkeypatch)
    assert executed == []
    assert "not 'training'" in result["error"]


@pytest.mark.parametrize("text", REFUSED)
def test_loop_does_not_navigate_on_a_question(text: str, monkeypatch: pytest.MonkeyPatch) -> None:
    executed, result = _run(_turn(text), "files", monkeypatch)
    assert executed == []
    assert result["error"].startswith("The user asked for information")


# ── Project models guide entry ───────────────────────────────────────────────

def test_project_models_kb_describes_the_real_row_actions() -> None:
    entry = load_kb()["project-models"]
    workflow = entry.sections["Workflow"]
    assert "Open in inference" in workflow and "Copy path" in workflow
    assert "no remove action" in workflow
    assert "Models** tile" in workflow and "no separate Models link" in workflow


@pytest.mark.parametrize("question", ["copy the path of my merged model", "where are my trained models"])
def test_app_help_finds_project_models(question: str) -> None:
    assert "project-models" in [h["entry"] for h in search_kb(question)][:2]


def test_export_questions_still_go_to_export() -> None:
    assert search_kb("how do I export a gguf")[0]["entry"] == "export"
