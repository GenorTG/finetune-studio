"""App knowledge base: loader, template-verified controls, and ``app_help`` relevance."""
from __future__ import annotations

from pathlib import Path

import pytest

from finetune_studio.guide import registry
from finetune_studio.guide.kb import SECTION_ORDER, load_kb, table_of_contents
from finetune_studio.guide.search import search_kb

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "templates"


def _template_text(page_key: str) -> str:
    page = registry.PAGES[page_key]
    return (TEMPLATES / page.template).read_text(encoding="utf-8") + (TEMPLATES / "base.html").read_text(encoding="utf-8")


def test_kb_loads_one_entry_per_page_or_feature() -> None:
    kb = load_kb()
    assert len(kb) >= 18
    # Every page that has user-facing workflow is covered by at least one entry.
    covered = {e.page for e in kb.values()}
    required = {"projects", "files", "pairs", "rag", "training", "testing", "benchmarks", "export",
                "chat", "models_library", "inference", "settings", "wizard", "overview"}
    assert required <= covered


def test_every_entry_has_all_sections_and_a_known_page() -> None:
    for entry in load_kb().values():
        assert entry.page in registry.PAGES, entry.id
        assert entry.controls_page in registry.PAGES, entry.id
        assert set(entry.sections) == set(SECTION_ORDER), f"{entry.id}: {sorted(entry.sections)}"
        assert all(text.strip() for text in entry.sections.values()), entry.id
        assert entry.route.startswith("/"), entry.id


@pytest.mark.parametrize("entry_id", sorted(load_kb()))
def test_key_controls_exist_in_the_page_template(entry_id: str) -> None:
    """Every element id / field name a KB entry tells the user about is in the real template."""
    entry = load_kb()[entry_id]
    html = _template_text(entry.controls_page)
    assert entry.controls, f"{entry_id} lists no key controls"
    for selector in entry.controls:
        tokens = registry.selector_tokens(selector)
        assert tokens, f"{entry_id}: {selector!r} is not #id or [name=...]"
        for kind, token in tokens:
            assert f'{kind}="{token}"' in html, f"{entry_id}: {kind} {token!r} not found in {entry.page} template"


def test_table_of_contents_matches_entries() -> None:
    toc = table_of_contents()
    assert [row["id"] for row in toc] == list(load_kb())
    assert all(row["page"].startswith("/") for row in toc)


# (question, expected KB entry id) — phrased the way a new user would ask.
QUESTIONS = [
    ("how do I train a model on my documents?", {"training", "app-map", "wizard"}),
    ("what epochs should I use", {"training-settings"}),
    ("how many optimizer steps are enough", {"training-settings", "dataset-quality"}),
    ("which route should I pick for preference data with chosen and rejected answers", {"training-routes"}),
    ("what is the difference between sft and dpo", {"training-routes"}),
    ("how do I use tool calling data", {"training-routes"}),
    ("what does held-out evaluation measure", {"testing"}),
    ("difference between from memory and with context scores", {"testing"}),
    ("which gguf quant should I export", {"export"}),
    ("how do I make good question answer pairs", {"dataset-quality", "pairs"}),
    ("upload a pdf and check it parsed", {"files"}),
    ("what is rag and how do I build an index", {"rag"}),
    ("how do I run on cpu only", {"settings"}),
    ("why is start training disabled", {"troubleshooting", "training"}),
    ("where do I download a base model", {"models"}),
    ("what is the helper model", {"helper-model"}),
    ("how do I load a model and set the context length", {"inference"}),
    ("what does the quick work wizard do", {"wizard"}),
]


@pytest.mark.parametrize(("question", "expected"), QUESTIONS)
def test_app_help_finds_the_right_entry(question: str, expected: set[str]) -> None:
    hits = search_kb(question)
    assert hits, question
    assert hits[0]["entry"] in expected, f"{question!r} → {[h['entry'] for h in hits]}"


def test_app_help_returns_nothing_for_gibberish() -> None:
    assert search_kb("zxqv blorf wibble") == []


def test_app_help_results_are_bounded_and_unique() -> None:
    hits = search_kb("training", limit=10)
    assert len(hits) <= 5
    assert len({h["entry"] for h in hits}) == len(hits)
    assert all(len(h["text"]) <= 1200 for h in hits)


def test_kb_claims_match_the_preset_advisor() -> None:
    """The tier floors the KB quotes are the advisor's real constants."""
    from finetune_studio.training.preset_advisor import _TIER_ANCHORS

    text = load_kb()["training-settings"].sections["Workflow"]
    for tier, floor in (("balanced", 400), ("precision", 700), ("overkill", 1000)):
        assert _TIER_ANCHORS[tier]["steps_floor"] == floor
        assert str(floor) in text
