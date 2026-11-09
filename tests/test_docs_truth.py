"""Public docs and the Guide say what the app does (B6-03..B6-08 audit, 2026-10-31).

Each claim below was false in shipped docs: an export "coverage fallback" that does not exist (export blocks or is
forced), a Guide denying the Pairs page's preference-pair builder, a tutorial requiring a local model for Smart Brain
chat (the API assistant can answer), old 2/3/4/5 step numbers (the nav says 1/2/3/4), an update ``--check`` that
"pulls and migrates" (it does neither) and an Apply update "not yet tested" that was run end to end on 2026-10-28.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from finetune_studio.guide import registry
from finetune_studio.guide.search import search_kb

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = {name: (ROOT / name).read_text(encoding="utf-8")
          for name in ("README.md", "docs/TUTORIAL.md", "docs/index.html", "docs/DEPLOYMENT.md", "docs/INSTALL.md")}
KB = ROOT / "src/finetune_studio/guide/kb"


@pytest.mark.parametrize("name", sorted(PUBLIC))
def test_no_extractive_coverage_fallback_is_advertised(name: str) -> None:
    text = PUBLIC[name]
    assert "extractive pairs" not in text and "Chunk-coverage fallback" not in text


def test_export_gate_is_described_as_blocking_unless_forced() -> None:
    assert "export anyway" in PUBLIC["README.md"] and "export anyway" in PUBLIC["docs/TUTORIAL.md"]


def test_tutorial_step_labels_match_the_project_nav() -> None:
    nav = (ROOT / "src/finetune_studio/webui/templates/base.html").read_text(encoding="utf-8")
    assert ">1 files<" in nav and ">2 Smart Brain<" in nav and ">3 test<" in nav and ">4 use it<" in nav   # the sub-nav
    labels = dict((m.group(2), m.group(1)) for m in re.finditer(r"\(step \*\*(\d) · ([a-z ]+)\*\*\)",
                                                                  PUBLIC["docs/TUTORIAL.md"]))
    assert labels == {"files": "1", "smart brain": "2", "test": "3", "use it": "4"}


def test_pages_captions_use_the_nav_step_numbers() -> None:
    page = PUBLIC["docs/index.html"]
    for caption in ("FILES · STEP 1", "SMART BRAIN · STEP 2", "TESTING · STEP 3", "MODELS · STEP 4"):
        assert caption in page, caption
    assert not re.search(r"(PAIRS|TRAINING|TESTING|MODELS) · STEP 5", page)


def test_tutorial_explains_both_smart_brain_answer_sources() -> None:
    text = PUBLIC["docs/TUTORIAL.md"]
    assert "needs a chat\n   model loaded" not in text
    section = text[text.index("**Ask your documents**"):][:700]
    assert "Answer with" in section and "API connection" in section and "sent to that provider" in section


def test_update_check_is_read_only_and_apply_update_is_not_called_untested() -> None:
    dep = PUBLIC["docs/DEPLOYMENT.md"]
    flag = dep[dep.index("Flags: `--check`"):][:300]
    assert "no pull" in flag and "no migration" in flag and "pull + migrations" not in dep
    assert "not yet tested end to end" not in PUBLIC["docs/index.html"]


def test_heretic_is_listed_beside_abliterated_in_export_docs() -> None:
    for name in ("docs/TUTORIAL.md", "docs/DEPLOYMENT.md", "docs/index.html"):
        assert "Heretic" in PUBLIC[name], name


def test_guide_documents_and_can_point_at_study_set_and_preference_builders() -> None:
    routes = (KB / "08-training-routes.md").read_text(encoding="utf-8")
    assert "does not author comparisons" not in routes and "Preference-comparison authoring" not in routes
    assert "Preference pairs (DPO)" in routes
    pairs = (KB / "03-pairs.md").read_text(encoding="utf-8")
    assert "Build study set" in pairs and "Build preference pairs" in pairs
    selectors = {c.selector for c in registry.CONTROLS.values()}
    for sel in ("#dp-study-card", "#dp-study-build", "#dp-pref-card", "#dp-pref-build", "#dp-pref-kinds"):
        assert sel in selectors, sel


@pytest.mark.parametrize("question", [
    "how do I build preference pairs from my approved answers",
    "what is the study set and how do I build it",
])
def test_app_help_routes_builder_questions_to_the_pairs_page(question: str) -> None:
    hits = search_kb(question)
    assert hits and "pairs" in [h["entry"] for h in hits[:2]], [h["entry"] for h in hits]


def test_install_and_storage_docs_match_the_code() -> None:
    install, dep = PUBLIC["docs/INSTALL.md"], PUBLIC["docs/DEPLOYMENT.md"]
    assert "install.sh --python" not in install and "PYTHON_VERSION=3.12 ./install.sh" in install
    assert "projects.db" not in install and "`data/finetune_studio.db` in the checkout" in install
    assert "~/.finetune-studio/hf_models/" in install and "~/.finetune-studio/hf_models/" in dep
    assert "rm -rf ~/.cache/huggingface      # downloaded models" not in install
    assert "`$FTS_DB` overrides" not in dep and "`$FTS_DB` moves this one" in dep
    assert "/api/system/update/events" in dep and "every 2 s" not in dep


def test_docs_use_the_ui_labels_and_export_sizes() -> None:
    tutorial, readme = PUBLIC["docs/TUTORIAL.md"], PUBLIC["README.md"]
    assert "**search inside files**" in tutorial and "**in knowledge**" not in tutorial
    assert "Ask the model (grounded)" not in tutorial + readme
    for text in (tutorial, readme):
        assert "2.3 GB" not in text and "4.5 GB" in text
    assert "not the\n      full HuggingFace dataset downloads" not in PUBLIC["docs/index.html"]
