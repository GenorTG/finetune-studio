"""Public-bot option "Hide file names in answers": the package's answers cite no source files, the answer model never
sees them, and the sources reach the operator only on request (export config, CLI, Export card, server)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from finetune_studio.smart_brain import standalone_server as srv
from finetune_studio.smart_brain.export_config import BrainExportConfig
from finetune_studio.smart_brain.identity import GROUNDING

HITS = [{"rank": 1, "chunk_id": "a", "filename": "public/legal/terms_of_service_v4_2025.docx",
         "text": "Returns are accepted within 30 days.", "rrf_score": 0.03, "document_id": "d1"},
        {"rank": 2, "chunk_id": "b", "filename": "Harrowgate_deal_sheet_OPP-24-0917.xlsx", "text": "120 units",
         "rrf_score": 0.02, "document_id": "d2"}]


@pytest.mark.parametrize(("answer", "clean"), [
    ("Returns are accepted within 30 days [terms_of_service_v4_2025.docx].", "Returns are accepted within 30 days."),
    ("Yes [a.pdf, b.docx], within 30 days.", "Yes, within 30 days."),
    ("See terms_of_service_v4_2025.docx for more.", "See our documents for more."),
    ("Within 30 days [1].", "Within 30 days [1]."),
])
def test_strip_file_names(answer: str, clean: str) -> None:
    assert srv.strip_file_names(answer, ["terms_of_service_v4_2025.docx"]) == clean


def test_the_identity_rule_is_swapped_and_the_model_sees_no_file_names() -> None:
    assert srv.CITE_RULE in GROUNDING and srv.CITE_RULE in srv.DEFAULT_SYSTEM_PROMPT    # the rule this swaps exists
    prompt = srv.hide_names_prompt(f"You are Tarrowfield. {GROUNDING}")
    assert srv.CITE_RULE not in prompt and srv.NO_NAMES_RULE in prompt
    system = srv._anonymous_turn(HITS, 4000, f"You are Tarrowfield. {GROUNDING}", "")
    assert "terms_of_service" not in system and "Harrowgate_deal_sheet" not in system
    assert "[1] (" in system and "Returns are accepted within 30 days." in system


def _corpus(hide: bool) -> MagicMock:
    corpus = MagicMock(spec=srv.Corpus)
    corpus.cfg = {**srv.DEFAULT_CONFIG, "answer": True, "llm_model": "m", "llm_base_url": "http://x/v1",
                  "hide_file_names": hide}
    corpus.identity = {"system_prompt": f"You are Tarrowfield. {GROUNDING}"}
    corpus.search.return_value = HITS
    corpus._chat.return_value = "Within 30 days [terms_of_service_v4_2025.docx]."
    return corpus


def test_package_answer_hides_names_and_sources_unless_asked() -> None:
    corpus = _corpus(True)
    out = srv.Corpus.answer(corpus, "How long do I have to return a machine?")
    assert out["answer"] == "Within 30 days." and out["sources"] == [] and out["sources_hidden"] is True
    system = corpus._chat.call_args.args[0][0]["content"]
    assert "terms_of_service" not in system
    asked = srv.Corpus.answer(corpus, "How long do I have to return a machine?", include_sources=True)
    assert asked["answer"] == "Within 30 days." and [s["document_id"] for s in asked["sources"]] == ["d1", "d2"]
    assert asked["sources_hidden"] is False


def test_off_by_default_keeps_citations_and_sources() -> None:
    assert srv.DEFAULT_CONFIG["hide_file_names"] is False and BrainExportConfig().hide_file_names is False
    out = srv.Corpus.answer(_corpus(False), "How long?")
    assert out["answer"].endswith("[terms_of_service_v4_2025.docx].") and len(out["sources"]) == 2
    assert "sources_hidden" not in out


def test_config_env_export_and_cli() -> None:
    assert srv.resolve_config({}, {"RAG_HIDE_FILE_NAMES": "1"}, {})[0]["hide_file_names"] is True
    assert BrainExportConfig(hide_file_names=True).runtime_config()["hide_file_names"] is True
    from finetune_studio.cli import _parser
    args = _parser.build_parser().parse_args(["brain", "export", "--project", "p", "--hide-file-names"])
    assert args.hide_file_names is True


def test_the_rag_answer_tool_offers_include_sources() -> None:
    assert "include_sources" in srv._ANSWER_TOOL["inputSchema"]["properties"]


def test_the_export_card_drives_the_option() -> None:
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates/brain.html").read_text()
    assert 'id="mcp-hide-names"' in html and "hide_file_names: g('mcp-hide-names').checked" in html
    from finetune_studio.webui.routes.brain_export import StandaloneRequest
    assert StandaloneRequest(hide_file_names=True).hide_file_names is True
