"""Coverage-fill questions must be self-contained and specific, or the pair is dropped.

Regression for the "What does the source say about “It”?" rows that capped
suite/held-out scores (``.tmp/e2e-final/REPORT.md``): extractive questions are
now built from a distinctive subject plus the section/file scope, and a chunk
with no such question is reported as ``no_specific_question`` instead of
getting a vague pair.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from finetune_studio.data import project_filesystem as pfs
from finetune_studio.data.prep.coverage_fill import (
    _make_pairs_from_chunk,
    _plan_pairs,
    fill_all_project_gaps,
    fill_coverage_gaps,
    fill_sources_gaps,
)
from finetune_studio.data.prep.coverage_question import (
    answer_is_standalone,
    build_question,
    dominant_entity,
    is_near_duplicate,
    proper_words,
    question_issues,
    scope_for,
    split_sections,
    title_from_filename,
)

DOCS = Path(__file__).parent / "fixtures" / "coverage_docs"

_PRONOUN_SUBJECT = re.compile(r"(?:about|regarding) [“\"]?(?:it|this|they|that|these|those|there)\b", re.IGNORECASE)


@pytest.fixture()
def proj(tmp_path, monkeypatch):
    monkeypatch.setenv("FTS_ROOT", str(tmp_path))
    from finetune_studio.data.fs import paths

    monkeypatch.setattr(paths, "_ROOT", tmp_path)
    monkeypatch.setattr(paths, "_PROJECTS", tmp_path / "projects")
    return "qproj"


def _chunks(text: str) -> list[str]:
    """Blank-line blocks, with heading-only blocks glued to the block that follows."""
    out: list[str] = []
    carry = ""
    for block in (b.strip() for b in re.split(r"\n\s*\n", text)):
        if not block:
            continue
        if len(block.split()) <= 6 and "\n" not in block and not re.search(r"[.!?]$", block):
            carry += block + "\n"
            continue
        out.append(carry + block)
        carry = ""
    return out or [text]


# ── the reported failures ────────────────────────────────────────────────

@pytest.mark.parametrize("sentence", [
    "It boils a full kettle in about three minutes using a 2200 W heating element.",
    "They agreed that this should be done soon.",
    "This is required by the Concord.",
    "However, the lid stays closed.",
    "Their oath-stone is carved from margin-stone.",
])
def test_sentences_leaning_on_context_are_not_standalone_answers(sentence: str) -> None:
    assert not answer_is_standalone(sentence)
    assert build_question(sentence, scope="Handbook", variant=0) is None


@pytest.mark.parametrize("question", [
    "What does the source say about “It”?",
    "What does the source say about they?",
    "What is stated in the source regarding Rinse with fresh water three?",
    "What does the source say about never use vinegar, because it?",
    "What does the source say about the?",
    "What does the source say about ab?",
    "What does the source say about …?",
])
def test_known_bad_questions_are_flagged(question: str) -> None:
    assert question_issues(question, scope="Handbook")


def test_generic_subject_needs_a_scope() -> None:
    q = "What does the source say about the body?"
    assert "generic_without_scope" in question_issues(q)
    assert question_issues("In “Specifications”, what does the source say about the body?",
                           scope="Specifications") == []


def test_subject_that_only_repeats_the_scope_is_rejected() -> None:
    q = "In “Customer support”, what does the source say about customer support?"
    assert "repeats_scope" in question_issues(q, scope="Customer support")


# ── chunk shapes ─────────────────────────────────────────────────────────

SHAPES: dict[str, tuple[str, str]] = {
    "headed_sections": (
        (
            "Warranty\nThe warranty covers defects in the heating element and the lid hinge. "
            "Damage caused by dropping the kettle is not covered.\n"
        ),
        "",
    ),
    "markdown_headings": (
        "# Tide Guide\n## Gauging the tide\nThe Orrin Deep tide rises 4.2 metres at the spring moon.\n",
        "",
    ),
    "flat_prose_named": (
        (
            "The Concord pays a Ledger-Keeper 1875 crowns per annum plus 71 measures of black-whiskey. "
            "A Salt-Speaker is elected by the nine Tide-Wardens of Orrin Deep every seventh winter."
        ),
        "",
    ),
    "flat_prose_filename_scope": (
        (
            "The body is brushed stainless steel and the cord is 0.75 m long. "
            "Urgent safety issues are escalated the same day."
        ),
        "aurora_kettle_handbook.txt",
    ),
    "bullets": (
        (
            "Release notes\n- Delta uploads cut transfer time for large files by up to 60 percent.\n"
            "- The Quiet Hours setting pauses background syncing between 22:00 and 07:00.\n"
        ),
        "",
    ),
    "numbered_list": (
        (
            "Setup\n1. Plug the base into a grounded socket near the Aurora Kettle.\n"
            "2. Fill the kettle to the MAX line before the first boil.\n"
        ),
        "",
    ),
    "single_long_sentence": (
        (
            "The Vaelindrath Concord archive is kept in the Lantern Vault beneath Harrowgate, "
            "and it may be opened only at dusk by a sworn Ledger-Keeper."
        ),
        "",
    ),
    "unicode_names": (
        (
            "Zażółć Gęślą Jaźń is the founding poem of the Łódź Guild of Cartographers. "
            "Cartographers of the Łódź Guild swear it on the first frost."
        ),
        "",
    ),
}


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_chunk_shapes_emit_only_self_contained_grounded_questions(name: str) -> None:
    text, filename = SHAPES[name]
    pairs, _ = _plan_pairs(text, seen_questions=set(), filename=filename)
    assert pairs, f"{name}: expected at least one specific question"
    scope_hints = {s.lower() for h, _p in split_sections(text) for s in [h] if s}
    for q, a in pairs:
        assert _PRONOUN_SUBJECT.search(q) is None, q
        assert "…" not in q and "|" not in q, q
        assert a.rstrip(".").lower() in re.sub(r"\s+", " ", text).lower()
        assert answer_is_standalone(a), a
        # the production gate agrees, and a bare question names something distinctive
        scope = scope_for("", filename, text) or ("x" if q.startswith("In ") else "")
        assert question_issues(q, scope=scope, proper=proper_words(text)) == [], q
        assert scope_hints or filename or any(c.isupper() or c.isdigit() for c in q[1:]), q


@pytest.mark.parametrize("text", [
    ("the committee met on tuesday and it was decided that the matter would be revisited. "
     "they agreed that this should be done soon, and that those involved would be notified."),
    "It is noted that there are several issues. See above for details. This was discussed.",
    "Yes. See above. OK then.",
    "~~~ *** ###",
    "unit_id | class | score | 12 | 34 | 56. A, B, C, D, E, F, G, H, 1, 2, 3, 4.",
])
def test_unspecific_chunks_yield_no_pairs(text: str) -> None:
    assert _make_pairs_from_chunk(text, seen_questions=set()) == []


def test_pronoun_opener_is_dropped_but_named_sentence_in_same_chunk_survives() -> None:
    text = (
        "Aurora Kettle Handbook\n"
        "It boils a full kettle in about three minutes. "
        "The Aurora Kettle holds 1.6 L of water when filled to the MAX line."
    )
    pairs = _make_pairs_from_chunk(text, seen_questions=set())
    assert len(pairs) == 1
    assert pairs[0][1].startswith("The Aurora Kettle holds")


def test_heading_glued_to_body_never_leaks_into_the_subject() -> None:
    text = (
        "Aurora Kettle Support Handbook\n\n"
        "Customer support replies to every email within two business days. "
        "Support is available Monday to Friday, 9:00 to 17:00 Central European Time."
    )
    for q, _a in _make_pairs_from_chunk(text, seen_questions=set()):
        assert "Handbook Customer" not in q, q


# ── scope / dedupe helpers ───────────────────────────────────────────────

def test_scope_prefers_heading_then_file_then_entity() -> None:
    text = "Aurora Kettle ships in blue. The Aurora Kettle is sold worldwide."
    assert scope_for("Warranty", "x.txt", text) == "Warranty"
    assert scope_for("Overview", "aurora_kettle_handbook.txt", text) == "Aurora Kettle Handbook"
    assert scope_for("", "scan_0042.txt", text) == "Aurora Kettle"
    assert scope_for("", "", "no names here at all.") == ""


@pytest.mark.parametrize("name, expected", [
    ("aurora_kettle_handbook.pdf", "Aurora Kettle Handbook"),
    ("4060452b9f1c3a77.txt", ""),
    ("document.txt", ""),
    ("scan_0042.txt", ""),
    ("Release-Notes-v4.md", "Release Notes v4"),
])
def test_title_from_filename(name: str, expected: str) -> None:
    assert title_from_filename(name) == expected


def test_dominant_entity_ignores_sentence_openers() -> None:
    assert dominant_entity("The Concord pays. The Concord archives. If Concord fails.") == "Concord"


def test_near_duplicate_questions_are_collapsed() -> None:
    a = "In “Warranty”, what does the source say about the heating element?"
    b = "In “Warranty”, what is stated regarding heating element?"
    c = "In “Warranty”, what does the source say about the lid hinge?"
    assert is_near_duplicate(b, [a])
    assert not is_near_duplicate(c, [a])


def test_no_two_pairs_in_a_chunk_ask_about_the_same_subject() -> None:
    text = (
        "Warranty\nThe warranty covers the heating element. "
        "The heating element is covered for three years from the purchase date. "
        "The lid hinge is covered only against manufacturing defects."
    )
    pairs = _make_pairs_from_chunk(text, seen_questions=set())
    qs = [q for q, _a in pairs]
    for i, q in enumerate(qs):
        assert not is_near_duplicate(q, qs[:i]), qs


def test_seen_questions_from_other_chunks_are_respected() -> None:
    text = "Warranty\nThe lid hinge is covered only against manufacturing defects."
    first = _make_pairs_from_chunk(text, seen_questions=set())
    assert first
    assert _make_pairs_from_chunk(text, seen_questions={first[0][0]}) == []


def test_proper_words_use_bullets_and_repetition_evidence() -> None:
    text = "- Fixed a crash.\n- Fixed a leak.\nPip is a penguin. Pip likes fish."
    names = proper_words(text)
    assert "pip" in names
    assert "fixed" not in names  # a repeated sentence-opening verb is not a name
    q = build_question("Fixed a crash when renaming a folder.", scope="Fixes", variant=0, proper=names)
    assert q is None or "Fixed" not in q


# ── the drop is surfaced, not hidden ─────────────────────────────────────

VAGUE = (
    "It is noted that there are several issues. They agreed that this should be done soon. "
    "This was discussed at length by those present."
)


def test_fill_drops_vague_chunk_and_reports_the_reason(proj: str) -> None:
    result = fill_coverage_gaps(proj, "vaguesrc", chunk_texts={1: VAGUE}, filename="scan_0042.txt")
    assert result.pairs_created == 0 and result.chunks_filled == 0
    assert result.no_specific_question == 1
    assert result.chunks_still_uncovered == [
        {"chunk_idx": 1, "chars": len(VAGUE), "reason": "no_specific_question"}
    ]
    assert result.as_dict()["no_specific_question"] == 1
    assert pfs.list_qa_pairs(proj, source_id="vaguesrc") == []


def test_fill_all_surfaces_dropped_chunks_to_the_export_gate(proj: str) -> None:
    sid = "mixedsrc0001"
    good = "The Concord pays a Ledger-Keeper 1875 crowns per annum plus 71 measures of black-whiskey."
    pfs.write_qa_source(proj, {
        "id": sid, "sha256": "mixedsha1234", "filename": "scan_0042.txt",
        "mime_type": "text/plain", "char_count": len(good) + len(VAGUE),
        "chunk_count": 2, "parser": "text_v1", "status": "ready", "data_path": "", "path": "",
    })
    d = pfs.file_dir(proj, "mixedsha1234")
    (d / "chunks").mkdir()
    (d / "chunks" / "0000.txt").write_text(good, encoding="utf-8")
    (d / "chunks" / "0001.txt").write_text(VAGUE, encoding="utf-8")

    for summary in (fill_all_project_gaps(proj), fill_sources_gaps(proj, [sid])):
        assert summary["no_specific_question"] >= 1
        unc = [u for u in summary["uncovered_chunks"] if u["source"] == sid]
        assert [u["chunk_idx"] for u in unc] == [2]  # gate blocks export on exactly this chunk
        assert unc[0]["reason"] == "no_specific_question"
    rows = pfs.list_qa_pairs(proj, source_id=sid)
    assert {r["chunk_idx"] for r in rows} == {1}
    assert all("source say about “It”" not in r["question"] for r in rows)


def test_filename_scope_is_stamped_into_stored_questions(proj: str) -> None:
    text = "The body is brushed stainless steel and the cord is 0.75 m long."
    fill_coverage_gaps(proj, "scopesrc", chunk_texts={1: text}, filename="aurora_kettle_handbook.txt")
    rows = pfs.list_qa_pairs(proj, source_id="scopesrc")
    assert rows and all("Aurora Kettle Handbook" in r["question"] for r in rows)


# ── quality table over the five fictional documents ──────────────────────

def _quality(files: list[Path]) -> tuple[int, int, int]:
    """(questions, bad questions, chunks left uncovered) over every chunk of ``files``."""
    total = bad = uncovered = 0
    for f in files:
        for chunk in _chunks(f.read_text(encoding="utf-8")):
            pairs = _make_pairs_from_chunk(chunk, seen_questions=set(), filename=f.name)
            uncovered += not pairs
            for q, _a in pairs:
                total += 1
                bad += bool(question_issues(q, scope="x" if q.startswith("In ") else ""))
    return total, bad, uncovered


def test_five_document_corpus_has_no_vague_questions() -> None:
    files = sorted(DOCS.glob("*"))
    assert len(files) == 5
    total, bad, uncovered = _quality(files)
    assert total >= 20 and bad == 0
    # only the deliberately context-dependent scan chunk may be left uncovered
    assert uncovered == 1
    scan = DOCS / "scan_0042.txt"
    assert _make_pairs_from_chunk(scan.read_text(encoding="utf-8"), seen_questions=set(),
                                  filename=scan.name) == []


def test_five_document_corpus_questions_name_something_specific() -> None:
    for f in sorted(DOCS.glob("*")):
        for chunk in _chunks(f.read_text(encoding="utf-8")):
            for q, _a in _make_pairs_from_chunk(chunk, seen_questions=set(), filename=f.name):
                assert _PRONOUN_SUBJECT.search(q) is None, (f.name, q)
                assert not re.search(r"“(?:It|They|This)”", q), (f.name, q)
