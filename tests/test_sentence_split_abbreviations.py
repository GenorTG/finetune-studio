"""Extractive answers must not be cut at 'Dr.' (live walkthrough: 'The keynote was given by Dr.' was approved
training data and the tuned model then answered 'Dr.' and stopped)."""
from __future__ import annotations

from finetune_studio.data.prep.coverage_fill import split_sentences


def test_title_abbreviation_does_not_end_the_sentence() -> None:
    out = split_sentences("Held in Port Selene on 9 March. The keynote was given by Dr. Maren Voss. Doors open at nine.")
    assert "The keynote was given by Dr. Maren Voss." in out
    assert not any(s.endswith("by Dr.") for s in out)


def test_other_abbreviations_and_initials_are_kept_whole() -> None:
    out = split_sentences("The harbour master Mr. Ilsabet Corr signed the order. Report to J. Smith at St. Elm quay today.")
    assert out == ["The harbour master Mr. Ilsabet Corr signed the order.", "Report to J. Smith at St. Elm quay today."]


def test_real_sentence_ends_still_split() -> None:
    out = split_sentences("The warranty lasts three years from purchase. Claims need the serial number printed under the base.")
    assert len(out) == 2
