"""IFEval checkers (benchmarks/ifeval.py) and the ifeval real suite (mocked HF split, no download)."""
from __future__ import annotations

from typing import Any

import pytest

from finetune_studio.benchmarks.ifeval import (
    CHECKERS,
    check_response,
    encode_answer,
    is_supported,
)
from finetune_studio.benchmarks.real_benchmarks import RealBenchmarkSuite
from finetune_studio.testing.strict_scoring import score_strict


def _ok(iid: str, response: str, **kw: Any) -> bool:
    return CHECKERS[iid](response, kw)


@pytest.mark.parametrize(
    ("iid", "kw", "good", "bad"),
    [
        ("punctuation:no_comma", {}, "no commas here", "one, two"),
        ("length_constraints:number_words", {"relation": "at least", "num_words": 3}, "one two three", "one two"),
        ("length_constraints:number_words", {"relation": "less than", "num_words": 3}, "one two", "one two three"),
        ("keywords:forbidden_words", {"forbidden_words": ["cat"]}, "a concatenation", "the Cat sat"),
        ("detectable_format:number_highlighted_sections", {"num_highlights": 2}, "*a* and **b**", "*a* only"),
        ("keywords:frequency", {"keyword": "ok", "frequency": 2, "relation": "at least"}, "ok OK", "ok"),
        ("combination:repeat_prompt", {"prompt_to_repeat": "Say hi."}, "say hi. Hi!", "Hi! Say hi."),
        ("startend:quotation", {}, '"quoted"', 'not "quoted"'),
        ("change_case:english_lowercase", {}, "all lower", "Not lower"),
        ("change_case:english_capital", {}, "ALL CAPS", "ALL CAPs"),
        ("keywords:existence", {"keywords": ["alpha", "beta"]}, "Alpha and beta", "alpha only"),
        ("detectable_format:title", {}, "<<My Title>>\ntext", "<< >> no title"),
        ("keywords:letter_frequency", {"letter": "z", "let_frequency": 2, "let_relation": "at least"}, "Zz", "z"),
        ("detectable_format:number_bullet_lists", {"num_bullets": 2}, "* a\n- b", "* a\n* b\n* c"),
        ("detectable_content:number_placeholders", {"num_placeholders": 2}, "[name] at [address]", "[name]"),
        ("length_constraints:number_paragraphs", {"num_paragraphs": 2}, "p1\n***\np2", "p1\n***\n***\np2"),
        ("startend:end_checker", {"end_phrase": "Any questions?"}, "Text. Any questions?", "Any questions? Text."),
        ("detectable_content:postscript", {"postscript_marker": "P.S."}, "Body\nP.S. more", "Body only"),
        ("change_case:capital_word_frequency", {"capital_frequency": 2, "capital_relation": "at least"},
         "NASA and ESA", "NASA only"),
        ("combination:two_responses", {}, "first\n******\nsecond", "same\n******\nsame"),
        ("detectable_format:json_format", {}, '```json\n{"a": 1}\n```', "{a: 1}"),
        ("detectable_format:multiple_sections", {"section_spliter": "Section", "num_sections": 2},
         "Section 1\nx\nSection 2\ny", "Section 1\nx"),
        ("length_constraints:nth_paragraph_first_word",
         {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "then"}, "First.\n\nThen, more.", "First.\n\nNow."),
        ("detectable_format:constrained_response", {}, "My answer is yes.", "Yes."),
    ],
)
def test_checker_pass_and_fail(iid: str, kw: dict, good: str, bad: str) -> None:
    assert _ok(iid, good, **kw), good
    assert not _ok(iid, bad, **kw), bad


def test_unsupported_instructions_are_left_out_not_approximated() -> None:
    assert is_supported(["punctuation:no_comma"])
    assert not is_supported(["punctuation:no_comma", "language:response_language"])
    assert not is_supported(["length_constraints:number_sentences"])
    assert not is_supported([])


def test_prompt_passes_only_when_every_instruction_passes() -> None:
    key = encode_answer(["punctuation:no_comma", "change_case:english_lowercase"], [{"x": None}, {}])
    assert check_response(key, "all fine here") == (True, [])
    assert check_response(key, "All, wrong") == (False, ["punctuation:no_comma", "change_case:english_lowercase"])
    s = score_strict(question="Write ...", correct_answer=key, model_answer="nope, comma")
    assert s is not None and s.verdict == "fail" and s.scoring_method == "ifeval_strict"
    assert score_strict(question="Write ...", correct_answer=key, model_answer="  ").validity == "no_answer"


def _row(key: int, ids: list[str], kwargs: list[dict], prompt: str) -> dict:
    return {"key": key, "prompt": prompt, "instruction_id_list": ids, "kwargs": kwargs}


def test_ifeval_suite_samples_only_scorable_prompts_and_scores_them() -> None:
    split = [
        _row(1, ["language:response_language"], [{"language": "de"}], "Antworte auf Deutsch."),
        _row(2, ["punctuation:no_comma"], [{}], "Describe tea without commas."),
        _row(3, ["change_case:english_capital"], [{}], "Shout a greeting."),
    ]

    def loader(path, name=None, *, split: str, cache_dir=None, revision=None):
        assert (path, name, split) == ("google/IFEval", None, "train")
        return split_rows

    split_rows = split
    suite = RealBenchmarkSuite(dataset_loader=loader)
    cases, meta = suite.load_cases("ifeval", num_samples=10)
    assert [c.name for c in cases] == ["ifeval_00002", "ifeval_00003"]
    assert meta.loaded_split_size == 2

    class Engine:
        def __init__(self) -> None:
            self.max_tokens: list[int] = []

        def generate(self, messages, *, max_tokens=512, temperature=0.0, **_):
            self.max_tokens.append(max_tokens)
            return {"Describe tea without commas.": "Tea is warm and calm", "Shout a greeting.": "Hello there"}[
                messages[0]["content"]]

    engine = Engine()
    out = suite.evaluate("ifeval", engine, num_samples=10)
    assert (out.total, out.correct) == (2, 1)
    assert engine.max_tokens == [1024, 1024]
    assert "english_capital" in out.results[1].reasoning
