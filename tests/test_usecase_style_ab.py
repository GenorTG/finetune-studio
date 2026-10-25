"""Style template: blind A/B in both orders; a win needs both orders; position bias becomes a tie; sign test."""
from __future__ import annotations

import json

from finetune_studio.usecases import style_ab as ab


def _case(i: int = 0) -> ab.ABCase:
    return ab.ABCase(key=f"k{i}", question=f"Q{i}", reference="Short. Warm.", base_answer=f"AAA {i}",
                     tuned_answer=f"ZZZ {i}")


def _judge(pick: str):
    """A judge that picks the answer containing ``pick`` (it only sees ANSWER 1 / ANSWER 2)."""
    seen: list[str] = []

    def chat(messages: list[dict]) -> str:
        prompt = messages[1]["content"]
        seen.append(prompt)
        first = prompt.split("ANSWER 1:\n", 1)[1].split("\n\nANSWER 2:", 1)[0]
        winner = "1" if pick in first else "2"
        return json.dumps({"reasoning": "closer", "winner": winner})

    return chat, seen


def test_the_judge_is_blind_and_asked_in_both_orders() -> None:
    chat, seen = _judge("ZZZ")
    r = ab.judge_case(chat, _case(), style_guide="Be brief.")
    assert r.outcome == "tuned" and (r.first, r.second) == ("2", "1")
    assert len(seen) == 2 and not any(w in p.lower() for p in seen for w in ("base", "tuned", "trained", "untrained"))
    assert seen[0].index("AAA 0") < seen[0].index("ZZZ 0") and seen[1].index("ZZZ 0") < seen[1].index("AAA 0")
    assert "STYLE GUIDE:\nBe brief." in seen[0] and "REFERENCE (house style):\nShort. Warm." in seen[0]


def test_a_judge_that_always_picks_answer_1_produces_ties_not_wins() -> None:
    r = ab.judge_case(lambda _m: '{"winner": "1", "reasoning": ""}', _case())
    assert r.outcome == "tie"


def test_base_wins_only_when_both_orders_agree_and_a_tie_reply_is_a_tie() -> None:
    chat, _ = _judge("AAA")
    assert ab.judge_case(chat, _case()).outcome == "base"
    assert ab.judge_case(lambda _m: '{"winner": "tie"}', _case()).outcome == "tie"


def test_failed_or_unreadable_judge_calls_are_errors_never_outcomes() -> None:
    def boom(_m):
        raise RuntimeError("HTTP 500")

    r = ab.judge_case(boom, _case())
    assert r.outcome == "" and "HTTP 500" in r.error
    calls: list[int] = []

    def mumble(_m):
        calls.append(1)
        return "I like both."

    r = ab.judge_case(mumble, _case())
    assert r.outcome == "" and r.error == "the judge reply named no winner" and len(calls) == 2  # one reminder


def test_reply_parsing_tolerates_think_blocks_and_loose_text() -> None:
    assert ab.parse_reply('<think>{"winner": "1"}</think>{"reasoning": "x", "winner": "2"}') == ("2", "x")
    assert ab.parse_reply('verdict -> "winner": "tie" (both fine)') == ("tie", "")
    assert ab.parse_reply("no idea") == ("", "")


def test_summary_is_a_sign_test_on_wins_and_losses() -> None:
    chat, _ = _judge("ZZZ")
    results = ab.run_ab(chat, [_case(i) for i in range(10)], concurrent=True)
    summary = ab.summarize(results)
    assert (summary.wins, summary.losses, summary.ties, summary.errors) == (10, 0, 0, 0)
    assert summary.p_value < 0.01 and summary.verdict == "trained preferred" and summary.win_rate == 1.0
    mixed = [ab.ABResult("a", "tuned"), ab.ABResult("b", "base"), ab.ABResult("c", "tie"), ab.ABResult("d", "")]
    s = ab.summarize(mixed)
    assert (s.wins, s.losses, s.ties, s.errors, s.verdict) == (1, 1, 1, 1, "could be noise")
    assert s.win_rate == 1 / 3 and ab.summarize([]).verdict == "nothing judged"


def test_pairs_are_the_questions_both_models_answered() -> None:
    base = [{"case_name": "c1", "question": "Q1", "correct_answer": "R1", "model_answer": "<think>x</think>b1"},
            {"case_name": "c2", "question": "Q2", "correct_answer": "R2", "model_answer": ""}]
    tuned = [{"case_name": "c1", "question": "Q1", "correct_answer": "R1", "model_answer": "t1"},
             {"case_name": "c2", "question": "Q2", "correct_answer": "R2", "model_answer": "t2"},
             {"case_name": "c3", "question": "Q3", "correct_answer": "R3", "model_answer": "t3"}]
    pairs = ab.pair_cases(base, tuned)
    assert [(p.question, p.reference, p.base_answer, p.tuned_answer) for p in pairs] == [("Q1", "R1", "b1", "t1")]
