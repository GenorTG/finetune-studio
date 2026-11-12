"""Missing-part search: an answer that says part of the question is not stated gets one more search for that part and
one more answer (Tarrowfield in-008: the £211,450 passage was rank 9 of 10 and the model said "not stated").
The package's /ask, the studio chat, the CLI and Testing's grounded suite all run this rule."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from finetune_studio.smart_brain import standalone_server as srv


def _hit(cid: str, text: str, rank: int = 1, filename: str = "") -> dict:
    return {"rank": rank, "chunk_id": cid, "filename": filename or f"{cid}.pdf", "text": text, "rrf_score": 0.03}


LEASE = [_hit("sig", "Signed for the customer by Edwina Rook, Director.", filename="lease.pdf")]
VALUE = _hit("val", "Total contract value: £211,450 over 36 months.", filename="lease.pdf")


@pytest.mark.parametrize(("answer", "says"), [
    ("Edwina Rook signed it [lease.pdf]. The value of the lease is not stated in the context.", True),
    ("The documents do not mention who signed it.", True),
    ("The context doesn't specify the governing law.", True),
    ("I could not find the answer in the documents.", True),
    ("There is no information about the multiple in the passages.", True),
    ("Edwina Rook signed it; it is worth £211,450 [lease.pdf].", False),
    ("The notice period is not less than 30 days.", False),
])
def test_says_not_stated(answer: str, says: bool) -> None:
    assert srv.says_not_stated(answer) is says


def test_the_missing_part_is_searched_and_the_answer_written_again() -> None:
    replies = iter(["Edwina Rook signed it [lease.pdf]; the value is not stated.",       # first answer
                    "Pelling & Rook lease total contract value",                           # the sub-query
                    "£211,450 over 36 months, signed by Edwina Rook [lease.pdf]."])      # the answer again
    chat = MagicMock(side_effect=lambda msgs: next(replies))
    search = MagicMock(return_value=[VALUE, LEASE[0]])
    out = srv.answer_with_follow_ups("What's the value of the Pelling & Rook lease and who signed it?", LEASE,
                                     search, chat)
    assert out["answer"].startswith("£211,450")
    search.assert_called_once_with("Pelling & Rook lease total contract value")
    assert out["followup"]["query"] == "Pelling & Rook lease total contract value"
    assert out["followup"]["added_sources"] == ["lease.pdf"] and out["followup"]["added_passages"] == 1
    assert out["followup"]["answer_before"].endswith("the value is not stated.")
    assert [h["chunk_id"] for h in out["hits"]] == ["sig", "val"]
    assert out["follow_up_searches"] == ["Pelling & Rook lease total contract value"]
    assert chat.call_count == 3
    last_system = chat.call_args_list[-1].args[0][0]["content"]
    assert "£211,450" in last_system and srv.FOLLOW_UP_INSTRUCTION not in last_system   # no more searches offered


def test_one_retry_only_and_nothing_new_means_no_second_answer() -> None:
    replies = iter(["The value is not stated.", "lease value"])
    chat = MagicMock(side_effect=lambda msgs: next(replies))
    out = srv.answer_with_follow_ups("What is the lease worth?", LEASE, MagicMock(return_value=list(LEASE)), chat)
    assert out["answer"] == "The value is not stated." and chat.call_count == 2
    assert out["followup"]["added_passages"] == 0


def test_off_switches_and_a_full_answer_cost_nothing() -> None:
    for kwargs in ({"missing_part_retry": False}, {"follow_ups": 0}):
        chat = MagicMock(return_value="The value is not stated.")
        search = MagicMock()
        out = srv.answer_with_follow_ups("What is the lease worth?", LEASE, search, chat, **kwargs)
        assert out["followup"] is None and chat.call_count == 1
        search.assert_not_called()
    chat = MagicMock(return_value="£211,450 [lease.pdf].")
    assert srv.answer_with_follow_ups("Worth?", [VALUE], MagicMock(), chat)["followup"] is None


def test_without_a_usable_sub_query_the_least_covered_clause_is_searched() -> None:
    replies = iter(["Liffeyside's contract is worth €318,000 [a.pdf]; the governing law is not stated.",
                    "I'm sorry, I can't help with writing searches because the instructions are unclear to me today."
                    " " * 200])
    chat = MagicMock(side_effect=lambda msgs: next(replies, "Irish law, Dublin courts [b.pdf]."))
    search = MagicMock(return_value=[_hit("law", "governed by the laws of Ireland; courts of Dublin")])
    srv.answer_with_follow_ups("Which law governs the Liffeyside contract and what's it worth?",
                               [_hit("v", "€318,000")], search, chat)
    assert search.call_args.args[0] == "Which law governs the Liffeyside contract"


def test_the_package_reports_followup_and_the_switch_is_a_config_key() -> None:
    cfg, _ = srv.resolve_config({}, {"RAG_RETRY_MISSING": "0"}, {})
    assert cfg["retry_missing"] is False
    assert srv.resolve_config({}, {}, {})[0]["retry_missing"] is True


def test_the_testing_suite_retries_and_records_it() -> None:
    from finetune_studio.testing import brain_suite
    from finetune_studio.testing.suite import BenchmarkCase

    class Engine:
        def __init__(self) -> None:
            self.replies = iter(["The value is not stated.", "lease value", "£211,450 [lease.pdf]."])

        def generate(self, messages, **_kw) -> str:
            return next(self.replies)

    class Search:
        def search(self, q, top_k=10):
            return [dict(VALUE)] if q == "lease value" else [dict(LEASE[0])]

        def format_context(self, hits, max_chars=4000):
            return "\n".join(f"[{h['rank']}] {h['text']}" for h in hits)

    case = BenchmarkCase(name="c", category="lease", question="What is the lease worth?", correct_answer="£211,450")
    res = brain_suite.run_brain_case(Engine(), Search(), case)
    assert res.case_result.model_answer == "£211,450 [lease.pdf]."
    assert res.followup["query"] == "lease value" and res.as_api_dict()["followup"]["added_sources"] == ["lease.pdf"]
    assert any("searched again" in str(m.get("content")) for m in res.case_result.transcript)
