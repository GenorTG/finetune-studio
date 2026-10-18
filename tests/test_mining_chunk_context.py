"""Neighbouring text around a mining chunk: a record split across chunks is attributed to the right record."""
from __future__ import annotations

from finetune_studio.data.mining import prompts
from finetune_studio.data.mining.job import CONTEXT_BEFORE_CHARS, document_identifiers, neighbour_text
from finetune_studio.data.prep.qa_validate import validate_qa_batch

# The scanned CAR as the Vision reader split it (2026-10-18): chunk 3 opens with measure M5's due date.
CAR = [
    "CAR no. KC-CAR-24-088. ROOT CAUSE (5 x why). Why 1 - release without evaluation? Shift lead signed alone. "
    "Why 3 - why not FSTL?",
    "No on-call number for FSTL, owls could not reach anyone. 4. MEASURES. M4 Train 38 shift leads. Owner I. "
    "Baptista-Okafor. Due 08.10.2024. M5 Look back at all excursions over 4 h since 01.01.2024 - 7 cases found, "
    "2 need late evaluation. Owner Aaltonen.",
    "Due 04.10.2024. 5. SUBMISSION. Plan sent to Aldermark (H. Solvang-Marchetti) 26.09.2024.",
]


def test_neighbour_text_is_the_tail_before_and_the_head_after() -> None:
    before, after = neighbour_text(CAR, 2)
    assert before.endswith("Owner Aaltonen.") and "M5 Look back" in before and after == ""
    before, after = neighbour_text(CAR, 0)
    assert before == "" and after.startswith("No on-call number for FSTL")
    long = ["word " * 200, "x"]
    tail, _ = neighbour_text(long, 1)
    assert len(tail) < CONTEXT_BEFORE_CHARS and not tail.startswith(" ")   # cut at a word boundary


def test_prompts_show_the_context_as_not_part_of_the_passage() -> None:
    block = prompts.context_block(*neighbour_text(CAR, 2))
    for messages in (prompts.inventory_messages("CAR", CAR[2], block),
                     prompts.mine_messages("CAR", CAR[2], ["Due date 04.10.2024"], block),
                     prompts.fix_messages("CAR", CAR[2], ["Due date 04.10.2024"], context=block)):
        user = messages[-1]["content"]
        assert "NOT part of the passage" in user and "Owner Aaltonen." in user
    assert prompts.context_block("", "") == ""
    assert "Neighbouring" not in prompts.mine_messages("CAR", CAR[2], ["x"])[-1]["content"]


def test_an_answer_may_name_the_record_the_context_holds() -> None:
    """Grounding = chunk + neighbours + document ids (as ``_pairs_round``): the M5 answer for chunk 3 is accepted."""
    before, after = neighbour_text(CAR, 2)
    ids = document_identifiers("Corrective Action Request KC-CAR-24-088", CAR[0])
    pair = [{"q": "When is measure M5 of corrective action request KC-CAR-24-088 due?",
             "a": "Measure M5 is due 04.10.2024."}]
    assert validate_qa_batch(pair, f"{CAR[2]}\n{before}\n{after}\n{ids}").accepted
