"""Whole-file corrections reach the chunks that hold only the old value.

Core mine 2026-10-21 (RESULTS §18; open since §1): the whole-file pass listed "the corrected amount is 12,750 EUR ...,
replacing Benedikta's stated 14,500 EUR", but chunk 6 holds only the letter's "Korvane will credit 14,500 EUR" and its pair
stated the superseded figure as the fact."""
from __future__ import annotations

from finetune_studio.data.mining import prompts
from finetune_studio.data.mining.job import correction_notes
from finetune_studio.data.prep.qa_validate import validate_qa_batch

CREDIT = ("Korvane will credit Oakhaven against the June invoice; the corrected amount is 12,750 EUR as cleared by Priya, "
          "replacing Benedikta's stated 14,500 EUR.")
CHUNKS = [
    "Benedikta: I have cleared it with Priya: Korvane will credit 14,500 EUR against your June invoice, no strings.",
    "Jarek: Priya has NOT cleared 14,500. She cleared 12,750 EUR, I have it in writing.",
    "Gunnar: tyre pressures on the three Volvos done.",
]


def _item(text: str, values: list[str], origin: str = "agent_file", **kw) -> dict:
    return {"id": "i", "origin": origin, "text": text, "values": values, "chunk_idx": 2, **kw}


def test_the_chunk_with_only_the_old_value_gets_the_correction() -> None:
    notes = correction_notes([_item(CREDIT, ["12,750 EUR", "14,500 EUR", "June invoice"])], CHUNKS)
    assert notes[1] == [CREDIT] and 3 not in notes      # chunk 3 holds neither value
    both = ["Priya cleared 12,750 EUR, not 14,500 EUR.", CHUNKS[2]]
    assert correction_notes([_item(CREDIT, ["12,750 EUR", "14,500 EUR"])], both) == {}   # the correcting chunk itself


def test_rules_and_other_items_are_not_corrections() -> None:
    rule = ("Overtime worked between 22:00 and 06:00 is paid at 175% in total instead of the 200% Sunday/public-holiday "
            "rate.")
    probation = "The probation notice of 5 working days is replaced, after probation, by 1 month."
    chunks = ["Sunday and public-holiday work is paid at 200%.", "Night work 22:00-06:00 is paid at 175%.",
              "During probation 5 working days."]
    items = [_item(rule, ["22:00", "06:00", "175%", "200%"]), _item(probation, ["5 working days", "1 month"]),
             _item(CREDIT, ["12,750 EUR", "14,500 EUR"], origin="agent"),                   # a chunk item, not whole-file
             _item(CREDIT, ["12,750 EUR", "14,500 EUR"], excluded_reason="not a fact"),
             _item("The corrected bridge: 4.5m, +1.2, +0.9, -0.4, -0.3 gives 5.9m.", ["4.5m", "+1.2", "+0.9", "-0.4", "5.9m"])]
    assert correction_notes(items, chunks) == {}


def test_the_prompt_carries_the_note_and_an_answer_may_name_the_new_value() -> None:
    block = prompts.corrections_block([CREDIT])
    user = prompts.mine_messages("Oakhaven thread", CHUNKS[0], ["credit 14,500 EUR"], block)[-1]["content"]
    assert "Corrected elsewhere in this file" in user and "12,750 EUR" in user and "NOT part of the passage" in user
    assert prompts.corrections_block([]) == ""
    pair = [{"q": "What credit did Benedikta's letter promise Oakhaven against the June invoice?",
             "a": "The letter promised 14,500 EUR, later corrected to 12,750 EUR."}]
    assert not validate_qa_batch(pair, CHUNKS[0]).accepted                       # the chunk alone: 12,750 is invented
    assert validate_qa_batch(pair, f"{CHUNKS[0]}\n{CREDIT}").accepted            # the job's grounding adds the note


def test_the_direction_of_a_correction_decides_who_gets_the_note() -> None:
    """Core mine v4 (2026-10-21): a note on the chunk with the CURRENT 3 months made the helper write that the current rule
    'was later corrected'. Only a chunk with the old value and none of the new gets the note."""
    from finetune_studio.data.mining.job import correction_sides

    notice = "The full notice in 12.2 of 3 months for employees with more than 3 years of service replaces the v3 figure of 2 months."
    assert correction_sides(notice, ["3 months", "3 years", "2 months"]) == (["2 months"], ["3 months", "3 years"])
    assert correction_sides("The cut-off was changed from 13:30 to 15:00.", ["13:30", "15:00"]) == (["13:30"], ["15:00"])
    assert correction_sides("The list of 16 pallets was corrected to 18 pallets.", ["16 pallets", "18 pallets"]) == \
        (["16 pallets"], ["18 pallets"])
    assert correction_sides("Plug C-37 was replaced by C-41.", ["C-37", "C-41"]) == (["C-37"], ["C-41"])
    assert correction_sides("The corrected Q3 EBITDA is EUR 5.9m against EUR 4.5m.", ["EUR 5.9m", "EUR 4.5m"]) is None
    chunks = ["Notice: the Company gives 3 months after more than 3 years of service.",
              "[3] Under v3 the Company's notice for employees with more than 3 years of service was 2 months.",
              "Probation lasts 12 months."]                                     # "2 months" must not match "12 months"
    item = _item(notice, ["3 months", "3 years", "2 months"])
    assert correction_notes([item], chunks) == {}                               # chunk 2 also names "3 years": new
    chunks[1] = "[3] Under v3 the Company's notice was 2 months."
    assert correction_notes([item], chunks) == {2: [notice]}


def test_every_wording_of_the_whole_file_pass_finds_its_direction() -> None:
    """The whole-file pass words the same correction differently each run (core mines v4a / v4b, 2026-10-21)."""
    from finetune_studio.data.mining.job import correction_sides

    cases = [
        ("The credit amount Benedikta first states is superseded; the corrected cleared amount is 12,750 EUR, not 14,500 EUR.",
         ["12,750 EUR", "14,500"], (["14,500"], ["12,750 EUR"])),
        ("The settlement figure of EUR 88,000 in draft v3 was superseded and the current settlement amount is EUR 96,400.00.",
         ["EUR 88,000", "EUR 96,400.00"], (["EUR 88,000"], ["EUR 96,400.00"])),
        ("The cut-off was originally Thursday 16:00, then moved on 2024-09-18 to Thursday 13:30.",
         ["16:00", "13:30"], (["16:00"], ["13:30"])),
        ("The visit was pencilled for Wed 02.10 but Jarek's note moves it to Thu 03.10, superseding the earlier date.",
         ["Wed 02.10", "Thu 03.10"], (["Wed 02.10"], ["Thu 03.10"])),
        ("Korvane will credit Oakhaven; the corrected amount is 12,750 EUR as cleared by Priya, replacing Benedikta's stated "
         "14,500 EUR.", ["12,750 EUR", "14,500 EUR"], (["14,500 EUR"], ["12,750 EUR"])),
    ]
    for text, values, want in cases:
        assert correction_sides(text, values) == want, text
    assert correction_sides("The 338 FTE forecast is not yet reconciled to the 2025 budget.", ["338 FTE", "2025 budget"]) is None


def test_an_invented_or_refining_relation_gets_no_note() -> None:
    """Core mine v4c: the whole-file pass wrote "the audit of 14 to 18 October supersedes the earlier 14 October" and the
    note made the helper call the ISO date 'later corrected'."""
    iso = "The recertification audit is planned for 14 to 18 October 2024, superseding the earlier reference to 14 October 2024."
    chunks = ["The audit team would like to see corrections in place before the recertification audit starting on "
              "14 October 2024.", "Recertification audit: 14 to 18 October 2024, seven audit days."]
    assert correction_notes([_item(iso, ["14 to 18 October 2024", "14 October 2024"])], chunks) == {}   # adds detail only
    invented = "The visit on 02.10 was superseded by the visit on 03.10."
    calm = ["The electrician comes on 02.10 to the depot.", "Fire drill on 03.10 at the depot."]          # nothing changed
    assert correction_notes([_item(invented, ["02.10", "03.10"])], calm) == {}
    wrapped = ["Notice under v4: 3 months.", "[3] Under v3 the Company's notice for employees with more than 3 years of\n"
               "service was 2 months.", "The leave year starts in January."]
    notice = "The notice of 3 months replaces the v3 figure of 2 months."
    assert correction_notes([_item(notice, ["3 months", "2 months"])], wrapped) == {2: [notice]}   # cue across a line wrap


def test_a_value_never_matches_part_of_a_longer_number() -> None:
    """"2 months" was found in "12 months", "500" in "14,500", "38" in "OPP-24-0338": coverage and the recall script counted
    facts as stated by answers about something else (v4d baseline: 2 of 1,130 manifest facts, 2 of 3,184 items)."""
    from finetune_studio.data.mining.match import contains, loosely_contains

    assert not contains("2 months", "Probation lasts 12 months") and contains("2 months", "notice was 2 months.")
    assert not contains("500", "EUR 14,500") and not contains("EUR 4,000", "EUR 4,000,000")
    assert contains("01.10", "due 01.10.2024") and contains("38.50", "EUR 38.5 per day")
    assert contains("45,000 - 89,999", "45,000-89,999") and contains("12,750 EUR", "cleared 12,750 EUR, in writing")
    assert not loosely_contains("38 reefer plugs", "The next step for OPP-24-0338 is reefer plugs")
