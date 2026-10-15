"""Prompts of the per-file mining stages. ``PROMPT_VERSION`` is stored on every job so runs stay comparable."""
from __future__ import annotations

from collections.abc import Sequence

PROMPT_VERSION = "mining-v2"

INVENTORY_SYSTEM = """You list the data points of a company document so that nothing in it is forgotten when it is turned into \
training questions.

Output ONLY a JSON array of {"fact": "...", "values": ["...", "..."]} objects. No markdown fences, no commentary. Start with [.

Rules:
- One entry per distinct fact: every number, amount, date, time, threshold, name, code, identifier, contact, deadline, rule, \
condition, exception, decision, responsibility, cause and every table row or list item. Do not summarise; do not skip details \
that look minor.
- "fact" is one short sentence that states the fact with its subject (who/what it is about).
- "values" are the pieces a correct answer MUST contain, copied EXACTLY as they are written in the passage (same digits, units, \
spelling): numbers, dates, codes, names, and for a rule stated in words its key words (e.g. "not suspended pending appeal"). \
Use the shortest exact piece; never paraphrase a value and never invent one.
- "values" are data, not sentences: never a whole clause or a quote of a chat message.
- Skip pure boilerplate (page numbers, "confidential" banners) and the document's structure (section titles, headings, what \
the next section is): they state no fact."""

INVENTORY_USER = """Document: {title}

Passage:
\"\"\"
{chunk}
\"\"\"

List EVERY data point of this passage. Respond with ONLY the JSON array."""

FILE_INVENTORY_USER = """Document: {title}

Whole document:
\"\"\"
{text}
\"\"\"

List ONLY the facts that need more than one part of this document to be stated correctly: a value that a later part \
corrects or supersedes (state which value is current and that it replaced the earlier one), a decision that changes an \
earlier plan, a total or comparison across sections, a person or code defined in one place and used in another. Do not list \
facts that one passage states on its own. Respond with ONLY the JSON array (an empty array [] when there are none)."""

MINE_SYSTEM = """You turn company documents into question-and-answer training pairs. The goal is COMPLETENESS: a model trained on \
your pairs must be able to answer any question about this passage.

Output ONLY a JSON array of {"q": "...", "a": "..."} objects. No markdown fences, no commentary, no thinking blocks. Start with [.

Rules:
- ONE fact per pair. A table row may be one pair (ask about the row by its key, answer with the column names and values); a \
sentence with several facts becomes several pairs.
- Every data point in the checklist must be stated by at least one pair; also cover any other fact of the passage.
- Every question must stand alone: name the subject (document, system, policy, product, person, customer, row key) instead of \
"it", "this" or "the passage", and use the document name to make questions specific.
- Every answer is a complete sentence that states the exact values as written in the passage (keep numbers, units, codes and \
names verbatim). Never add information that is not in the passage.
- When the passage corrects or supersedes a value, the answer gives the current value and says it replaced the earlier one.
- For chat logs, e-mails and minutes: state the fact itself (what was decided, measured, promised, when, by whom). Never write \
"What did <person> say at <time>?" questions or answers that just quote a message.
- Never ask the same fact twice in different words.
- A "[Figure N · ... · read by OCR, not checked]" block is machine-read text of a picture (a scan, chart or table image): \
its words and numbers are usually right but their ORDER is not, so never pair a number with a label (a year, a row, a bar) \
unless the text itself states them together; state only what it says unambiguously (a title, a sentence, a stated pair). A \
block "read by <model>" is a vision model's reading of the picture and can be used like a table."""

MINE_USER = """Document: {title}

Passage:
\"\"\"
{chunk}
\"\"\"

Checklist of data points in this passage (each must be stated by at least one pair):
{checklist}

Write the question-answer pairs. Respond with ONLY the JSON array."""

FIX_USER = """Document: {title}

Passage:
\"\"\"
{chunk}
\"\"\"

These data points of the passage are NOT stated by any accepted pair yet:
{checklist}
{rejected}
Write question-answer pairs that state each of them (one fact per pair, questions that name their subject, answers with the \
exact values as written). A data point that combines several values may be stated across several pairs.

If a data point is NOT something a trained model should know (a heading, a page or clause number on its own, a pleasantry, \
gossip, a remark about the document itself), do not write a pair for it: add {{"skip": <its number>, "reason": "<why, a few \
words>"}} to the same array instead. A reviewer confirms every skip, so skip only what is clearly not a fact. Never skip a \
data point because you think it is already covered, duplicated or combines other values: you cannot see the other pairs, \
so write the pair. Respond with ONLY the JSON array."""

REJECTED_BLOCK = """
A reviewer rejected these pairs for this passage. Do not repeat their mistakes:
{rows}
"""


def numbered(lines: Sequence[str]) -> str:
    return "\n".join(f"{i}. {line}" for i, line in enumerate(lines, 1))


def inventory_messages(title: str, chunk: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": INVENTORY_SYSTEM},
            {"role": "user", "content": INVENTORY_USER.format(title=title, chunk=chunk[:8000])}]


def file_inventory_messages(title: str, text: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": INVENTORY_SYSTEM},
            {"role": "user", "content": FILE_INVENTORY_USER.format(title=title, text=text)}]


def mine_messages(title: str, chunk: str, checklist: Sequence[str]) -> list[dict[str, str]]:
    return [{"role": "system", "content": MINE_SYSTEM},
            {"role": "user", "content": MINE_USER.format(title=title, chunk=chunk[:8000], checklist=numbered(checklist))}]


def fix_messages(title: str, chunk: str, checklist: Sequence[str],
                 rejected: Sequence[tuple[str, str, str]] = ()) -> list[dict[str, str]]:
    block = REJECTED_BLOCK.format(rows="\n".join(f"- Q: {q}\n  A: {a}\n  Reason: {why or 'rejected'}" for q, a, why in rejected[:20])) \
        if rejected else ""
    return [{"role": "system", "content": MINE_SYSTEM},
            {"role": "user", "content": FIX_USER.format(title=title, chunk=chunk[:8000], checklist=numbered(checklist),
                                                        rejected=block)}]
