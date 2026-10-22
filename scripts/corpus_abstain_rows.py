"""Write "I don't know" training rows for the Korvane corpus (abstention experiment).

    .venv/bin/python scripts/corpus_abstain_rows.py [--n 240] [--seed 7] [--with-golden] --out train_abstain.jsonl

A fact-tuned adapter answers every question with a confident value, including ones its documents never cover (RESULTS §3:
0/20 declined). These rows teach the other answer: questions the corpus cannot answer, each answered with a short decline.
Every question is unanswerable by construction, never by a model's guess (RESULTS §4: 112 of 150 helper-written "unknown"
questions were answerable from other files):

- a REAL person or company with an attribute no corpus file mentions (each attribute keyword is checked against every
  ``src/*.src`` file, so the whole 80-file corpus, not only the trained core tier);
- a golden question with its named entity swapped for a FICTITIOUS one (a name, site code or product absent from the corpus).

Attribute types of the held-out ``eval/unanswerable_core.jsonl`` (salary, revenue, phone, IP address, ...) are never used, so the
quiz measures whether declining generalises, not whether a template was memorised. With ``--with-golden`` the golden pairs
(``--golden``, default the core-tier set) are written too, shuffled together with the decline rows.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
from pathlib import Path

KORVANE = Path(__file__).resolve().parent.parent / "tests" / "corpus" / "korvane"

PEOPLE = ("Gunnar Halvorsrud", "Benedikta Aaltonen", "Olek Maraszek", "Priya Venkataraman-Holt", "Dieter Ahlbeck-Roe",
          "Ines Baptista-Okafor", "Lotte Sandberg", "Kamil Zdunek", "Tomasz Wrzesiński", "Wiktor Szczepaniak",
          "Jarek Lindqvist", "Marta Kowarzik", "Hanneke Dijkgraaf", "Marcin Olszewski", "Wojciech Lenartowicz",
          "Kinga Sokołowska-Reid", "Aneta Gorczyca-Reyes", "Sanna Virtanen", "Gwendolen Pryce-Aldous", "Joris Kuipers-Bax",
          "Femke Oosterhuis", "Bart Wiegersma", "Silje Haugen", "Tuomas Rautio", "Radosław Pietrzak", "Halina Dziurdziak")
COMPANIES = ("Fruttoria Baltica", "Voltrac Battery Systems", "Lintu Cloud Hosting", "Arendal Port Services",
             "Seaboard Insurance Brokers", "Mecklenburger Beeren", "Bastion Ochrona", "Stavanger Skalldyr AS",
             "Polar Chain Eesti", "Lietuvos Pienas UAB", "Brandenburg Eis GmbH", "Haldane Rooke Partners")

# (question template, corpus keywords that must be absent). Nothing here overlaps the held-out unanswerable set.
PERSON_ATTRS = (
    ("What is {e}'s date of birth?", ("date of birth", "born on")),
    ("What is {e}'s home address?", ("home address",)),
    ("What hobby does {e} have?", ("hobby", "hobbies")),
    ("What is {e}'s star sign?", ("star sign", "zodiac")),
    ("Which university did {e} study at?", ("university",)),
    ("What is {e}'s middle name?", ("middle name",)),
    ("What is {e}'s personal best in the marathon?", ("marathon",)),
    ("What is the name of {e}'s pet?", ("pet's", "pet ")),
    ("Where did {e} go on holiday last summer?", ("holiday destination", "last summer")),
    ("What is the licence plate of {e}'s private car?", ("licence plate", "number plate", "registration plate")),
    ("How does {e} commute to work?", ("commute",)),
    ("What is {e}'s Instagram handle?", ("instagram",)),
    ("What is {e}'s chess rating?", ("chess",)),
)
COMPANY_ATTRS = (
    ("What is the stock exchange ticker of {e}?", ("stock exchange", "share price")),
    ("What is {e}'s IBAN?", ("iban",)),
    ("What is {e}'s company slogan?", ("slogan",)),
    ("What is the name of {e}'s mascot?", ("mascot",)),
    ("What is {e}'s fax number?", ("fax",)),
    ("Which charity does {e} support?", ("charity",)),
    ("What is {e}'s Instagram account?", ("instagram",)),
    ("What is the name of {e}'s company podcast?", ("podcast",)),
)
FAKE_PEOPLE = ("Halvard Ringstad", "Ottilie Brzezicka", "Marek Vandersloot", "Ylva Kettunen-Shaw", "Dorota Abernethy",
               "Sven Mikołajczyk", "Rosalind Ekström", "Piet Wawrzyniak", "Ada Lindqvist-Moreau", "Konrad Haverkamp")
FAKE_COMPANIES = ("Nordlys Frost AS", "Wislana Owoce Sp. z o.o.", "Elbe Kühlhaus GmbH", "Saimaa Fresh Oy",
                  "Kattegat Reefer Lines", "Pomorska Mleczarnia SA")
FAKE_CODES = ("OSL-4", "CPH-6", "TLL-2", "RIX-8", "MMX-3", "BRE-7")
FAKE_PRODUCTS = ("KC-Tag 9", "KC-Hub 900", "KC-Sense 5", "KC-Gate 12")

ANSWERS = (
    "I don't know. That isn't covered in the material I was trained on.",
    "I don't have that information; the Korvane documents I learned from don't mention it.",
    "That isn't in my training material, so I don't know. Please check the source documents.",
    "I don't know. None of the documents I was trained on say this.",
    "I can't answer that: it is not stated in the Korvane documents I know.",
)


def corpus_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="replace").lower() for p in sorted((KORVANE / "src").rglob("*.src")))


def real_entity_rows(text: str) -> list[str]:
    out = []
    for names, attrs in ((PEOPLE, PERSON_ATTRS), (COMPANIES, COMPANY_ATTRS)):
        for tmpl, keys in attrs:
            if any(k in text for k in keys):
                continue  # the corpus mentions this attribute somewhere: not unanswerable by construction
            out += [tmpl.format(e=n) for n in names if n.lower() in text]
    return out


def swapped_rows(text: str, golden: list[dict]) -> list[str]:
    fakes = [f for f in FAKE_PEOPLE + FAKE_COMPANIES + FAKE_CODES + FAKE_PRODUCTS if f.lower() not in text]
    real = sorted({*PEOPLE, *COMPANIES, *re.findall(r"\b[A-Z]{3}-\d\b", " ".join(r["q"] for r in golden)),
                   *re.findall(r"\bKC-[A-Z][a-z]+ \d+\b", " ".join(r["q"] for r in golden))}, key=len, reverse=True)
    out = []
    for row in golden:
        for ent in real:
            if ent in row["q"]:
                kind = (FAKE_CODES if re.fullmatch(r"[A-Z]{3}-\d", ent) else FAKE_PRODUCTS if ent.startswith("KC-")
                        else FAKE_COMPANIES if ent in COMPANIES else FAKE_PEOPLE)
                pool = [f for f in kind if f in fakes]
                if pool:
                    out.append(row["q"].replace(ent, pool[len(out) % len(pool)]))
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=240, help="decline rows to write (half real-entity, half swapped)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--golden", default=str(KORVANE / "golden" / "approved_pairs_core_2026-10-08.jsonl.gz"))
    ap.add_argument("--with-golden", action="store_true", help="also write the golden pairs (the full training set)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    text = corpus_text()
    with gzip.open(args.golden, "rt", encoding="utf-8") as fh:
        golden = [json.loads(line) for line in fh if line.strip()]
    held_out = {json.loads(line)["q"].strip().lower() for line in (KORVANE / "eval" / "unanswerable_core.jsonl").open()}

    real, swapped = real_entity_rows(text), swapped_rows(text, golden)
    rng.shuffle(real)
    rng.shuffle(swapped)
    half = args.n // 2
    qs = list(dict.fromkeys(real[:half] + swapped[: args.n - min(half, len(real))]))
    qs = [q for q in qs if q.strip().lower() not in held_out]
    rows = [{"q": q, "a": rng.choice(ANSWERS), "origin": "abstain"} for q in qs]
    if args.with_golden:
        rows += [{"q": r["q"], "a": r["a"], "origin": "golden"} for r in golden]
    rng.shuffle(rows)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"real-entity pool {len(real)}, swapped pool {len(swapped)}; wrote {len(rows)} rows "
          f"({len(qs)} decline) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
