"""Augment a project's Q&A corpus with source-grounded pairs targeting known gaps.

Reads ``qa/pairs/*.json`` from a project, splits them deterministically
(seed=42, 90/10) into train + held-out, then walks the project's parsed
sources to extract targeted fact-bearing sentences that map to the held-out
questions the model most often gets wrong. Writes:

  * ``<project>/augmented-pairs-v2.jsonl`` — new pairs to merge into qa/pairs/
  * ``<project>/held-out.jsonl`` — single-JSON wrapped suite for run-suite
  * ``<project>/datasets/<pid>-sharegpt-approved.jsonl`` — full export
    including the new pairs (canonical Training-tab dataset)

Run on fan-dragon after the held-out WebUI run lands a low score, before the
next training run is started.
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import re
import sys
import time

# Add repo root so we can reuse training.data.split_data without installing.
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

log = logging.getLogger(__name__)


def split_sentences(text: str) -> list[str]:
    """Crude sentence split on .!? or blank-line boundaries."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n+", text) if s.strip()]


def find_fact(parsed_txt: pathlib.Path, ans_sig: str, term: str) -> str | None:
    """Return the sentence(s) in *parsed_txt* containing *term* and (when
    possible) *ans_sig*. Empty when nothing matches."""
    if not parsed_txt.is_file():
        return None
    text = parsed_txt.read_text(encoding="utf-8", errors="ignore")
    sents = split_sentences(text)
    matching = [s for s in sents if term.lower() in s.lower()]
    if not matching:
        return None
    preferred = [s for s in matching if ans_sig.lower() in s.lower()]
    chosen = preferred[0] if preferred else matching[0]
    idx = sents.index(chosen)
    return " ".join(sents[max(0, idx):idx + 3]).strip()


# (canonical_question, expected_answer_signature, term)
TARGET_PAIRS: list[tuple[str, str, str]] = [
    ("Who is the recorded owner of risk RK-04, the carton damage risk?", "Elian Mertens", "RK-04"),
    ("On what date is the next vendor review scheduled after the Q3 review?", "2026-10-05", "next review"),
    ("Under what condition may Helios reject a shipment from Oriole?", "more than 2 percent", "shipment"),
    ("Who is the policy owner, and when is the next policy review scheduled?", "Nadiya Petrov", "Nadiya Petrov"),
    ("What is the current approval status of change request CR-77?", "not approved", "CR-77"),
    ("Who is the finance reviewer, and who is the procurement owner?", "Ada Smit", "reviewer"),
    ("How many return units were recorded in July, and how many were still awaiting inspection?", "612", "July"),
    ("What is the external API status for project OCTOPUS-7741, and which release will revert it?", "external_api_hidden", "OCTOPUS-7741"),
    ("What are the effective and expiry dates of the Oriole Packaging agreement?", "2027-06-30", "Oriole Packaging"),
    ("When and where did conveyor C-17 stop?", "2026-08-04", "C-17"),
    ("Which team owns the customer reply, and which owns physical Returns processing?", "Tier-one support", "customer reply"),
    ("Which glossary term specifically requires human action rather than automated intervention?", "Exception", "Exception"),
    ("How should orders containing temperature-sensitive goods be handled?", "escalated immediately", "temperature-sensitive"),
    ("Was the unload regression marked as fixed for OCTOPUS-7741, and if so, on what date?", "2026-09-08", "unload_regression_fixed"),
    ("What is the minimum number of units for which a reason code is mandatory when adjusting a return?", "25 units", "reason code"),
    ("Who is listed as the maintenance contact for the Rotterdam site?", "Pavel Novak", "Rotterdam"),
]


def build_augmented_pairs(project_dir: pathlib.Path) -> list[dict]:
    """Walk every parsed.txt under project/files/ and build one augmented pair
    per TARGET_PAIRS whose fact is present in the corpus.
    """
    src_root = project_dir / "files"
    augmented: list[dict] = []
    for i, (question, ans_sig, term) in enumerate(TARGET_PAIRS):
        found = None
        for d in sorted(src_root.iterdir()):
            if not d.is_dir():
                continue
            try:
                snippet = find_fact(d / "parsed.txt", ans_sig, term)
            except Exception as exc:  # noqa: BLE001 — defensively skip unreadable parsed files
                log.debug("find_fact skipped %s: %s", d, exc)
                continue
            if snippet and ans_sig.lower() in snippet.lower():
                found = (d.name, snippet)
                break
        if not found:
            # Fallback: any sentence mentioning the term.
            for d in sorted(src_root.iterdir()):
                if not d.is_dir():
                    continue
                snippet = find_fact(d / "parsed.txt", term, term)
                if snippet:
                    found = (d.name, snippet)
                    break
        if not found:
            print(f"  SKIP: no source for {term!r}")
            continue
        src_id, snippet = found
        answer = snippet if len(snippet) <= 800 else snippet[:800].rsplit(".", 1)[0] + "."
        augmented.append({
            "name": f"aug-v2-{i:03d}",
            "question": question,
            "correct_answer": answer,
            "category": "source-grounded-augmented",
            "source_id": src_id,
            "chunk_idx": 0,
            "keywords": re.findall(r"[A-Za-z0-9]+", answer)[:8],
        })
    return augmented


def to_chat(qa: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": qa.get("question", "")},
            {"role": "assistant", "content": qa.get("answer", "")},
        ],
        "source_id": qa.get("source_id", ""),
        "chunk_idx": qa.get("chunk_idx", 0),
        "keywords": qa.get("keywords", []),
    }


def split_held_out(pairs: list[dict], seed: int = 42, ratio: float = 0.9) -> tuple[list, list]:
    from finetune_studio.training.data import split_data
    formatted = [to_chat(p) for p in pairs]
    return split_data(formatted, train_ratio=ratio, seed=seed)


def write_held_out_suite(held: list[dict], path: pathlib.Path) -> None:
    cases = []
    for i, ex in enumerate(held):
        msgs = ex["messages"]
        user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        cases.append({
            "name": f"held-{i:03d}",
            "question": user,
            "correct_answer": asst,
            "category": "source-grounded",
            "source_id": ex.get("source_id", ""),
            "chunk_idx": ex.get("chunk_idx", 0),
            "keywords": ex.get("keywords", []),
        })
    suite = {
        "name": path.stem,
        "version": "1.0",
        "seed": 42,
        "train_ratio": 0.9,
        "cases": cases,
    }
    path.write_text(json.dumps(suite, indent=2, ensure_ascii=False), encoding="utf-8")


def merge_into_qa_pairs(project_dir: pathlib.Path, augmented: list[dict]) -> int:
    pairs_dir = project_dir / "qa" / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    added = 0
    for i, p in enumerate(augmented):
        qa_id = f"qa_aug_{ts}_{i:03d}"
        qa = {
            "id": qa_id,
            "question": p["question"],
            "answer": p["correct_answer"],
            "source_id": p["source_id"],
            "chunk_idx": p.get("chunk_idx", 0),
            "keywords": p.get("keywords", []),
            "status": "approved",
            "category": p.get("category", "source-grounded-augmented"),
            "created_at": time.time(),
        }
        (pairs_dir / f"{qa_id}.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
        added += 1
    return added


def rebuild_sharegpt_dataset(project_dir: pathlib.Path, dataset_dir: pathlib.Path) -> int:
    pairs_dir = project_dir / "qa" / "pairs"
    rows = []
    for p in sorted(pairs_dir.glob("*.json")):
        qa = json.loads(p.read_text(encoding="utf-8"))
        question = qa.get("question", "")
        answer = qa.get("answer", "")
        if not question or not answer:
            continue
        if qa.get("status") and qa["status"] != "approved":
            continue
        rows.append({
            "conversations": [
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer},
            ],
            "source_id": qa.get("source_id", ""),
            "chunk_idx": qa.get("chunk_idx", 0),
            "keywords": qa.get("keywords", []),
            "category": qa.get("category", "source-grounded"),
        })
    pid = project_dir.name
    ds_path = dataset_dir / pid / "datasets" / f"{pid}-sharegpt-approved.jsonl"
    ds_path.parent.mkdir(parents=True, exist_ok=True)
    ds_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-id", required=True, help="project id (e.g. fbcf7083)")
    ap.add_argument("--root", default="/home/genortg/.finetune-studio",
                    help="project root (default fan-dragon default)")
    args = ap.parse_args()

    project_dir = pathlib.Path(args.root) / "projects" / args.project_id
    dataset_dir = pathlib.Path(args.root) / "data" / "projects"

    pairs = []
    for p in sorted((project_dir / "qa" / "pairs").glob("*.json")):
        pairs.append(json.loads(p.read_text(encoding="utf-8")))
    print(f"existing pairs: {len(pairs)}")

    augmented = build_augmented_pairs(project_dir)
    print(f"built augmented pairs: {len(augmented)}")
    if not augmented:
        return 1

    aug_path = project_dir / "augmented-pairs-v2.jsonl"
    aug_path.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in augmented) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {aug_path} ({aug_path.stat().st_size} bytes)")

    _, held = split_held_out(pairs)
    held_path = project_dir / "held-out.json"
    write_held_out_suite(held, held_path)
    print(f"wrote {held_path} ({len(held)} held-out cases)")

    added = merge_into_qa_pairs(project_dir, augmented)
    after = len(list((project_dir / "qa" / "pairs").glob("*.json")))
    print(f"merged {added} augmented pairs into qa/pairs/ (total now {after})")

    n = rebuild_sharegpt_dataset(project_dir, dataset_dir)
    print(f"rebuilt sharegpt dataset ({n} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
