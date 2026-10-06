#!/usr/bin/env python3
"""Does preference tuning change behaviour? SFT -> DPO/KTO on a fictional corpus, scored before/after.

Runs through ``TrainingEngine`` (the same spawn-child path the WebUI uses), never the live
service. Corpus = invented orders with five documented facts each, so a model cannot know
them from pre-training. Stages:

1. ``sft``  - LoRA SFT on the facts (3 question phrasings per fact), merged.
2. ``pref`` - DPO (or KTO) from the merged SFT model on pairs where
   chosen = the source-faithful answer, rejected = a confident wrong one, plus abstain pairs
   (question the documents do not answer -> ``REFUSAL`` vs a confident fabrication).
3. ``eval`` - greedy answers on held-out phrasings / undocumented attributes for the base
   model, the SFT model and the preference model.

Scores: in-corpus accuracy (gold value appears in the answer), false-refusal rate on those
questions, and refusal rate on out-of-corpus questions. n is small by design (a ~10 min run):
read the counts, not just the percentages.

    CUDA_VISIBLE_DEVICES=0 python scripts/pref_quality_eval.py --model <hf dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

REFUSAL = "That isn't covered in the provided documents."
SYSTEM = (
    "Answer only from the provided documents. If the documents do not contain the answer, "
    f"reply exactly: {REFUSAL}"
)

ORGS = [
    "Order of the Salt Lantern", "Brineward Cartographers' Guild", "Choir of Hollow Bells",
    "Harrowgate Tidewrights", "Society of the Ninth Mirror", "Ashmere Glassblowers' Union",
]
FOUNDERS = ["Ilsabet Corr", "Maren Voss", "Dovan Ketterich", "Orla Pennick", "Sefa Lindqvist", "Tomas Vael"]
PLACES = ["Orrin Deep", "Port Selene", "Lantern Vault", "Harrowgate Quay", "Marrow Hill", "Frostmere"]

# attribute -> (documented sentence template, answer sentence template, train/eval phrasings)
ATTRS: dict[str, dict] = {
    "founded": {
        "doc": "The {org} was founded in the year {v}.",
        "ans": "The {org} was founded in the year {v}.",
        "train": ["In which year was the {org} founded?", "When was the {org} founded?",
                  "What year did the {org} come into being?"],
        "eval": ["Tell me the founding year of the {org}.", "The {org} began in which year?"],
    },
    "founder": {
        "doc": "Its founder was {v}.",
        "ans": "The {org} was founded by {v}.",
        "train": ["Who founded the {org}?", "Who is the founder of the {org}?",
                  "Which person started the {org}?"],
        "eval": ["Name the founder of the {org}.", "The {org} was started by whom?"],
    },
    "members": {
        "doc": "It has {v} sworn members.",
        "ans": "The {org} has {v} sworn members.",
        "train": ["How many members does the {org} have?", "What is the membership of the {org}?",
                  "How many sworn members belong to the {org}?"],
        "eval": ["Give the number of members in the {org}.", "The {org} counts how many members?"],
    },
    "seat": {
        "doc": "Its seat is {v}.",
        "ans": "The {org} is headquartered at {v}.",
        "train": ["Where is the {org} headquartered?", "Where is the seat of the {org}?",
                  "Which place is the base of the {org}?"],
        "eval": ["State the headquarters of the {org}.", "The {org} is based where?"],
    },
    "fee": {
        "doc": "The annual fee is {v} crowns.",
        "ans": "The annual fee of the {org} is {v} crowns.",
        "train": ["What is the annual fee of the {org}?", "How much does the {org} charge per year?",
                  "What yearly dues does the {org} ask for?"],
        "eval": ["Tell me the yearly fee of the {org}.", "The {org} charges how many crowns a year?"],
    },
}
# Undocumented attributes: the train ones feed abstain pairs, the eval ones are never seen.
OOD_TRAIN = {
    "motto": "What is the motto of the {org}?", "leader": "Who currently leads the {org}?",
    "ships": "How many ships does the {org} own?", "robes": "What colour are the robes of the {org}?",
}
# New phrasings of the *trained* undocumented attributes: tests that abstaining generalises
# beyond the exact training questions (the novel-attribute set below is the harder test).
OOD_TRAIN_PARAPHRASE = {
    "motto": "Which motto does the {org} live by?", "leader": "Who is the head of the {org} right now?",
    "ships": "How big is the fleet of the {org}?", "robes": "Which colour do members of the {org} wear?",
}
OOD_EVAL = {
    "festival": "On what date does the {org} hold its yearly festival?",
    "treasury": "How large is the treasury of the {org}?",
    "patron": "Who is the patron saint of the {org}?",
    "emblem": "What emblem appears on the banner of the {org}?",
    "rival": "Which organisation is the main rival of the {org}?",
}


def build_facts(seed: int = 7) -> list[dict]:
    """30 facts (6 orgs x 5 attributes) with deterministic invented values."""
    rng = random.Random(seed)
    facts = []
    for i, org in enumerate(ORGS):
        values = {
            "founded": str(rng.randrange(1040, 1490)),
            "founder": FOUNDERS[i],
            "members": str(rng.randrange(40, 990)),
            "seat": PLACES[i],
            "fee": str(rng.randrange(12, 480)),
        }
        for attr, value in values.items():
            facts.append({"org": org, "attr": attr, "value": value})
    return facts


def wrong_value(facts: list[dict], fact: dict, rng: random.Random) -> str:
    """A confident wrong value of the same type (another org's value, or a shifted number)."""
    pool = [f["value"] for f in facts if f["attr"] == fact["attr"] and f["value"] != fact["value"]]
    if fact["attr"] in ("founded", "members", "fee"):
        shift = rng.choice([-1, 1]) * rng.randrange(7, 90)
        return str(max(1, int(fact["value"]) + shift))
    return rng.choice(pool)


def answer_text(fact: dict, value: str | None = None) -> str:
    return ATTRS[fact["attr"]]["ans"].format(org=fact["org"], v=value or fact["value"])


def corpus_docs(facts: list[dict]) -> dict[str, str]:
    """One short source document per org (written next to the run for provenance)."""
    docs: dict[str, str] = {}
    for org in ORGS:
        sents = [ATTRS[f["attr"]]["doc"].format(org=org, v=f["value"]) for f in facts if f["org"] == org]
        docs[org] = f"{org}. " + " ".join(sents)
    return docs


def make_sft_rows(facts: list[dict], n_abstain: int = 0, seed: int = 5) -> list[dict]:
    """Fact rows (3 phrasings each) plus ``n_abstain`` refusal rows on the *train* OOD attributes."""
    rows = [
        {"messages": [
            {"role": "user", "content": phr.format(org=f["org"])},
            {"role": "assistant", "content": answer_text(f)},
        ]}
        for f in facts for phr in ATTRS[f["attr"]]["train"]
    ]
    combos = [(org, attr) for org in ORGS for attr in OOD_TRAIN]
    random.Random(seed).shuffle(combos)
    rows += [
        {"messages": [
            {"role": "user", "content": OOD_TRAIN[attr].format(org=org)},
            {"role": "assistant", "content": REFUSAL},
        ]}
        for org, attr in combos[:n_abstain]
    ]
    return rows


def make_pref_rows(facts: list[dict], seed: int = 11, n_pairs: int = 60, n_abstain: int = 20) -> list[dict]:
    """``n_pairs`` fact pairs (wrong-value rejected) + ``n_abstain`` abstain pairs."""
    rng = random.Random(seed)
    rows: list[dict] = []
    picks = [(f, ATTRS[f["attr"]]["train"][rng.randrange(3)]) for f in facts for _ in range(2)]
    rng.shuffle(picks)
    for fact, phr in picks[:n_pairs]:
        rows.append({
            "prompt": [{"role": "user", "content": phr.format(org=fact["org"])}],
            "chosen": [{"role": "assistant", "content": answer_text(fact)}],
            "rejected": [{"role": "assistant", "content": answer_text(fact, wrong_value(facts, fact, rng))}],
        })
    combos = [(org, attr) for org in ORGS for attr in OOD_TRAIN]
    rng.shuffle(combos)
    fabrications = {
        "motto": "The motto of the {org} is 'Steady Hands, Steady Tide'.",
        "leader": "The {org} is currently led by Warden Ressik Dun.",
        "ships": "The {org} owns 14 ships.",
        "robes": "The robes of the {org} are deep grey.",
    }
    for org, attr in combos[:n_abstain]:
        rows.append({
            "prompt": [{"role": "user", "content": OOD_TRAIN[attr].format(org=org)}],
            "chosen": [{"role": "assistant", "content": REFUSAL}],
            "rejected": [{"role": "assistant", "content": fabrications[attr].format(org=org)}],
        })
    rng.shuffle(rows)
    return rows


def make_eval_set(facts: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Held-out phrasings of documented facts, trained-attribute paraphrases, novel undocumented attributes."""
    inc = [{"q": phr.format(org=f["org"]), "gold": f["value"], "attr": f["attr"]}
           for f in facts for phr in ATTRS[f["attr"]]["eval"]]
    seen = [{"q": tmpl.format(org=org), "attr": attr}
            for org in ORGS for attr, tmpl in OOD_TRAIN_PARAPHRASE.items()]
    novel = [{"q": tmpl.format(org=org), "attr": attr} for org in ORGS for attr, tmpl in OOD_EVAL.items()]
    return inc, seen, novel


def is_refusal(text: str) -> bool:
    low = text.lower()
    return "isn't covered" in low or "not covered" in low or "is not covered" in low


def generate(model_dir: str, questions: list[str], *, max_new: int = 48, adapter: str | None = None) -> list[str]:
    """Greedy answers, thinking off, under the same system prompt as training."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to("cuda:0").eval()
    out: list[str] = []
    for i in range(0, len(questions), 32):
        chunk = questions[i:i + 32]
        prompts = [
            tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            ) for q in chunk
        ]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        out += [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for g in gen]
    del model
    torch.cuda.empty_cache()
    return out


def score(model_dir: str, inc: list[dict], seen: list[dict], ood: list[dict]) -> dict:
    answers = generate(model_dir, [x["q"] for x in inc + seen + ood])
    a_in, a_seen, a_ood = answers[:len(inc)], answers[len(inc):len(inc) + len(seen)], answers[len(inc) + len(seen):]
    correct = sum(1 for x, a in zip(inc, a_in, strict=True) if x["gold"].lower() in a.lower() and not is_refusal(a))
    false_ref = sum(1 for a in a_in if is_refusal(a))
    return {
        "in_corpus_n": len(inc), "in_corpus_correct": correct, "in_corpus_false_refusals": false_ref,
        "ood_seen_n": len(seen), "ood_seen_refused": sum(1 for a in a_seen if is_refusal(a)),
        "ood_n": len(ood), "ood_refused": sum(1 for a in a_ood if is_refusal(a)),
        "samples": {"in": list(zip([x["q"] for x in inc], a_in, strict=True))[:4],
                    "ood": list(zip([x["q"] for x in ood], a_ood, strict=True))[:4]},
    }


class VramWatcher(threading.Thread):
    """Polls GPU0 used MiB once a second; ``peak`` is the max seen."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.peak = 0
        self.stop_flag = threading.Event()

    def run(self) -> None:
        while not self.stop_flag.is_set():
            res = subprocess.run(
                ["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False,
            )
            try:
                self.peak = max(self.peak, int(res.stdout.strip()))
            except ValueError:
                pass
            time.sleep(1)


def run_engine(config, rows: list[dict], label: str) -> tuple[float, str]:
    """One engine run to completion; returns (wall seconds, last log lines)."""
    from finetune_studio.training.engine import TrainingEngine

    engine = TrainingEngine()
    started = time.time()
    engine.start(config, rows, SYSTEM)
    last = ""
    while engine._process is not None:  # the listener thread clears it once the child is gone
        msg = f"[{label}] {engine.state.status} {engine.state.current_step}/{engine.state.total_steps} {engine.state.message}"
        if msg != last:
            print(msg, flush=True)
            last = msg
        time.sleep(1)
    wall = time.time() - started
    if engine.state.status != "done":
        raise SystemExit(f"{label} failed: {engine.state.status} {engine.state.error}")
    print(f"[{label}] done in {wall:.0f}s\n  " + "\n  ".join(engine.state.log_lines[-6:]), flush=True)
    return wall, "\n".join(engine.state.log_lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF dir of the base model (e.g. Qwen3-0.6B)")
    ap.add_argument("--out", required=True, help="scratch dir for corpus, runs and results")
    ap.add_argument("--mode", choices=["dpo", "kto"], default="dpo")
    ap.add_argument("--sft-epochs", type=int, default=6)
    ap.add_argument("--sft-lr", type=float, default=2e-4)
    ap.add_argument("--pref-epochs", type=int, default=3)
    ap.add_argument("--pref-lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--sft-weight", type=float, default=0.0, help="DPO only: extra NLL-on-chosen weight")
    ap.add_argument("--pairs", type=int, default=60)
    ap.add_argument("--abstain", type=int, default=20)
    ap.add_argument("--sft-abstain", type=int, default=0,
                    help="refusal rows mixed into SFT (0 = SFT never sees a refusal)")
    ap.add_argument("--skip-sft", action="store_true", help="reuse <out>/sft/merged")
    ap.add_argument("--data-seed", type=int, default=11, help="seed for pair picking / wrong values")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from finetune_studio.training.engine import TrainingConfig

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    facts = build_facts()
    for org, text in corpus_docs(facts).items():
        (out / f"doc_{org.split()[-1].lower()}.txt").write_text(text)
    inc, seen, ood = make_eval_set(facts)
    sft_dir = out / "sft"
    pref_dir = out / f"{args.mode}{args.tag}"
    watcher = VramWatcher()
    watcher.start()
    t0 = time.time()
    report: dict = {"args": vars(args), "stages": {}}

    if not args.skip_sft:
        cfg = TrainingConfig(
            model_path=args.model, output_dir=str(sft_dir), lora_rank=32, lora_alpha=64,
            learning_rate=args.sft_lr, num_epochs=args.sft_epochs, batch_size=4,
            gradient_accumulation_steps=1, max_seq_length=512, warmup_steps=5, logging_steps=5,
            merge_on_save=True, unsloth=False, save_checkpoints=False, training_mode="sft",
        )
        wall, _ = run_engine(cfg, make_sft_rows(facts, args.sft_abstain), "sft")
        report["stages"]["sft_train_s"] = round(wall)
    rows = make_pref_rows(facts, seed=args.data_seed, n_pairs=args.pairs, n_abstain=args.abstain)
    (out / "pref_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    cfg = TrainingConfig(
        model_path=str(sft_dir / "merged"), output_dir=str(pref_dir), lora_rank=16, lora_alpha=32,
        learning_rate=args.pref_lr, num_epochs=args.pref_epochs, batch_size=4,
        gradient_accumulation_steps=1, max_seq_length=512, warmup_steps=0, logging_steps=5,
        merge_on_save=True, unsloth=False, save_checkpoints=False, eval_steps=6,
        training_mode=args.mode, preference_beta=args.beta,
        preference_sft_weight=args.sft_weight,
    )
    wall, log_text = run_engine(cfg, rows, args.mode)
    report["stages"][f"{args.mode}_train_s"] = round(wall)
    report["train_log_tail"] = log_text.splitlines()[-12:]

    report["scores"] = {
        "base": score(args.model, inc, seen, ood),
        "sft": score(str(sft_dir / "merged"), inc, seen, ood),
        args.mode: score(str(pref_dir / "merged"), inc, seen, ood),
    }
    watcher.stop_flag.set()
    report["vram_peak_mib_gpu0"] = watcher.peak
    report["wall_s"] = round(time.time() - t0)
    (out / f"report_{args.mode}{args.tag}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "train_log_tail"}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
