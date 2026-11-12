"""Generate the Quillmoor benchmark corpus.

    .venv/bin/python tests/corpus/quillmoor/generate.py --tier s --out .tmp/quillmoor/s

Writes:
  files/                 the generated documents (42 extensions), the thing you upload into the app
  manifest/facts.jsonl   every fact: value, accepted renderings, outdated values, which files state it
  manifest/docs.jsonl    every file: kind, type, how many facts, size
  eval/questions.jsonl   the question set (see qmgen/questions.py for the types)
  train/train.jsonl      chat-format training rows (direct phrasing, current values only)
  summary.json           counts per type / format / tier
Deterministic: the same tier and seed give byte-identical text content (binary containers may embed timestamps).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qmgen.docs import Builder
from qmgen.questions import QuestionMaker, doc_format
from qmgen.world import World


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=["s", "m", "l"], default="s")
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-render", action="store_true", help="plan, check and write the manifests only")
    ap.add_argument("--per-format", type=int, default=0, help="paraphrase questions per file type (default by tier)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    t0 = time.time()
    w = World(a.tier, a.seed)
    b = Builder(w)
    plans = b.plan()
    b.assign_formats(plans)
    docs = [b.build(p) for p in plans]
    # every recorded value must literally be in the document's text
    bad = [(d.relpath, m.written) for d in docs for m in d.facts if m.written not in d.plain_text()]
    if bad:
        print(f"{len(bad)} fact marks not in their document text, e.g. {bad[:3]}", file=sys.stderr)
        return 2
    unstated = [f.fid for f in w.facts.values() if not f.stated_in]
    print(f"world: {sum(len(v) for v in w.e.values())} entities, {len(w.facts)} facts ({len(unstated)} never stated), "
          f"{len(docs)} documents in {time.time() - t0:.1f}s", flush=True)
    for sub in ("manifest", "eval", "train"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    if not a.no_render:
        from qmgen.render import render_all
        t1 = time.time()
        res = render_all(docs, out / "files", workers=a.workers)
        errs = {k: v for k, v in res.items() if v != "ok"}
        print(f"rendered {len(res) - len(errs)}/{len(res)} files in {time.time() - t1:.0f}s", flush=True)
        if errs:
            for k, v in list(errs.items())[:20]:
                print(f"  {k}: {v}", file=sys.stderr)
            print("corpus generation stopped: at least one document failed to render", file=sys.stderr)
            return 3
    by_rel = {d.relpath: d for d in docs}
    with open(out / "manifest" / "docs.jsonl", "w") as fh:
        for d in docs:
            p = out / "files" / d.relpath
            fh.write(json.dumps({"path": d.relpath, "kind": d.kind, "ext": d.ext, "format": doc_format(d), "facts": len(d.facts),
                                 "bytes": p.stat().st_size if p.exists() else None,
                                 "marks": [[m.fact_id, m.written, m.superseded] for m in d.facts]}, ensure_ascii=False) + "\n")
    with open(out / "manifest" / "facts.jsonl", "w") as fh:
        ids = {d.doc_id: d.relpath for d in docs}
        fh.writelines(
            json.dumps({"id": f.fid, "etype": f.etype, "entity": f.ekey, "attr": f.attr, "value": f.canonical(),
                        "accept": f.accepted(), "old": [f.canonical(o) for o in f.old],
                        "stated_in": [ids[x] for x in f.stated_in], "old_in": [ids[x] for x in f.old_in]},
                       ensure_ascii=False) + "\n"
            for f in w.facts.values()
        )
    qm = QuestionMaker(w, docs)
    per_format = a.per_format or {"s": 12, "m": 20, "l": 24}[a.tier]
    qs = qm.build(per_format=per_format)
    with open(out / "eval" / "questions.jsonl", "w") as fh:
        fh.writelines(json.dumps(q, ensure_ascii=False) + "\n" for q in qs)
    rows = qm.training_rows()
    with open(out / "train" / "train.jsonl", "w") as fh:
        fh.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    fmt_files = Counter(doc_format(d) for d in docs)
    fmt_facts = Counter()
    for d in docs:
        fmt_facts[doc_format(d)] += len(d.facts)
    summary = {
        "tier": a.tier, "seed": a.seed, "entities": {k: len(v) for k, v in w.e.items()}, "facts": len(w.facts),
        "facts_unstated": len(unstated), "facts_superseded": sum(1 for f in w.facts.values() if f.old),
        "documents": len(docs), "bytes": sum((out / "files" / r).stat().st_size for r in by_rel if (out / "files" / r).exists()),
        "files_per_format": dict(sorted(fmt_files.items())), "fact_marks_per_format": dict(sorted(fmt_facts.items())),
        "documents_per_kind": dict(Counter(d.kind for d in docs).most_common()),
        "questions": dict(Counter(q["type"] for q in qs)), "train_rows": len(rows),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ("facts", "documents", "bytes", "questions", "train_rows")}), flush=True)
    print(f"done in {time.time() - t0:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
