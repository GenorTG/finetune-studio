"""Did the parsers keep every fact? Checks manifest values against the project's parsed source text.

    FTS_ROOT=~/.finetune-studio .venv/bin/python scripts/corpus_parse_check.py --pid <project> [--tier core]

Isolates parser loss from miner loss: a fact whose values are absent from ``parsed.txt`` can never be mined, whatever the
helper does (tables dropped, scanned page unreadable, header lines stripped, chart numbers inside an image ...).
Exit 1 when any fact is lost.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_coverage import BASE, manifest_facts, norm, squash


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", required=True)
    ap.add_argument("--tier", default="core", choices=["core", "extended", "all"])
    a = ap.parse_args()
    from finetune_studio.data.fs.paths import file_dir

    with urllib.request.urlopen(f"{BASE}/api/projects/{a.pid}/data-prep/sources", timeout=60) as r:
        sources = json.loads(r.read())["sources"]
    by_name = {s["filename"]: s for s in sources}
    lost: list[dict] = []
    total = 0
    print(f"{'file':58} kept/total   parsed chars")
    for name in sorted({f["file"] for f in manifest_facts(a.tier, None)}):
        facts = [f for f in manifest_facts(a.tier, None) if f["file"] == name]
        src = by_name.get(name)
        if src is None:
            print(f"{name:58} NOT UPLOADED")
            lost += facts
            total += len(facts)
            continue
        parsed = (file_dir(a.pid, src["sha256"], create=False) / "parsed.txt").read_text(encoding="utf-8", errors="replace")
        miss = [f for f in facts if not all(norm(v) in norm(parsed) or squash(v) in squash(parsed) for v in f["values"])]
        total += len(facts)
        lost += miss
        print(f"{name:58} {len(facts) - len(miss):4}/{len(facts):<4}   {len(parsed):7}")
        for f in miss[:25]:
            print(f"   LOST {f['id']} [{f['kind']} · {f['locator']}] {f['values']}")
    kinds = Counter(f["kind"] for f in lost)
    print(f"TOTAL kept {total - len(lost)}/{total}; lost by kind: {dict(kinds)}")
    return 1 if lost else 0


if __name__ == "__main__":
    raise SystemExit(main())
