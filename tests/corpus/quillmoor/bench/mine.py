"""Mining benchmark: the app's own "files -> Q&A pairs" step on a whole Quillmoor tier, and what it covers.

    .venv/bin/python tests/corpus/quillmoor/bench/mine.py --corpus .tmp/quillmoor/s --project agent-test-quillmoor-s

Every parsed file goes to POST /api/projects/{pid}/mining/jobs (the Pairs page's Mine button); the AI assistant's
Mining job writes the pairs. Then: how many known facts does at least one pair state (a pair from a file that states
the fact, whose answer holds the value in any accepted spelling)? Finally the pairs are exported as a dataset without
review (GET .../data-prep/export?only=all&force=true&grounded_share=0) for an adapter trained on mined data.
Results: <corpus>/results/mining.json.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path, PurePath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    check,
    client,
    load_jsonl,
    norm,
    project_by_name,
    write_json,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--skip-mining", action="store_true", help="only measure + export what is there")
    a = ap.parse_args()
    corpus = Path(a.corpus)
    c = client(timeout=600)
    pid = project_by_name(c, a.project)["id"]
    st = check(c.get(f"/api/projects/{pid}/files/pipeline"))["status"]
    sources = {v["source_id"]: fid for fid, v in st.items() if v.get("has_parsed") and v.get("source_id")}
    files = {f["id"]: f["original_name"] for f in check(c.get(f"/api/projects/{pid}/files"))["files"]}
    src_name = {sid: files.get(fid, "") for sid, fid in sources.items()}
    t0 = time.time()
    jobs: list[dict] = []
    if not a.skip_mining:
        check(c.post(f"/api/projects/{pid}/mining/jobs", json={"source_ids": list(sources), "kind": "mine"}))
        last = ""
        while True:
            jobs = check(c.get(f"/api/projects/{pid}/mining/jobs", params={"limit": 500}))["jobs"]
            counts = Counter(j["status"] for j in jobs)
            line = str(sorted(counts.items()))
            if line != last:
                print(f"t={time.time() - t0:.0f}s {line}", flush=True)
                last = line
            if not any(j["status"] in ("queued", "running") for j in jobs):
                break
            time.sleep(15)
    mine_s = round(time.time() - t0, 1)
    pairs = check(c.get(f"/api/projects/{pid}/data-prep/qa"))["items"]
    by_file: dict[str, list[str]] = defaultdict(list)
    for p in pairs:
        by_file[PurePath(src_name.get(p.get("source_id", ""), "")).name].append(norm(p.get("answer") or ""))
    facts = load_jsonl(corpus / "manifest" / "facts.jsonl")
    covered = Counter()
    total = Counter()
    for f in facts:
        if not f["stated_in"]:
            continue
        total[f["etype"]] += 1
        files_f = {PurePath(x).name for x in f["stated_in"]}
        vals = [norm(v) for v in f["accept"]]
        if any(any(v in ans for v in vals) for fn in files_f for ans in by_file.get(fn, [])):
            covered[f["etype"]] += 1
    n_cov, n_tot = sum(covered.values()), sum(total.values())
    calls = sum((j.get("stats") or {}).get("calls", 0) for j in jobs)
    exp = c.get(f"/api/projects/{pid}/data-prep/export", params={"only": "all", "force": "true", "grounded_share": 0})
    rows = sum(1 for line in exp.text.splitlines() if line.strip()) if exp.status_code == 200 else 0
    out = {"files": len(sources), "mine_s": mine_s, "jobs": dict(Counter(j["status"] for j in jobs)), "helper_calls": calls,
           "pairs": len(pairs), "pair_status": dict(Counter(p.get("status") for p in pairs)),
           "fact_recall": round(n_cov / max(n_tot, 1), 4), "facts_covered": n_cov, "facts": n_tot,
           "recall_by_entity": {k: round(covered[k] / total[k], 3) for k in sorted(total)},
           "export_status": exp.status_code, "export_rows": rows}
    write_json(corpus / "results" / "mining.json", out)
    print({k: out[k] for k in ("files", "mine_s", "jobs", "helper_calls", "pairs", "fact_recall", "export_rows")}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
