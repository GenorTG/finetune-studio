"""Grade answer files with the app's own AI judge (finetune_studio.testing.judge), in place and resumable.

    .venv/bin/python tests/corpus/quillmoor/bench/judge.py <answers.jsonl> [...]

The judge is the AI assistant's Test-judge job (Settings -> AI assistant), the same one the Testing page uses.
Each row gets ``verdict`` pass / partial / fail (empty + ``judge_error`` when the judge call failed: never guessed).
Outdated values are spelled out in the answer key, so an answer that gives the old value is graded wrong.
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WORKERS = 8
JUDGE_PROTOCOL = "all-accepted-spellings-v2"
VALID_VERDICTS = frozenset({"pass", "partial", "fail"})


def has_valid_judgment(row: dict) -> bool:
    verdict = row.get("verdict")
    judge_model = row.get("judge_model")
    return (
        isinstance(verdict, str)
        and verdict in VALID_VERDICTS
        and not row.get("judge_error")
        and row.get("benchmark_judge_protocol") == JUDGE_PROTOCOL
        and isinstance(judge_model, str)
        and bool(judge_model.strip())
    )


def needs_judge(row: dict, *, rejudge: bool = False) -> bool:
    return not row.get("error") and (rejudge or not has_valid_judgment(row))


def case_for(row: dict):
    from finetune_studio.testing.judge import JudgeCase
    expect_abstain = row["type"] in ("unanswerable", "sft-heldout")
    key = "" if row["type"] == "sft-heldout" else row["key"]
    if row["type"] == "superseded" and row.get("wrong"):
        key = f"{key} (the earlier value {', '.join(row['wrong'])} is outdated and counts as wrong)"
    return JudgeCase(question=row["question"], correct_answer=key, model_answer=row["answer"],
                     keywords=row["accept"], expect_abstain=expect_abstain)


def judge_files(paths: list[Path], provider_id: str = "", *, rejudge: bool = False) -> dict[str, int]:
    from finetune_studio.testing.judge import (
        default_judge_provider_id,
        judge_one,
        open_judge,
    )

    stats = {"judged": 0, "errors": 0, "skipped": 0}
    with open_judge(provider_id or default_judge_provider_id()) as judge:
        for path in paths:
            rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
            todo = [i for i, row in enumerate(rows) if needs_judge(row, rejudge=rejudge)]
            stats["skipped"] += len(rows) - len(todo)
            lock = threading.Lock()

            def one(i: int, rows: list[dict] = rows, lock: threading.Lock = lock) -> None:
                res = judge_one(judge.chat, case_for(rows[i]))
                with lock:
                    rows[i]["verdict"] = res.verdict
                    rows[i]["judge_reasoning"] = res.reasoning
                    rows[i]["judge_error"] = res.error
                    rows[i]["judge_model"] = judge.model
                    rows[i]["benchmark_judge_protocol"] = JUDGE_PROTOCOL
                    stats["judged" if has_valid_judgment(rows[i]) else "errors"] += 1

            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                list(ex.map(one, todo))
            tmp = path.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            tmp.replace(path)
    return stats


def score(rows: list[dict]) -> dict:
    """Score unique questions; report response and judge coverage separately from quality."""
    latest: dict[str, dict] = {}
    for index, row in enumerate(rows):
        key = str(row.get("id", f"__row_{index}"))
        current = latest.get(key)
        successful = not row.get("error") and bool(str(row.get("answer") or "").strip())
        current_successful = current is not None and not current.get("error") and bool(
            str(current.get("answer") or "").strip()
        )
        if current is None or successful or not current_successful:
            latest[key] = row

    unique_rows = list(latest.values())
    answered = [r for r in unique_rows if not r.get("error") and str(r.get("answer") or "").strip()]
    judged = [r for r in answered if has_valid_judgment(r)]
    untrusted_verdicts = sum(bool(r.get("verdict")) and not has_valid_judgment(r) for r in answered)
    out: dict = {
        "n": len(unique_rows),
        "answered": len(answered),
        "judged": len(judged),
        "untrusted_verdicts": untrusted_verdicts,
        "answer_coverage": round(len(answered) / max(len(unique_rows), 1), 4),
        "judge_coverage": round(len(judged) / max(len(unique_rows), 1), 4),
    }
    if judged:
        out["pass"] = round(sum(r["verdict"] == "pass" for r in judged) / len(judged), 4)
        out["score"] = round(sum({"pass": 1, "partial": 0.5}.get(r["verdict"], 0) for r in judged) / len(judged), 4)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rejudge", action="store_true", help="re-run every successful answer with the current judge protocol")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    print(judge_files(args.paths, rejudge=args.rejudge))
