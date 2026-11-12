"""`fts validate-hallucination` — count training answers with claims to check by hand (a pattern scan)."""
from __future__ import annotations

import json as json_mod


def cmd_validate_hallucination(args) -> None:
    from finetune_studio.training.hallucination_guard import TrainingDataValidator

    # Load data
    data = []
    bad_lines: list[int] = []
    with open(args.data) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json_mod.loads(line))
                except json_mod.JSONDecodeError:
                    bad_lines.append(n)

    if bad_lines:
        print(f"Skipped {len(bad_lines)} line(s) that are not JSON: {bad_lines[:20]}")
    print(f"Scanning {len(data)} examples for dates, figures, citations, URLs and e-mails (not a fact check)...")

    validator = TrainingDataValidator()
    result = validator.validate_dataset(data)

    print(f"\nFlagged claims: {result['total_risks']}")
    print(f"Risk types: {result['risk_types']}")
    print(f"Recommendation: {result['recommendation']}")

    if args.json:
        print(json_mod.dumps(result, indent=2))
