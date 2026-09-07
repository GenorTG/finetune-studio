"""`fts validate-hallucination` — flag risky training examples."""
from __future__ import annotations

import json as json_mod


def cmd_validate_hallucination(args) -> None:
    from finetune_studio.training.hallucination_guard import TrainingDataValidator

    # Load data
    data = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json_mod.loads(line))
                except json_mod.JSONDecodeError:
                    pass

    print(f"Checking {len(data)} examples for hallucination risks...")

    validator = TrainingDataValidator()
    result = validator.validate_dataset(data)

    print(f"\nTotal risks: {result['total_risks']}")
    print(f"Risk types: {result['risk_types']}")
    print(f"Recommendation: {result['recommendation']}")

    if args.json:
        print(json_mod.dumps(result, indent=2))
