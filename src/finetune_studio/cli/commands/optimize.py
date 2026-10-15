"""`fts optimize` — recommend training hyperparameters from data."""
from __future__ import annotations

import json as json_mod


def cmd_optimize(args) -> None:
    from finetune_studio.training.config_optimizer import TrainingConfigOptimizer

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

    print(f"Analyzing {len(data)} examples...")

    # Build current config
    current = {}
    if args.lr:
        current['learning_rate'] = args.lr
    if args.epochs:
        current['num_epochs'] = args.epochs
    if args.lora_rank:
        current['lora_rank'] = args.lora_rank

    optimizer = TrainingConfigOptimizer()
    recommendations = optimizer.analyze_and_recommend(data, current)

    if args.json:
        print(json_mod.dumps([{'parameter': r.parameter, 'current': r.current_value,
                             'recommended': r.recommended_value, 'reason': r.reason,
                             'priority': r.priority} for r in recommendations], indent=2))
    else:
        print(optimizer.generate_report(recommendations))
