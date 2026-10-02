"""`fts augment` — generate synthetic examples to fix dataset weaknesses."""
from __future__ import annotations

import json as json_mod
import sys

# CLI-friendly names -> ``DataAugmenter.generators`` keys.
_TYPE_ALIASES = {
    "knowledge": "knowledge",
    "refusal": "refusal",
    "language": "language_balance",
    "language_balance": "language_balance",
    "hallucination": "hallucination_guard",
    "hallucination_guard": "hallucination_guard",
    "persona": "persona_preservation",
    "persona_preservation": "persona_preservation",
}


def cmd_augment(args) -> None:
    from finetune_studio.training.data_augmentation import DataAugmenter
    from finetune_studio.training.data_quality import DataQualityAnalyzer

    # Resolve --type up front so an unknown name fails instead of being skipped.
    requested = [t.strip() for t in args.type.split(",") if t.strip()]
    unknown = [t for t in requested if t != "all" and t not in _TYPE_ALIASES]
    if unknown:
        print(
            f"Error: unknown augmentation type(s): {', '.join(unknown)}. "
            f"Valid: all, {', '.join(sorted(_TYPE_ALIASES))}"
        )
        sys.exit(2)

    # Load existing data
    data = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json_mod.loads(line))
                except json_mod.JSONDecodeError:
                    pass

    print(f"Loaded {len(data)} examples from {args.data}")

    # Analyze weaknesses
    analyzer = DataQualityAnalyzer()
    analysis = analyzer.analyze(args.data)

    weaknesses = []
    for issue in analysis['issues']:
        if 'language' in issue.get('type', ''):
            weaknesses.append('language_balance')
        elif 'hallucination' in issue.get('type', ''):
            weaknesses.append('hallucination_guard')
        elif 'empty' in issue.get('type', ''):
            weaknesses.append('refusal')

    # Add default augmentations
    if "all" in requested or not requested:
        weaknesses.extend(['knowledge', 'refusal'])
    weaknesses.extend(_TYPE_ALIASES[t] for t in requested if t != "all")

    weaknesses = list(set(weaknesses))
    print(f"Augmenting for: {', '.join(weaknesses)}")

    # Augment
    augmenter = DataAugmenter()
    augmented = augmenter.augment_dataset(data, weaknesses)

    print(f"Augmented dataset: {len(data)} -> {len(augmented)} examples")

    # Save
    with open(args.output, 'w') as f:
        f.writelines(json_mod.dumps(item, ensure_ascii=False) + '\n' for item in augmented)

    print(f"Saved to {args.output}")
