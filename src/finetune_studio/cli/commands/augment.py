"""`fts augment` — generate synthetic examples to fix dataset weaknesses."""
from __future__ import annotations

import json as json_mod


def cmd_augment(args) -> None:
    from finetune_studio.training.data_augmentation import DataAugmenter
    from finetune_studio.training.data_quality import DataQualityAnalyzer

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
    if args.type == 'all':
        weaknesses.extend(['knowledge', 'refusal'])
    else:
        weaknesses.extend(args.type.split(','))

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
