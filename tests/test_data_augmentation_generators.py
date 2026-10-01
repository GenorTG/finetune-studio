"""Regression test (2026-10-01 audit fix): DataAugmenter.augment_dataset's
uniform ``generator(count=...)`` dispatch crashed with a TypeError for two
of its five registered generators.

``generate_language_balanced_data(self, pl_examples, target_en=100)`` and
``generate_persona_preservation(self, persona_data, count=50)`` both need
the source dataset as their first positional argument (they transform
existing items rather than generating from scratch like the other three
generators), and the former's second parameter isn't even named ``count``.
Calling either via ``weaknesses=["language_balance"]`` or
``weaknesses=["persona_preservation"]`` — exactly what
``webui/routes/quality.py``'s ``/api/data/augment`` route and the
``fts augment`` CLI command both do when the data-quality analyzer detects
a language or persona issue — always raised before this fix.
"""

from __future__ import annotations

from finetune_studio.training.data_augmentation import DataAugmenter

SOURCE = [
    {
        "messages": [
            {"role": "user", "content": "Czym jest Python?"},
            {"role": "assistant", "content": "A programming language."},
        ],
    },
]


class TestAugmentDatasetGeneratorDispatch:
    def test_every_registered_weakness_runs_without_crashing(self):
        augmenter = DataAugmenter()
        for weakness in augmenter.generators:
            # Must not raise for any weakness the module itself registers.
            result = augmenter.augment_dataset(list(SOURCE), [weakness])
            assert isinstance(result, list)
            assert len(result) >= len(SOURCE)

    def test_language_balance_translates_recognized_polish_prefix(self):
        augmenter = DataAugmenter()
        result = augmenter.augment_dataset(list(SOURCE), ["language_balance"])
        translated = [
            item for item in result
            if item["messages"][0]["content"].startswith("What is")
        ]
        assert translated, result

    def test_persona_preservation_passes_through_user_first_items(self):
        augmenter = DataAugmenter()
        result = augmenter.augment_dataset(list(SOURCE), ["persona_preservation"])
        # Original item plus the persona-preserved copy.
        assert len(result) == 2
