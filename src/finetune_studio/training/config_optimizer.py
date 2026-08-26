from typing import Any

"""Suggest optimal training hyperparameters.

WHAT THIS FILE DOES
==================
Given a dataset and a target model, suggests optimal hyperparameters:
  - Learning rate (too high = unstable, too low = slow)
  - Batch size (limited by GPU memory)
  - Number of epochs (how many times to go through the data)
  - LoRA rank (how many parameters to train)
  - Sequence length (max tokens per example)

KEY CONCEPTS
============
- Hyperparameter search: instead of guessing, we can use heuristics
  based on dataset size and model size.
- Common defaults: learning rate=2e-4, batch size=4, epochs=3, LoRA rank=16.
- When to deviate: small datasets need fewer epochs; large models
  need lower learning rates.
"""

"""Training config optimizer — suggest optimal training settings."""
from dataclasses import dataclass


@dataclass
class TrainingRecommendation:
    parameter: str
    current_value: str
    recommended_value: str
    reason: str
    priority: str  # high, medium, low


class TrainingConfigOptimizer:
    """Analyze data and suggest optimal training configuration."""

    def __init__(self):
        self.rules = {
            "small_dataset": self._small_dataset_rules,
            "large_dataset": self._large_dataset_rules,
            "language_imbalance": self._language_imbalance_rules,
            "persona_focus": self._persona_focus_rules,
            "knowledge_preservation": self._knowledge_preservation_rules,
        }

    def analyze_and_recommend(self, data: list, current_config: dict | None = None) -> list:
        """Analyze data and recommend training configuration."""
        recommendations = []
        current = current_config or {}

        # Analyze dataset characteristics
        size = len(data)
        pl_ratio = self._calculate_pl_ratio(data)
        persona_ratio = self._calculate_persona_ratio(data)

        # Apply rules
        if size < 500:
            recommendations.extend(self._small_dataset_rules(current))
        elif size > 5000:
            recommendations.extend(self._large_dataset_rules(current))

        if pl_ratio > 0.8 or pl_ratio < 0.2:
            recommendations.extend(self._language_imbalance_rules(current, pl_ratio))

        if persona_ratio > 0.7:
            recommendations.extend(self._persona_focus_rules(current))

        if persona_ratio > 0.5:
            recommendations.extend(self._knowledge_preservation_rules(current))

        return recommendations

    def _calculate_pl_ratio(self, data: list) -> float:
        pl_count = 0
        total = 0
        for item in data:
            for msg in item.get("messages", []):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    pl_chars = sum(1 for c in content if c in "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
                    if pl_chars > len(content) * 0.05:
                        pl_count += 1
                    total += 1
        return pl_count / max(total, 1)

    def _calculate_persona_ratio(self, data: list) -> float:
        persona_keywords = ["company", "project", "work", "role", "team", "build", "develop"]
        persona_count = 0
        for item in data:
            for msg in item.get("messages", []):
                if msg.get("role") == "user":
                    content = msg.get("content", "").lower()
                    if any(kw in content for kw in persona_keywords):
                        persona_count += 1
                        break
        return persona_count / max(len(data), 1)

    def _small_dataset_rules(self, current: dict) -> list:
        return [
            TrainingRecommendation(
                parameter="learning_rate",
                current_value=str(current.get("learning_rate", "8e-5")),
                recommended_value="3e-5",
                reason="Lower LR for small datasets to prevent overfitting",
                priority="high"
            ),
            TrainingRecommendation(
                parameter="num_epochs",
                current_value=str(current.get("num_epochs", "4")),
                recommended_value="6",
                reason="More epochs for small datasets (but watch for overfitting)",
                priority="medium"
            ),
            TrainingRecommendation(
                parameter="lora_rank",
                current_value=str(current.get("lora_rank", "64")),
                recommended_value="32",
                reason="Lower rank for small datasets to reduce capacity",
                priority="medium"
            ),
            TrainingRecommendation(
                parameter="weight_decay",
                current_value=str(current.get("weight_decay", "0.005")),
                recommended_value="0.01",
                reason="Higher regularization for small datasets",
                priority="low"
            ),
        ]

    def _large_dataset_rules(self, current: dict) -> list:
        return [
            TrainingRecommendation(
                parameter="learning_rate",
                current_value=str(current.get("learning_rate", "8e-5")),
                recommended_value="1e-4",
                reason="Higher LR for large datasets to learn faster",
                priority="medium"
            ),
            TrainingRecommendation(
                parameter="num_epochs",
                current_value=str(current.get("num_epochs", "4")),
                recommended_value="2",
                reason="Fewer epochs for large datasets (enough data per epoch)",
                priority="medium"
            ),
        ]

    def _language_imbalance_rules(self, current: dict, pl_ratio: float) -> list:
        if pl_ratio > 0.8:
            return [TrainingRecommendation(
                parameter="data_augmentation",
                current_value="none",
                recommended_value="add_english_examples",
                reason=f"Dataset is {pl_ratio:.0%} Polish — add English examples for language balance",
                priority="high"
            )]
        else:
            return [TrainingRecommendation(
                parameter="data_augmentation",
                current_value="none",
                recommended_value="add_polish_examples",
                reason=f"Dataset is {1-pl_ratio:.0%} English — add Polish examples for language balance",
                priority="high"
            )]

    def _persona_focus_rules(self, current: dict) -> list:
        return [
            TrainingRecommendation(
                parameter="data_mixing",
                current_value="persona_only",
                recommended_value="70% persona + 30% general knowledge",
                reason="High persona ratio may cause catastrophic forgetting",
                priority="high"
            ),
        ]

    def _knowledge_preservation_rules(self, current: dict) -> list:
        return [
            TrainingRecommendation(
                parameter="general_knowledge_data",
                current_value="none",
                recommended_value="Add 50-100 general knowledge examples",
                reason="Preserve base model knowledge during fine-tuning",
                priority="high"
            ),
            TrainingRecommendation(
                parameter="learning_rate",
                current_value=str(current.get("learning_rate", "8e-5")),
                recommended_value="5e-5",
                reason="Lower LR helps preserve more base knowledge",
                priority="medium"
            ),
        ]

    def generate_report(self, recommendations: list) -> str:
        """Generate a human-readable recommendation report."""
        lines = ["Training Configuration Recommendations", "=" * 50]

        by_priority: dict[str, list[Any]] = {"high": [], "medium": [], "low": []}
        for rec in recommendations:
            by_priority[rec.priority].append(rec)

        for priority in ["high", "medium", "low"]:
            recs = by_priority[priority]
            if recs:
                lines.append(f"\n{priority.upper()} Priority:")
                for rec in recs:
                    lines.append(f"  {rec.parameter}:")
                    lines.append(f"    Current: {rec.current_value}")
                    lines.append(f"    Recommended: {rec.recommended_value}")
                    lines.append(f"    Reason: {rec.reason}")

        return "\n".join(lines)
