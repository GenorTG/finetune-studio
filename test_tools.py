#!/usr/bin/env python3
"""Test all new training tools."""
import sys
sys.path.insert(0, "src")

from finetune_studio.training.data_quality import DataQualityAnalyzer
from finetune_studio.training.knowledge_preservation import KnowledgePreserver
from finetune_studio.training.hallucination_guard import HallucinationGuardrail, TrainingDataValidator
from finetune_studio.training.data_augmentation import DataAugmenter
from finetune_studio.training.config_optimizer import TrainingConfigOptimizer
import json

print("=== Testing Training Tools ===\n")

# 1. Data Quality Analyzer
analyzer = DataQualityAnalyzer()
print("1. DataQualityAnalyzer: OK")

# 2. Knowledge Preserver
preserver = KnowledgePreserver()
knowledge = preserver.generate_knowledge_data(10)
print(f"2. KnowledgePreserver: {len(knowledge)} examples")

# 3. Hallucination Guard
guard = HallucinationGuardrail()
result = guard.check_response("What is Python?", "Python is a programming language.")
print(f"3. HallucinationGuard: passed={result.passed}, issues={len(result.issues)}")

result2 = guard.check_response("When was X invented?", "X was invented in 2020 by John Smith.")
print(f"   HallucinationGuard (risky): passed={result2.passed}, issues={len(result2.issues)}")

# 4. Data Augmenter
augmenter = DataAugmenter()
knowledge_data = augmenter.generate_knowledge_data(10)
refusal_data = augmenter.generate_refusal_data(5)
hallucination_data = augmenter.generate_hallucination_guard(5)
print(f"4. DataAugmenter: knowledge={len(knowledge_data)}, refusal={len(refusal_data)}, hallucination={len(hallucination_data)}")

# 5. Config Optimizer
optimizer = TrainingConfigOptimizer()
test_data = [
    {"messages": [{"role": "user", "content": "What is K-Finance?"}, {"role": "assistant", "content": "My financial app."}]}
] * 100
recs = optimizer.analyze_and_recommend(test_data, {"learning_rate": 8e-5, "num_epochs": 4, "lora_rank": 64})
print(f"5. ConfigOptimizer: {len(recs)} recommendations")

# 6. Training Data Validator
validator = TrainingDataValidator()
risky_data = [
    {"messages": [{"role": "user", "content": "test"}, {"role": "assistant", "content": "According to a study, 1.5 million people visit https://example.com"}]}
]
validation = validator.validate_dataset(risky_data)
print(f"6. TrainingDataValidator: {validation['total_risks']} risks found")

# 7. Knowledge Mixing
mixed = preserver.data_mixing(
    [{"messages": [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "I'm Chris."}]}],
    knowledge,
    persona_ratio=0.7
)
print(f"7. KnowledgeMixing: {len(mixed)} examples (70/30 split)")

# 8. Balance Dataset
balanced = preserver.balance_dataset(test_data + knowledge)
print(f"8. BalanceDataset: {len(balanced)} examples")

print("\n=== All Training Tools Working! ===")
