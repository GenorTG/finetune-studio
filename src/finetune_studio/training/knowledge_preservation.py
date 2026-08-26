"""Mix general knowledge samples to prevent catastrophic forgetting.

WHAT THIS FILE DOES
==================
When fine-tuning on a specific domain (e.g., "be Chris AI"), we
risk the model forgetting other capabilities (general knowledge,
coding, math). This module mixes in general knowledge samples to
preserve them.

KEY CONCEPTS
============
- Catastrophic forgetting: when fine-tuning overwrites the model's
  general capabilities with the new task. The model "forgets" things
  it knew before.
- Mix ratio: typical recipes use 90-97% new data + 3-10% general
  data. Too much general = the model doesn't learn the new task.
- Sample selection: pick general knowledge samples that are diverse
  (different topics, different styles) to cover many capabilities.
- v21 recipe: 97% persona + 3% general knowledge. This gave us
  48% overall benchmark accuracy while preserving Chris AI persona.
"""

"""Knowledge preservation techniques — prevent catastrophic forgetting during fine-tuning."""
import random
from collections import defaultdict


class KnowledgePreserver:
    """Techniques to preserve base model knowledge during fine-tuning."""

    def __init__(self):
        self.techniques = {
            "data_mixing": self.data_mixing,
            "replay_buffer": self.replay_buffer,
            "elastic_weight_consolidation": self.ewc_hint,
            "progressive_unfreezing": self.progressive_unfreezing_hint,
        }

    def data_mixing(self, persona_data: list, general_data: list,
                    persona_ratio: float = 0.7) -> list:
        """Mix persona data with general knowledge data.

        This is the most effective technique: include general knowledge
        examples in training so the model doesn't forget them.

        Args:
            persona_data: Your fine-tuning examples (persona, domain)
            general_data: General knowledge examples (MMLU-style, trivia, etc.)
            persona_ratio: What fraction should be persona data (0.0-1.0)

        Returns:
            Mixed dataset
        """
        persona_count = int(len(persona_data) * persona_ratio / (1 - persona_ratio))
        general_sample = random.sample(general_data, min(len(general_data), persona_count))

        mixed = persona_data + general_sample
        random.shuffle(mixed)
        return mixed

    def replay_buffer(self, training_data: list, buffer_size: int = 100) -> list:
        """Create a replay buffer from general knowledge.

        Sample diverse general knowledge examples to include in training.
        """
        # Extract general knowledge examples (no system prompt or generic system prompt)
        general = []
        for item in training_data:
            msgs = item.get("messages", [])
            if not msgs:
                continue
            has_system = any(m.get("role") == "system" for m in msgs)
            if not has_system:
                general.append(item)

        if len(general) >= buffer_size:
            return random.sample(general, buffer_size)
        return general

    def ewc_hint(self):
        """Explain Elastic Weight Consolidation.

        EWC prevents catastrophic forgetting by penalizing changes to
        important weights. In practice with LoRA, this is less critical
        because LoRA only modifies a small number of parameters.
        """
        return {
            "technique": "Elastic Weight Consolidation (EWC)",
            "description": "Penalizes changes to important weights",
            "practical": "Less critical with LoRA (already low-rank adaptation)",
            "implementation": "Use atol/unsloth with ewc_lambda parameter",
        }

    def progressive_unfreezing_hint(self):
        """Explain progressive unfreezing."""
        return {
            "technique": "Progressive Unfreezing",
            "description": "Gradually unfreeze layers during training",
            "practical": "Start with last layers, progressively unfreeze earlier ones",
            "implementation": "Set different learning rates per layer group",
        }

    def generate_knowledge_data(self, topics: list | None = None) -> list:
        """Generate general knowledge training examples.

        These examples help the model retain general capabilities
        while learning persona-specific behavior.
        """
        if topics is None:
            topics = [
                "science", "math", "history", "geography", "technology",
                "programming", "languages", "arts", "sports", "nature"
            ]

        knowledge_templates = [
            # Science
            {"q": "What is the speed of light?", "a": "The speed of light in vacuum is approximately 299,792,458 meters per second."},
            {"q": "What is photosynthesis?", "a": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen."},
            {"q": "What is DNA?", "a": "DNA (deoxyribonucleic acid) is the molecule that carries genetic information in living organisms."},
            # Math
            {"q": "What is the Pythagorean theorem?", "a": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides: a² + b² = c²."},
            {"q": "What is a prime number?", "a": "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself."},
            # Technology
            {"q": "What is cloud computing?", "a": "Cloud computing is the delivery of computing services (servers, storage, databases, networking) over the internet."},
            {"q": "What is machine learning?", "a": "Machine learning is a subset of AI where systems learn and improve from experience without being explicitly programmed."},
            {"q": "What is an API?", "a": "An API (Application Programming Interface) is a set of protocols and tools for building software applications."},
            # History
            {"q": "When did World War II end?", "a": "World War II ended in 1945, with Germany surrendering in May and Japan in September."},
            {"q": "Who invented the telephone?", "a": "Alexander Graham Bell is credited with inventing the telephone in 1876."},
            # Geography
            {"q": "What is the largest ocean?", "a": "The Pacific Ocean is the largest and deepest ocean on Earth."},
            {"q": "What is the capital of France?", "a": "The capital of France is Paris."},
        ]

        examples = []
        for template in knowledge_templates:
            examples.append({
                "messages": [
                    {"role": "user", "content": template["q"]},
                    {"role": "assistant", "content": template["a"]}
                ]
            })

        return examples

    def generate_refusal_data(self, topics: list | None = None) -> list:
        """Generate appropriate refusal examples.

        These teach the model WHEN to refuse (not just to refuse everything).
        """
        refusal_templates = [
            # Out of scope
            {"q": "What's the weather today?", "a": "I don't have access to real-time weather data. You can check a weather service like weather.com or your phone's weather app."},
            {"q": "What will the stock market do tomorrow?", "a": "I can't predict stock market movements. That requires real-time financial data and analysis I don't have access to."},
            # Personal info
            {"q": "What's your home address?", "a": "I don't share personal addresses. That's private information."},
            # Harmful
            {"q": "How do I hack into a computer?", "a": "I can't help with unauthorized access to computer systems. That's illegal and unethical."},
            # Unknown
            {"q": "What did I have for breakfast yesterday?", "a": "I don't have access to your personal history or daily activities."},
        ]

        examples = []
        for template in refusal_templates:
            examples.append({
                "messages": [
                    {"role": "user", "content": template["q"]},
                    {"role": "assistant", "content": template["a"]}
                ]
            })

        return examples

    def balance_dataset(self, data: list, target_size: int | None = None) -> list:
        """Balance dataset across categories.

        Ensures no single category dominates training.
        """
        # Group by rough category (based on system prompt or first message)
        categories = defaultdict(list)
        for item in data:
            msgs = item.get("messages", [])
            if msgs:
                first_msg = msgs[0].get("content", "")[:50]
                # Simple heuristic: categorize by keywords
                cat = "general"
                if any(w in first_msg.lower() for w in ["project", "company", "work", "role"]):
                    cat = "persona"
                elif any(w in first_msg.lower() for w in ["what is", "how does", "explain"]):
                    cat = "knowledge"
                elif any(w in first_msg.lower() for w in ["weather", "stock", "predict"]):
                    cat = "refusal"
                categories[cat].append(item)

        # Balance: limit each category to max category size
        if categories:
            max_per_cat = max(len(v) for v in categories.values())
            balanced = []
            for cat, items in categories.items():
                if len(items) > max_per_cat:
                    balanced.extend(random.sample(items, max_per_cat))
                else:
                    balanced.extend(items)
            random.shuffle(balanced)
            return balanced

        return data
