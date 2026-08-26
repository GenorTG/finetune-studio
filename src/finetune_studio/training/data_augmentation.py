"""Generate variations of training data.

WHAT THIS FILE DOES
==================
Augments training data to improve model robustness:
  - Paraphrase questions (ask the same thing in different ways)
  - Generate variations of answers (different phrasings)
  - Add synonyms and alternative word choices
  - Create typo versions (helps robustness without hurting accuracy)

KEY CONCEPTS
============
- Data augmentation: a technique from computer vision applied to
  NLP. Show the model the same concept in multiple ways.
- Tradeoffs: too much augmentation can dilute the training signal.
  We augment at a 1:1 or 1:2 ratio (original:augmented).
- Use cases: helps with overfitting (model memorizes training data)
  and with generalization (model handles real-world variations).
"""

"""Training data augmentation — generate data to fix weak areas."""
import random


class DataAugmenter:
    """Generate training data to fix specific weaknesses."""

    def __init__(self):
        self.generators = {
            "knowledge": self.generate_knowledge_data,
            "refusal": self.generate_refusal_data,
            "language_balance": self.generate_language_balanced_data,
            "hallucination_guard": self.generate_hallucination_guard,
            "persona_preservation": self.generate_persona_preservation,
        }

    def generate_knowledge_data(self, count: int = 100) -> list:
        """Generate general knowledge Q&A pairs.

        These preserve base model knowledge during fine-tuning.
        """
        templates = [
            # Science
            ("What is photosynthesis?", "Photosynthesis is the process by which plants convert sunlight, water, and CO2 into glucose and oxygen."),
            ("What is DNA?", "DNA is the molecule that carries genetic information in living organisms."),
            ("What is the speed of light?", "The speed of light in vacuum is approximately 299,792,458 m/s."),
            ("What is gravity?", "Gravity is the force of attraction between objects with mass."),
            ("What is evolution?", "Evolution is the change in heritable characteristics of biological populations over successive generations."),
            # Math
            ("What is the Pythagorean theorem?", "In a right triangle, a² + b² = c², where c is the hypotenuse."),
            ("What is a prime number?", "A prime number is a natural number greater than 1 with no divisors other than 1 and itself."),
            ("What is pi?", "Pi (π) is the ratio of a circle's circumference to its diameter, approximately 3.14159."),
            ("What is a derivative?", "A derivative measures the rate of change of a function with respect to its variable."),
            # Technology
            ("What is cloud computing?", "Cloud computing delivers computing services over the internet."),
            ("What is machine learning?", "Machine learning is a subset of AI where systems learn from data without explicit programming."),
            ("What is an API?", "An API is a set of protocols for building software applications."),
            ("What is Docker?", "Docker is a platform for developing, shipping, and running applications in containers."),
            ("What is Kubernetes?", "Kubernetes is an open-source container orchestration platform."),
            # History
            ("When did World War II end?", "World War II ended in 1945."),
            ("Who invented the telephone?", "Alexander Graham Bell invented the telephone in 1876."),
            ("What was the Renaissance?", "The Renaissance was a cultural movement in Europe from the 14th to 17th century."),
            # Geography
            ("What is the largest ocean?", "The Pacific Ocean is the largest and deepest ocean."),
            ("What is the capital of France?", "Paris is the capital of France."),
            ("What is the longest river?", "The Nile River is traditionally considered the longest at about 6,650 km."),
        ]

        examples = []
        for _ in range(count):
            q, a = random.choice(templates)
            examples.append({
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ]
            })
        return examples

    def generate_refusal_data(self, count: int = 50) -> list:
        """Generate appropriate refusal examples.

        These teach the model WHEN to refuse (not everything).
        """
        templates = [
            ("What's the weather today?", "I don't have access to real-time weather data. Check a weather service for current conditions."),
            ("What will the stock market do tomorrow?", "I can't predict stock market movements. That requires real-time financial data."),
            ("What's my credit score?", "I don't have access to your financial information. Check with your bank or credit bureau."),
            ("What did I have for breakfast?", "I don't have access to your personal daily activities."),
            ("How do I hack a computer?", "I can't help with unauthorized access to computer systems. That's illegal."),
            ("What's the meaning of life?", "That's a philosophical question with no single answer. Different people find meaning in different ways."),
            ("Who will win the election?", "I can't predict election outcomes. That depends on many complex factors."),
            ("What's behind Area 51?", "Area 51 is a classified US Air Force facility. I don't have access to classified information."),
            ("What's the future of AI?", "AI is rapidly evolving. I can discuss current trends but can't predict the future with certainty."),
            ("What's the best programming language?", "There's no single 'best' language — it depends on the use case. Python for ML, JavaScript for web, Rust for systems programming."),
        ]

        examples = []
        for _ in range(count):
            q, a = random.choice(templates)
            examples.append({
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ]
            })
        return examples

    def generate_language_balanced_data(self, pl_examples: list, target_en: int = 100) -> list:
        """Generate English versions of Polish examples.

        Ensures balanced language training.
        """
        # Simple translation templates
        translations = {
            "Czym jest": "What is",
            "Jak działa": "How does",
            "Dlaczego": "Why",
            "Kiedy": "When",
            "Gdzie": "Where",
            "Kto": "Who",
            "Ile": "How many/much",
            "Czy": "Is/Do",
            "Opowiedz o": "Tell me about",
        }

        en_examples = []
        for item in pl_examples[:target_en]:
            msgs = item.get("messages", [])
            if not msgs:
                continue

            user_msg = msgs[0].get("content", "")

            # Try simple translation
            en_user = user_msg
            for pl, en in translations.items():
                if pl in en_user:
                    en_user = en_user.replace(pl, en)
                    break

            # Keep the assistant response (assume bilingual model can handle)
            if len(msgs) > 1:
                en_examples.append({
                    "messages": [
                        {"role": "user", "content": en_user},
                        {"role": "assistant", "content": msgs[1].get("content", "")}
                    ]
                })

        return en_examples

    def generate_hallucination_guard(self, count: int = 50) -> list:
        """Generate examples that teach the model to avoid hallucinations."""
        templates = [
            ("When was X invented?", "I don't have specific information about when X was invented. You might want to check a reliable source."),
            ("What does the research say about Y?", "I can discuss general concepts, but I don't have access to specific research papers. Check academic databases for current research."),
            ("What's the exact number of Z?", "I don't have precise data on that. For accurate numbers, check official sources or recent reports."),
            ("Tell me about person A's personal life.", "I respect people's privacy and don't share personal information about individuals."),
            ("What happened on this specific date?", "I don't have detailed event logs for specific dates. Check news archives for historical events."),
        ]

        examples = []
        for _ in range(count):
            q, a = random.choice(templates)
            examples.append({
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ]
            })
        return examples

    def generate_persona_preservation(self, persona_data: list, count: int = 50) -> list:
        """Generate examples that reinforce persona without overwriting knowledge."""
        examples = []
        for item in persona_data[:count]:
            msgs = item.get("messages", [])
            if msgs and msgs[0].get("role") == "user":
                # Add a generic knowledge question after persona questions
                examples.append(item)

        return examples

    def augment_dataset(self, data: list, weaknesses: list | None = None) -> list:
        """Augment dataset based on detected weaknesses."""
        if weaknesses is None:
            weaknesses = ["knowledge", "refusal"]

        augmented = list(data)

        for weakness in weaknesses:
            if weakness in self.generators:
                new_data = self.generators[weakness](count=min(50, len(data)))
                augmented.extend(new_data)

        random.shuffle(augmented)
        return augmented
