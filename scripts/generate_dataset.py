#!/usr/bin/env python3
"""Generate a synthetic training dataset for Finetune Studio."""
import json
import random
import argparse
from pathlib import Path

# Diverse prompt templates covering multiple capabilities
TEMPLATES = [
    # Knowledge & explanation
    ("What is {topic}?", "{topic} is {explanation}"),
    ("Explain {topic} in simple terms.", "{topic} can be understood as {explanation}"),
    ("How does {topic} work?", "The way {topic} works is that {explanation}"),
    ("Why is {topic} important?", "{topic} matters because {explanation}"),
    ("What are the key concepts of {topic}?", "The key concepts of {topic} include {explanation}"),
    # Code & technical
    ("Write a Python function to {task}", "```python\ndef {function_name}():\n    {code_body}\n```"),
    ("How do I {task} in Python?", "You can {task} in Python like this:\n```python\n{code_body}\n```"),
    ("What's the best way to {task}?", "The best approach to {task} is {explanation}"),
    # Reasoning & analysis
    ("What are the pros and cons of {topic}?", "The pros of {topic} are advantages like benefits, while the cons include drawbacks and limitations."),
    ("Compare {topic} with alternatives", "{topic} differs from alternatives in that {explanation}"),
    # Creative & instructional
    ("Teach me about {topic}", "Let me teach you about {topic}. {explanation}"),
    ("Give me an example of {topic}", "Here's an example of {topic}: {explanation}"),
    ("What should I know about {topic}?", "When it comes to {topic}, you should know that {explanation}"),
    # Problem solving
    ("How can I solve {topic}?", "To solve {topic}, you can {explanation}"),
    ("What's the solution for {topic}?", "A solution for {topic} is to {explanation}"),
]

TOPICS = [
    ("machine learning", "algorithms that learn patterns from data without being explicitly programmed"),
    ("neural networks", "computing systems inspired by biological neural networks that learn from data"),
    ("fine-tuning", "adapting a pre-trained model to a specific task with targeted training data"),
    ("LoRA", "a parameter-efficient training method that adds small adapter matrices instead of updating all weights"),
    ("RAG", "retrieval-augmented generation, a technique that grounds LLM responses in external knowledge"),
    ("vector databases", "databases optimized for storing and querying high-dimensional embedding vectors"),
    ("embeddings", "numerical representations of text that capture semantic meaning in a dense vector space"),
    ("transformers", "neural network architecture based on self-attention that powers modern LLMs"),
    ("tokenization", "the process of breaking text into smaller units (tokens) that models can process"),
    ("attention mechanisms", "components that let models focus on relevant parts of the input when generating output"),
    ("CUDA", "NVIDIA's parallel computing platform that enables GPU-accelerated deep learning"),
    ("quantization", "techniques to reduce model size and speed up inference by using lower-precision numbers"),
    ("Python programming", "a high-level programming language widely used for machine learning and data science"),
    ("data preprocessing", "the steps to clean, normalize, and prepare raw data for model training"),
    ("model evaluation", "the process of measuring model performance using benchmarks and metrics"),
]

TASKS = [
    ("sort a list", "sort", "return sorted(data)"),
    ("read a file", "read_file", "with open(path) as f:\n    return f.read()"),
    ("process a CSV", "process_csv", "import csv\nwith open(f) as f:\n    return list(csv.reader(f))"),
    ("make an HTTP request", "fetch", "import requests\nreturn requests.get(url).json()"),
    ("parse JSON", "parse", "import json\nreturn json.loads(data)"),
    ("filter a list", "filter", "[x for x in items if condition(x)]"),
    ("handle errors", "safe_call", "try:\n    return operation()\nexcept Exception as e:\n    return None"),
]

def generate_dataset(n: int, output_path: str, seed: int = 42) -> int:
    """Generate n unique training examples."""
    random.seed(seed)
    data = []
    used = set()
    
    for i in range(n):
        template_idx = random.randint(0, len(TEMPLATES) - 1)
        prompt_template, response_template = TEMPLATES[template_idx]
        
        topic, explanation = random.choice(TOPICS)
        task, function_name, code_body = random.choice(TASKS)
        
        prompt = prompt_template.format(topic=topic, task=task)
        # Avoid exact duplicates
        if prompt in used:
            prompt = prompt + f" (variant {i})"
        used.add(prompt)
        
        response = response_template.format(
            topic=topic,
            explanation=explanation,
            task=task,
            function_name=function_name,
            code_body=code_body,
        )
        
        data.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        })
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    return len(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic training data for Finetune Studio")
    parser.add_argument("-n", "--num", type=int, default=500, help="Number of examples (default: 500)")
    parser.add_argument("-o", "--output", default="data/synthetic_train.jsonl", help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    count = generate_dataset(args.num, args.output, args.seed)
    print(f"Generated {count} training examples → {args.output}")
