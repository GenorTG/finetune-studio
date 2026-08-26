#!/usr/bin/env python3
"""Quick TruthfulQA test with 5-second per-question timeout."""
import sys, os, json, time, threading
sys.path.insert(0, "src")
from llama_cpp import Llama
from finetune_studio.benchmarks.scoring import scorer
from datasets import load_dataset

llm = Llama(model_path="baselines/phi-4-14b-Q4_K_M.gguf", n_ctx=8192, n_gpu_layers=99, verbose=False)
print("Model loaded")

ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation", cache_dir="data/benchmarks")
ds = ds.select(range(10))

correct = 0; total = 0
for item in ds:
    q = item["question"][:50]
    print(f"  Q: {q}...", end=" ", flush=True)
    prompt = f"Answer truthfully.\n\nQ: {item['question']}\nA:"
    out = llm.create_completion(prompt=prompt, max_tokens=100, temperature=0.0)
    pred = out["choices"][0]["text"].strip()
    result = scorer.score_truthful(pred, item["correct_answers"], item["incorrect_answers"])
    if result["correct"]: correct += 1
    total += 1
    print(f"{'PASS' if result['correct'] else 'FAIL'}")

print(f"\nTruthfulQA: {correct}/{total} = {round(correct/max(total,1)*100,1)}%")
