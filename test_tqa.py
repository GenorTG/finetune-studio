#!/usr/bin/env python3
from llama_cpp import Llama
from finetune_studio.benchmarks.scoring import scorer
from datasets import load_dataset

llm = Llama(model_path="baselines/phi-4-14b-Q4_K_M.gguf", n_ctx=8192, n_gpu_layers=99, verbose=False)
print("Model loaded")

ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation", cache_dir="data/benchmarks")
ds = ds.select(range(3))

for i, item in enumerate(ds):
    q = item["question"][:60]
    print(f"Q{i}: {q}...")
    prompt = f"Answer truthfully.\n\nQ: {item['question']}\nA:"
    out = llm.create_completion(prompt=prompt, max_tokens=100, temperature=0.0)
    pred = out["choices"][0]["text"].strip()
    print(f"  A: {pred[:100]}")
print("Done")
