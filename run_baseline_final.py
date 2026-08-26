#!/usr/bin/env python3
"""Phi-4 14B baseline — with timeout protection."""
import sys
import os
import json
import time
import re
import ast
import signal

sys.path.insert(0, "src")
from llama_cpp import Llama
from finetune_studio.benchmarks.scoring import scorer

MODEL_PATH = "baselines/phi-4-14b-Q4_K_M.gguf"
N_CTX = 8192
N_GPU_LAYERS = 99
NUM_SAMPLES = 20

def timeout_handler(signum, frame):
    raise TimeoutError("Benchmark timed out")

signal.signal(signal.SIGALRM, timeout_handler)

print(f"Loading {MODEL_PATH}...")
llm = Llama(model_path=MODEL_PATH, n_ctx=N_CTX, n_gpu_layers=N_GPU_LAYERS, verbose=False)
print("Model loaded!")

results = {}
from datasets import load_dataset

def run_benchmark(name, ds, eval_fn):
    """Run a single benchmark with timeout."""
    print(f"\n--- {name} ---")
    signal.alarm(600)  # 10 minute timeout
    try:
        correct = 0; total = 0
        for item in ds:
            c, e = eval_fn(item)
            if c: correct += 1
            total += 1
        signal.alarm(0)
        acc = round(correct/max(total,1)*100,1)
        results[name] = {"total": total, "correct": correct, "accuracy": acc}
        print(f"  {acc}% ({correct}/{total})")
    except TimeoutError:
        signal.alarm(0)
        print(f"  TIMEOUT after 10 minutes")
        results[name] = {"total": total, "correct": correct, "accuracy": round(correct/max(total,1)*100,1) if total > 0 else 0}
    except Exception as e:
        signal.alarm(0)
        print(f"  ERROR: {e}")
        results[name] = {"total": total, "correct": correct, "accuracy": 0, "error": str(e)}

# MMLU
ds = load_dataset("cais/mmlu", "all", split="test", cache_dir="data/benchmarks")
ds = ds.select(range(min(NUM_SAMPLES, len(ds))))

def eval_mmlu(item):
    q = item["question"]; c = item["choices"]
    if isinstance(c, str): c = ast.literal_eval(c)
    exp = ["A","B","C","D"][item["answer"]]
    prompt = f"Answer with ONLY the letter.\n\n{q}\nA) {c[0]}\nB) {c[1]}\nC) {c[2]}\nD) {c[3]}\n\nLetter:"
    out = llm.create_completion(prompt=prompt, max_tokens=20, temperature=0.0)
    pred = scorer.extract_mcq_letter(out["choices"][0]["text"], c)
    return pred == exp, exp

run_benchmark("mmlu", ds, eval_mmlu)

# HellaSwag
ds = load_dataset("Rowan/hellaswag", split="validation", cache_dir="data/benchmarks")
ds = ds.select(range(min(NUM_SAMPLES, len(ds))))

def eval_hellaswag(item):
    ctx = item["ctx"]; gold = int(item["label"])
    e = item["endings"]
    if isinstance(e, str): e = ast.literal_eval(e)
    prompt = f"What happens next?\n\n{ctx}\nA) {e[0]}\nB) {e[1]}\nC) {e[2]}\nD) {e[3]}\n\nAnswer:"
    out = llm.create_completion(prompt=prompt, max_tokens=20, temperature=0.0)
    pred = scorer.extract_mcq_letter(out["choices"][0]["text"], e)
    return pred == ["A","B","C","D"][gold], ["A","B","C","D"][gold]

run_benchmark("hellaswag", ds, eval_hellaswag)

# ARC
ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", cache_dir="data/benchmarks")
ds = ds.select(range(min(NUM_SAMPLES, len(ds))))

def eval_arc(item):
    q = item["question"]; cd = item["choices"]
    if isinstance(cd, str): cd = ast.literal_eval(cd)
    l,t = cd["label"], cd["text"]; ans = item["answerKey"]
    ct = "\n".join(f"{x}) {y}" for x,y in zip(l,t))
    prompt = f"Answer with just the letter.\n\n{q}\n{ct}\n\nLetter:"
    out = llm.create_completion(prompt=prompt, max_tokens=20, temperature=0.0)
    pred = scorer.extract_mcq_letter(out["choices"][0]["text"], t)
    return pred == ans, ans

run_benchmark("arc", ds, eval_arc)

# TruthfulQA
ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation", cache_dir="data/benchmarks")
ds = ds.select(range(min(NUM_SAMPLES, len(ds))))

def eval_truthful(item):
    q = item["question"]; ca = item["correct_answers"]; ia = item["incorrect_answers"]
    prompt = f"Answer truthfully.\n\nQ: {q}\nA:"
    out = llm.create_completion(prompt=prompt, max_tokens=100, temperature=0.0)
    pred = out["choices"][0]["text"].strip()
    result = scorer.score_truthful(pred, ca, ia)
    return result["correct"], result

run_benchmark("truthfulqa", ds, eval_truthful)

# GSM8K
ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir="data/benchmarks")
ds = ds.select(range(min(NUM_SAMPLES, len(ds))))

def eval_gsm8k(item):
    q = item["question"]
    exp = item["answer"].split("####")[-1].strip().replace(",","")
    prompt = f"Solve. After #### give ONLY the final number.\n\n{q}\n\n####"
    out = llm.create_completion(prompt=prompt, max_tokens=200, temperature=0.0)
    pred = scorer.extract_math_answer(out["choices"][0]["text"]).replace(",","")
    return pred == exp, exp

run_benchmark("gsm8k", ds, eval_gsm8k)

# Winogrande
ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation", cache_dir="data/benchmarks")
ds = ds.select(range(min(NUM_SAMPLES, len(ds))))

def eval_winogrande(item):
    s = item["sentence"]; a = item["answer"]; o1,o2 = item["option1"],item["option2"]
    prompt = f"Choose 1 or 2:\n{s.replace('_','______')}\n1: {o1}\n2: {o2}\nNumber:"
    out = llm.create_completion(prompt=prompt, max_tokens=10, temperature=0.0)
    pred = scorer.score_winogrande(out["choices"][0]["text"], o1, o2)
    return pred == a, a

run_benchmark("winogrande", ds, eval_winogrande)

# Summary
tc = sum(r["correct"] for r in results.values())
tq = sum(r["total"] for r in results.values())
ov = round(tc / max(tq, 1) * 100, 1)
print(f"\n{'='*60}")
print(f"PHI-4 14B BASELINE (n_ctx={N_CTX})")
print(f"{'='*60}")
for n,r in results.items(): print(f"  {n}: {r['accuracy']}% ({r['correct']}/{r['total']})")
print(f"\nOverall: {tc}/{tq} = {ov}%")
with open("baselines/phi4_baseline_final.json", "w") as f:
    json.dump({"model": "Phi-4 14B Q4_K_M", "n_ctx": N_CTX, "results": results, "overall": ov}, f, indent=2)
print("Saved to baselines/phi4_baseline_final.json")
