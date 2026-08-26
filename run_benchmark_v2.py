#!/usr/bin/env python3
"""Phi-4 14B baseline benchmark — v2 with better prompts."""
import sys
import os
import json
import time
import re
import ast

sys.path.insert(0, "src")
from llama_cpp import Llama

MODEL_PATH = "baselines/phi-4-14b-Q4_K_M.gguf"
N_CTX = 8192
N_GPU_LAYERS = 99
NUM_SAMPLES = 20

print(f"Loading {MODEL_PATH}...")
llm = Llama(model_path=MODEL_PATH, n_ctx=N_CTX, n_gpu_layers=N_GPU_LAYERS, verbose=False)
print("Model loaded!")

results = {}

# ══════════════════════════════════════════════════════════════
# MMLU
# ══════════════════════════════════════════════════════════════
print("\n--- MMLU ---")
from datasets import load_dataset
ds = load_dataset("cais/mmlu", "all", split="test", cache_dir="data/benchmarks")
ds = ds.select(range(min(NUM_SAMPLES, len(ds))))
correct = 0; total = 0
for item in ds:
    question = item["question"]
    choices = item["choices"]
    if isinstance(choices, str):
        choices = ast.literal_eval(choices)
    expected = ["A", "B", "C", "D"][item["answer"]]
    prompt = f"Answer the following multiple choice question. Output ONLY the letter.\n\n{question}\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\n\nLetter:"
    out = llm.create_completion(prompt=prompt, max_tokens=10, temperature=0.0)
    pred = out["choices"][0]["text"].strip()
    pred_letter = ""
    for c in pred:
        if c in "ABCD":
            pred_letter = c
            break
    if pred_letter == expected: correct += 1
    total += 1
results["mmlu"] = {"total": total, "correct": correct, "accuracy": round(correct/max(total,1)*100,1)}
print(f"  MMLU: {results['mmlu']['accuracy']}% ({correct}/{total})")

# ══════════════════════════════════════════════════════════════
# HellaSwag
# ══════════════════════════════════════════════════════════════
print("\n--- HellaSwag ---")
ds = load_dataset("Rowan/hellaswag", split="validation", cache_dir="data/benchmarks")
ds = ds.select(range(min(NUM_SAMPLES, len(ds))))
correct = 0; total = 0
for item in ds:
    ctx = item["ctx"]
    gold_idx = int(item["label"])
    endings = item["endings"]
    if isinstance(endings, str):
        endings = ast.literal_eval(endings)
    prompt = f"What happens next? Choose A, B, C, or D.\n\n{ctx}\nA) {endings[0]}\nB) {endings[1]}\nC) {endings[2]}\nD) {endings[3]}\n\nAnswer (just the letter):"
    out = llm.create_completion(prompt=prompt, max_tokens=20, temperature=0.0)
    pred = out["choices"][0]["text"].strip()
    pred_letter = ""
    for c in pred:
        if c in "ABCD":
            pred_letter = c
            break
    expected = ["A", "B", "C", "D"][gold_idx]
    if pred_letter == expected: correct += 1
    total += 1
results["hellaswag"] = {"total": total, "correct": correct, "accuracy": round(correct/max(total,1)*100,1)}
print(f"  HellaSwag: {results['hellaswag']['accuracy']}% ({correct}/{total})")

# ══════════════════════════════════════════════════════════════
# ARC Challenge
# ══════════════════════════════════════════════════════════════
print("\n--- ARC Challenge ---")
ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", cache_dir="data/benchmarks")
ds = ds.select(range(min(NUM_SAMPLES, len(ds))))
correct = 0; total = 0
for item in ds:
    question = item["question"]
    choices_dict = item["choices"]
    if isinstance(choices_dict, str):
        choices_dict = ast.literal_eval(choices_dict)
    labels = choices_dict["label"]
    texts = choices_dict["text"]
    answer = item["answerKey"]
    choice_text = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
    prompt = f"Answer the question with just the letter.\n\n{question}\n{choice_text}\n\nLetter:"
    out = llm.create_completion(prompt=prompt, max_tokens=10, temperature=0.0)
    pred = out["choices"][0]["text"].strip()
    pred_letter = ""
    for c in pred:
        if c in "ABCDE":
            pred_letter = c
            break
    if pred_letter == answer: correct += 1
    total += 1
results["arc_challenge"] = {"total": total, "correct": correct, "accuracy": round(correct/max(total,1)*100,1)}
print(f"  ARC Challenge: {results['arc_challenge']['accuracy']}% ({correct}/{total})")

# ══════════════════════════════════════════════════════════════
# TruthfulQA
# ══════════════════════════════════════════════════════════════
print("\n--- TruthfulQA ---")
ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation", cache_dir="data/benchmarks")
ds = ds.select(range(min(NUM_SAMPLES, len(ds))))
correct = 0; total = 0
for item in ds:
    question = item["question"]
    correct_answers = item["correct_answers"]
    incorrect_answers = item["incorrect_answers"]
    prompt = f"Answer truthfully and concisely.\n\nQ: {question}\nA:"
    out = llm.create_completion(prompt=prompt, max_tokens=100, temperature=0.0)
    pred = out["choices"][0]["text"].strip().lower()
    matches_correct = any(ca.lower() in pred for ca in correct_answers)
    matches_incorrect = any(ia.lower() in pred for ia in incorrect_answers)
    if matches_correct and not matches_incorrect: correct += 1
    total += 1
results["truthfulqa"] = {"total": total, "correct": correct, "accuracy": round(correct/max(total,1)*100,1)}
print(f"  TruthfulQA: {results['truthfulqa']['accuracy']}% ({correct}/{total})")

# ══════════════════════════════════════════════════════════════
# GSM8K
# ══════════════════════════════════════════════════════════════
print("\n--- GSM8K ---")
ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir="data/benchmarks")
ds = ds.select(range(min(NUM_SAMPLES, len(ds))))
correct = 0; total = 0
for item in ds:
    question = item["question"]
    expected = item["answer"].split("####")[-1].strip().replace(",", "")
    prompt = f"Solve step by step. After #### give ONLY the final number.\n\n{question}\n\n####"
    out = llm.create_completion(prompt=prompt, max_tokens=256, temperature=0.0)
    pred = out["choices"][0]["text"].strip()
    pred_num = ""
    if "####" in pred:
        pred_num = pred.split("####")[-1].strip().replace(",", "")
    else:
        nums = re.findall(r"[-+]?\d+\.?\d*", pred)
        if nums: pred_num = nums[-1]
    if pred_num == expected: correct += 1
    total += 1
results["gsm8k"] = {"total": total, "correct": correct, "accuracy": round(correct/max(total,1)*100,1)}
print(f"  GSM8K: {results['gsm8k']['accuracy']}% ({correct}/{total})")

# ══════════════════════════════════════════════════════════════
# Winogrande
# ══════════════════════════════════════════════════════════════
print("\n--- Winogrande ---")
ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation", cache_dir="data/benchmarks")
ds = ds.select(range(min(NUM_SAMPLES, len(ds))))
correct = 0; total = 0
for item in ds:
    sentence = item["sentence"]
    answer = item["answer"]
    option1 = item["option1"]
    option2 = item["option2"]
    prompt = f"Choose 1 or 2: {sentence.replace('_', '______')}\n1: {option1}\n2: {option2}\nNumber:"
    out = llm.create_completion(prompt=prompt, max_tokens=10, temperature=0.0)
    pred = out["choices"][0]["text"].strip()
    pred_num = "1" if "1" in pred[:5] else "2" if "2" in pred[:5] else ""
    if pred_num == answer: correct += 1
    total += 1
results["winogrande"] = {"total": total, "correct": correct, "accuracy": round(correct/max(total,1)*100,1)}
print(f"  Winogrande: {results['winogrande']['accuracy']}% ({correct}/{total})")

# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
total_correct = sum(r["correct"] for r in results.values())
total_questions = sum(r["total"] for r in results.values())
overall = round(total_correct / max(total_questions, 1) * 100, 1)

print(f"\n{'='*60}")
print(f"PHI-4 14B BASELINE RESULTS (v2)")
print(f"{'='*60}")
for name, r in results.items():
    print(f"  {name}: {r['accuracy']}% ({r['correct']}/{r['total']})")
print(f"\nOverall: {total_correct}/{total_questions} = {overall}%")

with open("baselines/phi4_baseline_v2.json", "w") as f:
    json.dump({"model": "Phi-4 14B Q4_K_M", "results": results, "overall": overall}, f, indent=2)
print(f"\nResults saved to baselines/phi4_baseline_v2.json")
