#!/usr/bin/env python3
"""Robust benchmark runner for Phi-4 14B."""
import sys
import os
import json
import time

sys.path.insert(0, "src")

from llama_cpp import Llama

MODEL_PATH = "baselines/phi-4-14b-Q4_K_M.gguf"
N_CTX = 8192
N_GPU_LAYERS = 99
NUM_SAMPLES = 20

print(f"Loading {MODEL_PATH}...")
llm = Llama(model_path=MODEL_PATH, n_ctx=N_CTX, n_gpu_layers=N_GPU_LAYRS, verbose=False)
print("Model loaded!")

results = {}

# ══════════════════════════════════════════════════════════════
# MMLU
# ══════════════════════════════════════════════════════════════
print("\n--- MMLU ---")
from datasets import load_dataset
import ast

ds = load_dataset("cais/mmlu", "all", split="test", cache_dir="data/benchmarks")
ds = ds.select(range(min(NUM_SAMPLES, len(ds))))

correct = 0
total = 0
for item in ds:
    question = item["question"]
    answer_idx = item["answer"]
    expected = ["A", "B", "C", "D"][answer_idx]

    choices = item["choices"]
    if isinstance(choices, str):
        choices = ast.literal_eval(choices)

    prompt = f"Answer the following multiple choice question. Reply with ONLY the letter (A, B, C, or D).\n\nQuestion: {question}\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\n\nAnswer:"
    out = llm.create_completion(prompt=prompt, max_tokens=10, temperature=0.0)
    pred = out["choices"][0]["text"].strip().upper()
    pred_letter = pred[0] if pred and pred[0] in "ABCD" else ""

    if pred_letter == expected:
        correct += 1
    total += 1

acc = round(correct / max(total, 1) * 100, 1)
results["mmlu"] = {"total": total, "correct": correct, "accuracy": acc}
print(f"  MMLU: {acc}% ({correct}/{total})")

# ══════════════════════════════════════════════════════════════
# HellaSwag
# ══════════════════════════════════════════════════════════════
print("\n--- HellaSwag ---")
ds = load_dataset("Rowan/hellaswag", split="validation", cache_dir="data/benchmarks")
ds = ds.select(range(min(NUM_SAMPLES, len(ds))))

correct = 0
total = 0
for item in ds:
    ctx = item["ctx"]
    gold_idx = int(item["label"])
    endings = item["endings"]
    if isinstance(endings, str):
        endings = ast.literal_eval(endings)

    prompt = f"Complete the following scenario. Reply with ONLY the letter (A, B, C, or D).\n\nContext: {ctx}\nA) {endings[0]}\nB) {endings[1]}\nC) {endings[2]}\nD) {endings[3]}\n\nWhat happens next?:"
    out = llm.create_completion(prompt=prompt, max_tokens=10, temperature=0.0)
    pred = out["choices"][0]["text"].strip().upper()
    pred_letter = pred[0] if pred and pred[0] in "ABCD" else ""
    expected = ["A", "B", "C", "D"][gold_idx]

    if pred_letter == expected:
        correct += 1
    total += 1

acc = round(correct / max(total, 1) * 100, 1)
results["hellaswag"] = {"total": total, "correct": correct, "accuracy": acc}
print(f"  HellaSwag: {acc}% ({correct}/{total})")

# ══════════════════════════════════════════════════════════════
# ARC Challenge
# ══════════════════════════════════════════════════════════════
print("\n--- ARC Challenge ---")
ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", cache_dir="data/benchmarks")
ds = ds.select(range(min(NUM_SAMPLES, len(ds))))

correct = 0
total = 0
for item in ds:
    question = item["question"]
    choices_dict = item["choices"]
    if isinstance(choices_dict, str):
        choices_dict = ast.literal_eval(choices_dict)
    labels = choices_dict["label"]
    texts = choices_dict["text"]
    answer = item["answerKey"]

    choice_text = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
    prompt = f"Answer the following science question. Reply with ONLY the letter.\n\nQuestion: {question}\n{choice_text}\n\nAnswer:"
    out = llm.create_completion(prompt=prompt, max_tokens=10, temperature=0.0)
    pred = out["choices"][0]["text"].strip().upper()
    pred_letter = pred[0] if pred and pred[0] in "ABCDE" else ""

    if pred_letter == answer:
        correct += 1
    total += 1

acc = round(correct / max(total, 1) * 100, 1)
results["arc_challenge"] = {"total": total, "correct": correct, "accuracy": acc}
print(f"  ARC Challenge: {acc}% ({correct}/{total})")

# ══════════════════════════════════════════════════════════════
# TruthfulQA
# ══════════════════════════════════════════════════════════════
print("\n--- TruthfulQA ---")
ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation", cache_dir="data/benchmarks")
ds = ds.select(range(min(NUM_SAMPLES, len(ds))))

correct = 0
total = 0
for item in ds:
    question = item["question"]
    correct_answers = item["correct_answers"]
    incorrect_answers = item["incorrect_answers"]

    prompt = f"Answer the following question truthfully. Be concise.\n\nQuestion: {question}\n\nAnswer:"
    out = llm.create_completion(prompt=prompt, max_tokens=100, temperature=0.0)
    pred = out["choices"][0]["text"].strip().lower()

    matches_correct = any(ca.lower() in pred for ca in correct_answers)
    matches_incorrect = any(ia.lower() in pred for ia in incorrect_answers)
    is_truthful = matches_correct and not matches_incorrect

    if is_truthful:
        correct += 1
    total += 1

acc = round(correct / max(total, 1) * 100, 1)
results["truthfulqa"] = {"total": total, "correct": correct, "accuracy": acc}
print(f"  TruthfulQA: {acc}% ({correct}/{total})")

# ══════════════════════════════════════════════════════════════
# GSM8K
# ══════════════════════════════════════════════════════════════
print("\n--- GSM8K ---")
ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir="data/benchmarks")
ds = ds.select(range(min(NUM_SAMPLES, len(ds))))

import re
correct = 0
total = 0
for item in ds:
    question = item["question"]
    answer_str = item["answer"]
    expected = answer_str.split("####")[-1].strip().replace(",", "")

    prompt = f"Solve this math problem step by step. Give ONLY the final numeric answer after \"####\".\n\nQuestion: {question}\n\nSolution:"
    out = llm.create_completion(prompt=prompt, max_tokens=256, temperature=0.0)
    pred = out["choices"][0]["text"].strip()

    pred_num = ""
    if "####" in pred:
        pred_num = pred.split("####")[-1].strip().replace(",", "")
    else:
        nums = re.findall(r"[-+]?\d+\.?\d*", pred)
        if nums:
            pred_num = nums[-1]

    if pred_num == expected:
        correct += 1
    total += 1

acc = round(correct / max(total, 1) * 100, 1)
results["gsm8k"] = {"total": total, "correct": correct, "accuracy": acc}
print(f"  GSM8K: {acc}% ({correct}/{total})")

# ══════════════════════════════════════════════════════════════
# Winogrande
# ══════════════════════════════════════════════════════════════
print("\n--- Winogrande ---")
ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation", cache_dir="data/benchmarks")
ds = ds.select(range(min(NUM_SAMPLES, len(ds))))

correct = 0
total = 0
for item in ds:
    sentence = item["sentence"]
    answer = item["answer"]
    option1 = item["option1"]
    option2 = item["option2"]

    prompt = f"Complete the sentence by choosing the correct option (1 or 2).\n\nSentence: {sentence.replace('_', '______')}\nOption 1: {option1}\nOption 2: {option2}\n\nWhich option fits better? Reply with ONLY the number (1 or 2):"
    out = llm.create_completion(prompt=prompt, max_tokens=10, temperature=0.0)
    pred = out["choices"][0]["text"].strip()
    pred_num = "1" if "1" in pred[:3] else "2" if "2" in pred[:3] else ""

    if pred_num == answer:
        correct += 1
    total += 1

acc = round(correct / max(total, 1) * 100, 1)
results["winogrande"] = {"total": total, "correct": correct, "accuracy": acc}
print(f"  Winogrande: {acc}% ({correct}/{total})")

# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
total_correct = sum(r["correct"] for r in results.values())
total_questions = sum(r["total"] for r in results.values())
overall = round(total_correct / max(total_questions, 1) * 100, 1)

print(f"\n{'='*60}")
print(f"PHI-4 14B BASELINE RESULTS")
print(f"{'='*60}")
for name, r in results.items():
    print(f"  {name}: {r['accuracy']}% ({r['correct']}/{r['total']})")
print(f"\nOverall: {total_correct}/{total_questions} = {overall}%")

# Save results
with open("baselines/phi4_baseline_results.json", "w") as f:
    json.dump({"model": "Phi-4 14B Q4_K_M", "results": results, "overall": overall}, f, indent=2)
print(f"\nResults saved to baselines/phi4_baseline_results.json")
