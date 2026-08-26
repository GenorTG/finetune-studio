#!/usr/bin/env python3
"""Debug: check what the model actually outputs for benchmarks."""
import sys
sys.path.insert(0, "src")

from llama_cpp import Llama
from finetune_studio.benchmarks import BenchmarkSuite

llm = Llama(model_path="/home/genortg/inference-server/models/chris-ai-v20.Q4_K_M.gguf", n_ctx=8192, n_gpu_layers=99, verbose=False)
with open("/home/genortg/inference-server/sysprompt_v20.txt") as f:
    SYSPROMPT = f.read()

class MockEngine:
    def generate(self, messages, max_tokens=256, temperature=0.0):
        user_msg = messages[0]["content"]
        prompt = f"<start_of_turn>user\n{SYSPROMPT}\n\n{user_msg}<end_of_turn>\n<start_of_turn>model\n"
        out = llm.create_completion(prompt=prompt, max_tokens=max_tokens, temperature=temperature, stop=["<end_of_turn>"])
        return {"response": out["choices"][0]["text"].strip()}

engine = MockEngine()
suite = BenchmarkSuite()

sep = "=" * 60

for bench_name in ["mmlu_sample", "hellaswag_sample", "gsm8k_sample", "humaneval_sample", "truthfulqa_sample"]:
    bench = suite.benchmarks[bench_name]
    print(f"\n{sep}")
    print(f"{bench_name}")
    print(sep)
    for i, sample in enumerate(bench.get_samples()):
        prompt = bench.format_prompt(sample)
        result = engine.generate([{"role": "user", "content": prompt}])
        pred = result["response"]
        correct = bench.evaluate(sample, pred)
        expected = bench.get_expected(sample)
        q = sample.get("question", "")[:80]
        print(f"Q{i+1}: {q}")
        print(f"  Model: {pred[:200]}")
        print(f"  Expected: {expected[:100]}")
        print(f"  Correct: {correct}")
        print()
