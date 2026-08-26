#!/usr/bin/env python3
"""Debug benchmark evaluation."""
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

bench = suite.benchmarks["mmlu_sample"]
for i, sample in enumerate(bench.get_samples(3)):
    prompt = bench.format_prompt(sample)
    result = engine.generate([{"role": "user", "content": prompt}])
    pred = result["response"]
    correct = bench.evaluate(sample, pred)
    expected = sample["answer"]
    print(f"Q: {sample['question'][:60]}")
    print(f"A: {pred[:100]}")
    print(f"Expected: {expected}")
    print(f"Correct: {correct}")
    print()
