#!/usr/bin/env python3
"""Test v21 persona and knowledge."""
import sys
sys.path.insert(0, "src")

from llama_cpp import Llama

llm = Llama(model_path="/home/genortg/finetune-studio/output_gemma4_v21/export_gguf_gguf/gemma-4-e4b-it.Q4_K_M.gguf", n_ctx=8192, n_gpu_layers=99, verbose=False)
with open("/home/genortg/inference-server/sysprompt_v20.txt") as f:
    SYSPROMPT = f.read()

def ask(q):
    prompt = f"<start_of_turn>user\n{SYSPROMPT}\n\n{q}<end_of_turn>\n<start_of_turn>model\n"
    out = llm.create_completion(prompt=prompt, max_tokens=200, temperature=0.3, stop=["<end_of_turn>"])
    return out["choices"][0]["text"].strip()

print("=== v21 PERSONA TEST ===")
persona_q = [
    ("Who are you?", ["Krzysztof", "Kutniowski"]),
    ("What IDE do you use?", ["Cursor"]),
    ("What is K-Finance?", ["K-Finance", "Android"]),
    ("When were you born?", ["prywatn", "privat", "nie podaj"]),
    ("What is the weather?", ["dont know", "can't", "dunno"]),
]
for q, expected in persona_q:
    r = ask(q)
    match = any(e.lower() in r.lower() for e in expected)
    icon = "PASS" if match else "FAIL"
    print(f"{icon}: {q}")
    print(f"  A: {r[:150]}")
    print()

print("=== v21 KNOWLEDGE TEST ===")
knowledge_q = [
    ("What is the speed of light?", ["299", "300000"]),
    ("What is DNA?", ["genetic", "DNA"]),
    ("What is cloud computing?", ["cloud", "internet"]),
    ("What is the Pythagorean theorem?", ["a2", "b2", "c2", "hypotenuse"]),
]
for q, expected in knowledge_q:
    r = ask(q)
    match = any(e.lower() in r.lower() for e in expected)
    icon = "PASS" if match else "FAIL"
    print(f"{icon}: {q}")
    print(f"  A: {r[:150]}")
    print()
