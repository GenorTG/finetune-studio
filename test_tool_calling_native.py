"""Run full tool calling benchmark on v21 using its native template."""
import sys
import os

sys.path.insert(0, "src")

# Force reload
for mod in list(sys.modules.keys()):
    if "finetune_studio" in mod:
        del sys.modules[mod]

from finetune_studio.benchmarks.tool_calling import (
    build_tool_prompt_for_model,
    TOOL_CALL_TESTS,
    ToolCallEvaluator,
)
from llama_cpp import Llama

v21_path = os.path.expanduser(
    "~/finetune-studio/output_gemma4_v21/export_gguf_gguf/gemma-4-e4b-it.Q4_K_M.gguf"
)

print(f"Loading {v21_path}...")
llm = Llama(model_path=v21_path, n_ctx=4096, n_threads=4, n_gpu_layers=99, verbose=False)

evaluator = ToolCallEvaluator()

correct = 0
total = 0
results = []

for i, test in enumerate(TOOL_CALL_TESTS):
    total += 1
    prompt = build_tool_prompt_for_model(v21_path, test)

    response = llm(
        prompt,
        max_tokens=200,
        temperature=0.0,
        stop=["<|turn|>", "<|tool_response|>", "</s>"],
    )
    output = response["choices"][0]["text"]

    tool_call = evaluator.parse_tool_call(output)
    result = evaluator.evaluate_tool_call(tool_call, test)

    status = "PASS" if result["correct"] else "FAIL"
    if result["correct"]:
        correct += 1

    results.append({
        "name": test.name,
        "status": status,
        "tool_called": result.get("tool_called"),
        "expected": test.expected_tools,
        "forbidden": test.forbidden_tools,
        "category": test.category,
    })

    print(f"[{i+1}/{len(TOOL_CALL_TESTS)}] {test.name} - {status}")
    print(f"  Q: {test.user_message[:60]}")
    print(f"  Out: {output[:80].strip()}")
    print(f"  Called: {result.get('tool_called')}, Expected: {test.expected_tools}")
    print()

# Summary
print("=" * 60)
print(f"RESULTS: {correct}/{total} ({100*correct/total:.1f}%)")
print()

# By category
by_cat = {}
for r in results:
    cat = TOOL_CALL_TESTS[next(i for i, t in enumerate(TOOL_CALL_TESTS) if t.name == r["name"])].category
    if cat not in by_cat:
        by_cat[cat] = [0, 0]
    by_cat[cat][1] += 1
    if r["status"] == "PASS":
        by_cat[cat][0] += 1

print("By category:")
for cat, (c, t) in sorted(by_cat.items()):
    print(f"  {cat}: {c}/{t} ({100*c/t:.0f}%)")