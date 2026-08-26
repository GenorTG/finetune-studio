import re
import sys

sys.path.insert(0, "src")
from finetune_studio.benchmarks.tool_calling import ToolCallEvaluator

ev = ToolCallEvaluator()

# Build the test output using chr() to avoid stripping
lt, pipe, gt = chr(60), chr(124), chr(62)
quote = chr(34)  # "

# Construct: <|tool_call|>call:calculator{expression:<|"|>15 * 37 + 42<|"|">}<tool_call|>
output = lt + pipe + "tool_call" + pipe + gt
output += "call:calculator{expression:"
output += lt + pipe + quote + pipe + gt
output += "15 * 37 + 42"
output += lt + pipe + quote + pipe + gt
output += "}"
output += lt + pipe + "tool_call" + pipe + gt

print("Output:", repr(output))
print("Output (decoded):", output)

# Test the parser
tc = ev.parse_tool_call(output)
print("Parsed:", tc)

# Test the regex directly
pattern = r"<\|tool_call\|>call:(\w+)\{(.+?)\}<tool_call\|>"
print("\nPattern:", pattern)
m = re.search(pattern, output, re.DOTALL)
if m:
    print("Regex match:", repr(m.group(0)))
    print("Name:", m.group(1))
    print("Args:", repr(m.group(2)))
else:
    print("No regex match")

# Check what the parser code actually has
import inspect
src = inspect.getsource(ev.parse_tool_call)
print("\nParser source (last 500 chars):")
print(src[-500:])