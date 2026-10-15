"""Train-time and eval-time chat formatting must share one formatter (CPU-only)."""

import inspect

from finetune_studio.testing import inference, suite
from finetune_studio.training import engine
from finetune_studio.training.data import format_for_sft
from finetune_studio.training.formatting import render_chat_text, with_system_prompt
from finetune_studio.training.vram import profile


class FakeTok:
    """ChatML-like template; records the kwargs it was called with."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        self.calls.append(kw)
        out = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
        return out + ("<|assistant|>" if add_generation_prompt else "")


def test_train_text_is_prefix_of_eval_text():
    tok = FakeTok()
    train = format_for_sft([{"messages": [{"role": "user", "content": "q"},
                                          {"role": "assistant", "content": "a"}]}], "SYS")[0]["messages"]
    prompt = with_system_prompt([{"role": "user", "content": "q"}], "SYS")
    train_text = render_chat_text(tok, train)
    eval_text = render_chat_text(tok, prompt, generation=True)
    assert train_text.startswith(eval_text[: -len("<|assistant|>")])
    assert eval_text.endswith("<|assistant|>")


def test_with_system_prompt_matches_bake_rule():
    assert with_system_prompt([{"role": "user", "content": "q"}], "")[0]["role"] == "user"
    already = [{"role": "system", "content": "own"}, {"role": "user", "content": "q"}]
    assert with_system_prompt(already, "SYS") == already
    assert with_system_prompt([{"role": "user", "content": "q"}], "SYS")[0]["content"] == "SYS"


def test_enable_thinking_only_forwarded_when_given():
    tok = FakeTok()
    render_chat_text(tok, [], generation=False)
    render_chat_text(tok, [], generation=True, enable_thinking=False)
    assert tok.calls == [{}, {"enable_thinking": False}]


def test_tool_schemas_are_passed_to_chat_template():
    tok = FakeTok()
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    render_chat_text(tok, [{"role": "user", "content": "lookup"}], tools=tools)
    assert tok.calls == [{"tools": tools}]


def test_run_suite_sends_system_prompt():
    seen = []

    class Eng:
        def generate(self, messages, **kw):
            seen.append(messages)
            return "ok"

    case = suite.BenchmarkCase(name="n", question="q", correct_answer="a")
    suite.run_suite(Eng(), [case], system_prompt="SYS")
    suite.run_suite(Eng(), [case])
    assert seen[0][0] == {"role": "system", "content": "SYS"}
    assert seen[1] == [{"role": "user", "content": "q"}]


def test_no_direct_apply_chat_template_in_train_or_eval_paths():
    for mod in (engine, profile, inference):
        assert "apply_chat_template" not in inspect.getsource(mod), mod.__name__
