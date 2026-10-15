"""GGUF text path must not hand llama.cpp a prompt that already starts with BOS."""
from __future__ import annotations

from finetune_studio.testing.inference import InferenceEngine


class _FakeLlama:
    def __init__(self) -> None:
        self.prompt = ""

    def __call__(self, prompt: str, **_kw: object) -> dict:
        self.prompt = prompt
        return {"choices": [{"text": " ok "}]}


def _engine(template: str) -> tuple[InferenceEngine, _FakeLlama]:
    eng = InferenceEngine()
    fake = _FakeLlama()
    eng.model = fake
    eng.vision = False
    eng._gguf_template = {"chat_template": template, "bos_token": "<bos>", "eos_token": "<eos>"}
    return eng, fake


def test_leading_bos_from_template_is_stripped() -> None:
    eng, fake = _engine("{{ bos_token }}{% for m in messages %}[{{ m.content }}]{% endfor %}")
    out = eng._generate_gguf([{"role": "user", "content": "hi"}], 8, 0.0, 0.9, 40, 1.1, None)
    assert out == "ok"
    assert not fake.prompt.startswith("<bos>")
    assert "[hi]" in fake.prompt


def test_prompt_without_bos_is_untouched() -> None:
    eng, fake = _engine("{% for m in messages %}[{{ m.content }}]{% endfor %}")
    eng._generate_gguf([{"role": "user", "content": "hi"}], 8, 0.0, 0.9, 40, 1.1, None)
    assert fake.prompt.startswith("[hi]")
