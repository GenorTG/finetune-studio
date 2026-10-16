"""Chat answers must stop at the template's end-of-turn token (training/formatting.chat_stop_token_ids)."""
from __future__ import annotations

from types import SimpleNamespace

from finetune_studio.training.formatting import chat_stop_token_ids

QWEN_VOCAB = {"<|endoftext|>": 248044, "<|im_start|>": 248045, "<|im_end|>": 248046}


class _Tok:
    def __init__(self, eos: int | None, vocab: dict[str, int], unk: int | None = None) -> None:
        self.eos_token_id, self._vocab, self.unk_token_id = eos, vocab, unk

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


def test_qwen_base_without_generation_config_still_stops_at_im_end() -> None:
    # The merged Qwen3.5-9B: generation_config derived from config.json says <|endoftext|> only.
    ids = chat_stop_token_ids(_Tok(248046, QWEN_VOCAB), SimpleNamespace(eos_token_id=248044))
    assert ids == [248044, 248046]


def test_end_tokens_come_from_the_vocab_even_when_the_tokenizer_eos_is_endoftext() -> None:
    assert chat_stop_token_ids(_Tok(248044, QWEN_VOCAB), None) == [248044, 248046]


def test_list_eos_is_kept_in_order_without_duplicates_and_unknown_tokens_are_skipped() -> None:
    vocab = {"<|eot_id|>": 128009, "<|im_end|>": 0}
    ids = chat_stop_token_ids(_Tok(128001, vocab, unk=0), SimpleNamespace(eos_token_id=[128001, 128008]))
    assert ids == [128001, 128008, 128009]


def test_a_processor_is_unwrapped_to_its_tokenizer() -> None:
    processor = SimpleNamespace(tokenizer=_Tok(248046, QWEN_VOCAB))
    assert chat_stop_token_ids(processor, SimpleNamespace(eos_token_id=None)) == [248046]


def test_hf_generate_passes_the_chat_stop_ids(monkeypatch) -> None:
    import torch

    from finetune_studio.testing import inference as inf

    seen: dict = {}

    class _Model:
        device = "cpu"
        generation_config = SimpleNamespace(eos_token_id=248044)

        def generate(self, **kw):
            seen.update(kw)
            return torch.tensor([[1, 2, 3, 248046]])

    class _HFTok(_Tok):
        pad_token_id = 248044

        def apply_chat_template(self, messages, **kw):
            return "prompt"

        def __call__(self, text, return_tensors=None):
            return SimpleNamespace(to=lambda device: {"input_ids": torch.tensor([[1, 2, 3]])})

        def decode(self, ids, skip_special_tokens=True):
            return "answer"

    eng = inf.InferenceEngine.__new__(inf.InferenceEngine)
    eng.model, eng.tokenizer = _Model(), _HFTok(248046, QWEN_VOCAB)
    monkeypatch.setattr("finetune_studio.accel.activate", lambda: None)
    assert eng._generate_hf([{"role": "user", "content": "q"}], 64, 0.0, 0.9, 40, 1.1, None) == "answer"
    assert seen["eos_token_id"] == [248044, 248046]
