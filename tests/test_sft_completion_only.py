"""SFT puts the loss on the answers (training/formatting.split_prompt_completion, TrainingEngine._sft_formatter)."""
from __future__ import annotations

from finetune_studio.training.engine import TrainingEngine
from finetune_studio.training.formatting import split_prompt_completion


class _ChatML:
    """ChatML like Qwen3.5: an empty think block before the answer when thinking is off."""

    eos_token = "<|im_end|>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=None, **_):
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages[:-1] if messages)
        last = messages[-1] if messages else None
        if last is not None and last["role"] == "assistant" and not add_generation_prompt:
            out += f"<|im_start|>assistant\n<think>\n\n</think>\n\n{last['content']}<|im_end|>\n"
        elif last is not None:
            out += f"<|im_start|>{last['role']}\n{last['content']}<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if enable_thinking is False else "<think>\n")
        return out


ROW = [{"role": "system", "content": "Be exact."}, {"role": "user", "content": "Which line?"},
       {"role": "assistant", "content": "Ext 4419."}]


def test_split_matches_the_inference_prompt_and_ends_at_eos() -> None:
    tok = _ChatML()
    prompt, completion = split_prompt_completion(tok, ROW)
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert completion == "Ext 4419.<|im_end|>"             # no "\n" left for TRL to follow with a second <|im_end|>
    assert prompt + completion + "\n" == tok.apply_chat_template(ROW)


def test_rows_that_do_not_end_with_an_answer_are_not_split() -> None:
    assert split_prompt_completion(_ChatML(), ROW[:2]) is None


def test_formatter_uses_prompt_completion_for_chat_rows() -> None:
    fmt = TrainingEngine._sft_formatter(_ChatML(), [{"messages": ROW}])
    out = fmt({"messages": ROW, "text": None, "tools": None})      # a Dataset row carries every column
    assert set(out) == {"prompt", "completion"} and out["completion"] == "Ext 4419.<|im_end|>"


def test_one_unsplittable_row_turns_the_whole_dataset_to_text_ending_at_eos() -> None:
    rows = [{"messages": ROW}, {"messages": ROW[:2]}]
    fmt = TrainingEngine._sft_formatter(_ChatML(), rows)
    out = fmt({"messages": ROW})
    assert set(out) == {"text"} and out["text"].endswith("Ext 4419.<|im_end|>")


def test_plain_text_rows_stay_text() -> None:
    fmt = TrainingEngine._sft_formatter(_ChatML(), [{"text": "raw corpus line"}])
    assert fmt({"text": "raw corpus line", "messages": None}) == {"text": "raw corpus line"}
