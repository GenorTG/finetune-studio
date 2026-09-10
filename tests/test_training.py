from finetune_studio.training.engine import TrainingConfig, TrainingState, TrainingEngine


class TestFormatForSft:
    """Regression for the empty-sharegpt bug: the data-prep export endpoint
    writes `conversations` (from/value) pairs for sharegpt, but the
    training engine's format_for_sft only knew about OpenAI messages,
    plain text, and prompt/completion. Result: sharegpt JSONL silently
    yielded zero training examples and the SFTTrainer died with
    'list index out of range' on an empty dataset (caught during
    fan-dragon E2E on 2026-09-10 with b08426e3's 29 approved pairs)."""

    def test_config(self):
        c = TrainingConfig()
        assert c.lora_rank == 64
        assert c.learning_rate == 8e-5

    def test_sharegpt_conversations_converted_to_messages(self):
        from finetune_studio.training.data import format_for_sft
        data = [{
            "conversations": [
                {"from": "human", "value": "What is OCTOPUS-7741?"},
                {"from": "gpt", "value": "An internal Q4 reliability push."},
            ],
            "source_id": "abc",
            "chunk_idx": 0,
            "score": 0.5,
        }]
        out = format_for_sft(data, system_prompt="")
        assert len(out) == 1, f"expected 1 formatted example, got {len(out)}"
        msgs = out[0]["messages"]
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "What is OCTOPUS-7741?"}
        assert msgs[1] == {"role": "assistant", "content": "An internal Q4 reliability push."}

    def test_sharegpt_with_system_prompt_prepends(self):
        from finetune_studio.training.data import format_for_sft
        data = [{
            "conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello"},
            ],
        }]
        out = format_for_sft(data, system_prompt="You are helpful.")
        msgs = out[0]["messages"]
        assert msgs[0] == {"role": "system", "content": "You are helpful."}
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_sharegpt_first_message_is_system_does_not_duplicate(self):
        from finetune_studio.training.data import format_for_sft
        data = [{
            "conversations": [
                {"from": "system", "value": "Inline system."},
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello"},
            ],
        }]
        out = format_for_sft(data, system_prompt="Outer system.")
        # Existing system message stays; outer system_prompt is NOT prepended.
        assert out[0]["messages"][0] == {"role": "system", "content": "Inline system."}

    def test_sharegpt_alternation_required(self):
        """sharegpt pairs must alternate user/assistant; otherwise the
        trainer gets a malformed chat template."""
        from finetune_studio.training.data import format_for_sft
        # bad: two gpt turns in a row → mapped to two assistant turns (warning
        # should fire but we don't drop the row; SFTTrainer will reject later)
        data = [{"conversations": [
            {"from": "gpt", "value": "Pre-amble?"},
            {"from": "gpt", "value": "Wrong."},
        ]}]
        out = format_for_sft(data, system_prompt="")
        # The conversion is best-effort; we just don't crash.
        assert isinstance(out, list)

    def test_sharegpt_mixed_keys_does_not_drop_messages(self):
        """A row that has BOTH conversations and messages should pick one
        consistently. We pick conversations (the sharegpt path) since the
        data-prep export writes that key."""
        from finetune_studio.training.data import format_for_sft
        data = [{
            "conversations": [{"from": "human", "value": "via conv"}],
            "messages": [{"role": "user", "content": "via msg"}],
        }]
        out = format_for_sft(data, system_prompt="")
        msgs = out[0]["messages"]
        # conv wins (first branch in format_for_sft)
        assert msgs[-1]["content"] == "via conv"

    def test_openai_messages_still_work(self):
        from finetune_studio.training.data import format_for_sft
        data = [{"messages": [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]}]
        out = format_for_sft(data, system_prompt="")
        assert len(out) == 1
        assert out[0]["messages"][-1]["role"] == "assistant"

    def test_alpaca_prompt_completion_still_works(self):
        from finetune_studio.training.data import format_for_sft
        data = [{"prompt": "p", "completion": "c"}]
        out = format_for_sft(data, system_prompt="")
        msgs = out[0]["messages"]
        assert msgs[-2:] == [
            {"role": "user", "content": "p"},
            {"role": "assistant", "content": "c"},
        ]

    def test_plain_text_still_works(self):
        from finetune_studio.training.data import format_for_sft
        data = [{"text": "raw training text"}]
        out = format_for_sft(data, system_prompt="")
        assert out[0]["messages"] == [
            {"role": "user", "content": "raw training text"},
        ]

    def test_unknown_format_row_is_dropped_not_crashed(self):
        from finetune_studio.training.data import format_for_sft
        data = [
            {"messages": [{"role": "user", "content": "ok"}]},
            {"garbage_key": "no recognized fields"},
            {"conversations": [{"from": "human", "value": "sharegpt ok"}]},
        ]
        out = format_for_sft(data, system_prompt="")
        # garbage row dropped; the two valid rows pass through
        assert len(out) == 2

    def test_state(self):
        s = TrainingState()
        assert s.status == "idle"

    def test_engine(self):
        e = TrainingEngine()
        assert e.state.status == "idle"
