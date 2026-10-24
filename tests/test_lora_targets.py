"""LoRA targets come from the loaded model, for any architecture (training/lora_targets.py).

The old hard-coded ``q_proj … down_proj`` list missed Qwen3.5's linear-attention
mixers in 24 of 32 layers, all of GPT-2's layers and every MoE expert. Tiny random
models (CPU, no download) stand in for each architecture family.
"""

from __future__ import annotations

import pytest
import torch
import transformers as T

from finetune_studio.training.lora_targets import (
    LoraTargets,
    discover_lora_targets,
    resolve_lora_targets,
)

TINY = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 256}


def _meta(model_cls, cfg):
    with torch.device("meta"):
        return model_cls(cfg)


def test_llama_style_decoder_gets_all_seven_projections() -> None:
    t = discover_lora_targets(_meta(T.LlamaForCausalLM, T.LlamaConfig(**TINY)), 8)
    assert t.modules == ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"]
    assert t.parameters == []


def test_gpt2_conv1d_layers_are_found() -> None:
    m = _meta(T.GPT2LMHeadModel, T.GPT2Config(n_embd=64, n_layer=2, n_head=4, vocab_size=256))
    assert discover_lora_targets(m, 8).modules == ["c_attn", "c_fc", "c_proj"]


def test_hybrid_linear_attention_mixers_are_trained() -> None:
    cfg = T.Qwen3_5TextConfig(**TINY, head_dim=16, layer_types=["linear_attention", "full_attention"],
                              linear_num_value_heads=4, linear_num_key_heads=2,
                              linear_key_head_dim=16, linear_value_head_dim=16)
    t = discover_lora_targets(_meta(T.Qwen3_5ForCausalLM, cfg), 8)
    assert {"in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "down_proj"} <= set(t.modules)
    # in_proj_a / in_proj_b are 4 wide (one per value head): narrower than rank 8.
    assert "in_proj_a" not in t.modules and "in_proj_b" not in t.modules


def test_layers_narrower_than_the_rank_are_skipped_only_then() -> None:
    m = _meta(T.LlamaForCausalLM, T.LlamaConfig(**TINY))
    assert "k_proj" in discover_lora_targets(m, 32).modules  # k_proj is 64 -> 32 wide
    with pytest.raises(ValueError, match="no linear layer"):
        discover_lora_targets(m, 4096)


def test_moe_experts_go_to_target_parameters_and_router_is_left_alone() -> None:
    cfg = T.Qwen3MoeConfig(**TINY, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32)
    t = discover_lora_targets(_meta(T.Qwen3MoeForCausalLM, cfg), 8)
    assert t.parameters == ["experts.down_proj", "experts.gate_up_proj"]
    assert "gate" not in t.modules
    assert t.peft_kwargs() == {"target_modules": t.modules, "target_parameters": t.parameters}
    assert "MoE experts" in t.describe()


def test_moe_expert_lora_actually_receives_gradients() -> None:
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    cfg = T.Qwen3MoeConfig(**TINY, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32)
    model = T.Qwen3MoeForCausalLM(cfg)
    targets = discover_lora_targets(model, 8)
    peft = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, task_type="CAUSAL_LM", **targets.peft_kwargs()))
    ids = torch.randint(0, 256, (2, 12))
    peft(input_ids=ids, labels=ids).loss.backward()
    expert_b = [p for n, p in peft.named_parameters() if "experts" in n and "lora_B" in n]
    assert expert_b and all(p.grad is not None and p.grad.abs().sum() > 0 for p in expert_b)


def _tiny_llava():
    vision = T.CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                num_attention_heads=2, image_size=28, patch_size=14)
    return T.LlavaConfig(vision_config=vision, text_config=T.LlamaConfig(**TINY), image_token_index=255)


def test_vision_tower_and_projector_are_never_targeted() -> None:
    """CLIP shares q_proj/k_proj/v_proj with the text model: full text paths, no vision path."""
    model = _meta(T.LlavaForConditionalGeneration, _tiny_llava())
    t = discover_lora_targets(model, 8)
    assert len(t.modules) == 2 * 7  # two text layers x seven projections, full paths
    assert all("language_model" in m for m in t.modules)
    assert not any("vision" in m or "projector" in m for m in t.modules)


def test_peft_wraps_only_text_layers_of_a_vlm() -> None:
    from peft import LoraConfig, get_peft_model

    model = T.LlavaForConditionalGeneration(_tiny_llava())
    targets = discover_lora_targets(model, 8)
    peft = get_peft_model(model, LoraConfig(r=8, task_type="CAUSAL_LM", **targets.peft_kwargs()))
    wrapped = {n.split(".lora_A")[0] for n, _ in peft.named_parameters() if ".lora_A." in n}
    assert len(wrapped) == 14
    assert not any("vision" in n or "projector" in n for n in wrapped)


def test_configured_list_overrides_discovery() -> None:
    m = _meta(T.LlamaForCausalLM, T.LlamaConfig(**TINY))
    assert resolve_lora_targets(m, 8, ["q_proj", "v_proj"]) == LoraTargets(modules=["q_proj", "v_proj"])
    assert resolve_lora_targets(m, 8, []) == discover_lora_targets(m, 8)


def test_engine_default_is_auto_discovery() -> None:
    from finetune_studio.training.engine import TrainingConfig

    assert TrainingConfig().lora_target_modules == []


def test_moe_expert_adapter_survives_save_reload_and_merge(tmp_path) -> None:
    """The run's adapter is saved, reloaded for testing and merged for export."""
    from peft import LoraConfig, PeftModel, get_peft_model

    cfg = T.Qwen3MoeConfig(**TINY, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32)
    torch.manual_seed(0)
    model = T.Qwen3MoeForCausalLM(cfg)
    targets = discover_lora_targets(model, 8)
    peft = get_peft_model(model, LoraConfig(r=8, task_type="CAUSAL_LM", init_lora_weights=False,
                                            **targets.peft_kwargs()))
    ids = torch.randint(0, 256, (1, 10))
    with torch.no_grad():
        trained = peft(input_ids=ids).logits
    peft.save_pretrained(tmp_path)
    torch.manual_seed(0)
    reloaded = PeftModel.from_pretrained(T.Qwen3MoeForCausalLM(cfg), tmp_path)
    with torch.no_grad():
        assert torch.allclose(reloaded(input_ids=ids).logits, trained, atol=1e-5)
        assert torch.allclose(reloaded.merge_and_unload()(input_ids=ids).logits, trained, atol=1e-4)
