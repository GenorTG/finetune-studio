"""Tests for training/vram/ package — constants, estimation, recommend."""
import pytest
from unittest.mock import patch, MagicMock


# ── Constants ────────────────────────────────────────────────────────────────

class TestConstants:
    def test_dtype_bytes_known_values(self):
        from finetune_studio.training.vram.constants import DTYPE_BYTES
        assert DTYPE_BYTES["float32"] == 4
        assert DTYPE_BYTES["float16"] == 2
        assert DTYPE_BYTES["bfloat16"] == 2
        assert DTYPE_BYTES["int8"] == 1
        assert DTYPE_BYTES["int4"] == 0.5
        assert DTYPE_BYTES["nf4"] == 0.5

    def test_model_presets_known(self):
        from finetune_studio.training.vram.constants import MODEL_PRESETS
        assert "qwen2.5-7b" in MODEL_PRESETS
        assert "llama-3.1-8b" in MODEL_PRESETS
        assert "phi-4-mini" in MODEL_PRESETS
        # Check shape of a preset
        p = MODEL_PRESETS["qwen2.5-7b"]
        assert p["params_b"] == 7
        assert "hidden" in p
        assert "layers" in p

    def test_cuda_overhead_positive(self):
        from finetune_studio.training.vram.constants import CUDA_OVERHEAD_GB
        assert CUDA_OVERHEAD_GB > 0

    def test_activation_safety_margin_positive(self):
        from finetune_studio.training.vram.constants import ACTIVATION_SAFETY_MARGIN
        assert ACTIVATION_SAFETY_MARGIN > 1.0


# ── Schema ───────────────────────────────────────────────────────────────────

class TestVRAMEstimate:
    def test_estimate_to_dict(self):
        from finetune_studio.training.vram.schema import VRAMEstimate
        est = VRAMEstimate(
            model_weights_gb=1.5,
            gradients_gb=1.5,
            optimizer_states_gb=6.0,
            activations_gb=2.0,
            total_gb=11.0,
            available_gb=24.0,
            headroom_gb=13.0,
            fits=True,
            method="qlora",
            model_size_b=7.0,
            max_batch_size=4,
            max_seq_length=2048,
        )
        d = est.to_dict()
        assert d["model_weights_gb"] == 1.5
        assert d["fits"] is True
        assert d["method"] == "qlora"
        assert d["total_gb"] == 11.0


# ── Estimation ────────────────────────────────────────────────────────────────

class TestEstimateVRAM:
    """Tests for the core VRAM estimation formula."""

    def test_qlora_smaller_than_lora(self):
        from finetune_studio.training.vram.estimate import estimate_vram
        with patch("finetune_studio.training.vram.estimate.detect_gpu") as mock_detect:
            mock_detect.return_value = MagicMock(free_vram_gb=24.0)
            qlora = estimate_vram(7.0, method="qlora", batch_size=2, seq_length=2048)
            lora = estimate_vram(7.0, method="lora", batch_size=2, seq_length=2048)
            # QLoRA model weights should be smaller (NF4 = 0.5 bytes vs BF16 = 2)
            assert qlora.model_weights_gb < lora.model_weights_gb

    def test_qlora_fits_on_24gb(self):
        from finetune_studio.training.vram.estimate import estimate_vram
        with patch("finetune_studio.training.vram.estimate.detect_gpu") as mock_detect:
            mock_detect.return_value = MagicMock(free_vram_gb=24.0)
            est = estimate_vram(7.0, method="qlora", batch_size=2, seq_length=2048)
            assert est.fits is True
            assert est.total_gb < 24.0

    def test_full_ft_needs_more_vram_than_qlora(self):
        from finetune_studio.training.vram.estimate import estimate_vram
        with patch("finetune_studio.training.vram.estimate.detect_gpu") as mock_detect:
            mock_detect.return_value = MagicMock(free_vram_gb=24.0)
            full = estimate_vram(7.0, method="full_ft", batch_size=1, seq_length=512)
            qlora = estimate_vram(7.0, method="qlora", batch_size=1, seq_length=512)
            assert full.total_gb > qlora.total_gb

    def test_qlora_factors(self):
        """QLORA model weights = params * 0.5 bytes (NF4)."""
        from finetune_studio.training.vram.estimate import estimate_vram
        with patch("finetune_studio.training.vram.estimate.detect_gpu") as mock_detect:
            mock_detect.return_value = MagicMock(free_vram_gb=100.0)
            est = estimate_vram(model_size_b=7.0, method="qlora", batch_size=1, seq_length=512)
            # Model weights should be 7B * 0.5 = 3.5 GB
            assert 3.0 <= est.model_weights_gb <= 4.0

    def test_headroom_calculation(self):
        from finetune_studio.training.vram.estimate import estimate_vram
        with patch("finetune_studio.training.vram.estimate.detect_gpu") as mock_detect:
            mock_detect.return_value = MagicMock(free_vram_gb=24.0)
            est = estimate_vram(7.0, method="qlora", batch_size=2, seq_length=2048)
            # headroom = available - total
            assert abs(est.headroom_gb - (est.available_gb - est.total_gb)) < 0.1

    def test_gradient_checkpointing_reduces_activations(self):
        from finetune_studio.training.vram.estimate import estimate_vram
        with patch("finetune_studio.training.vram.estimate.detect_gpu") as mock_detect:
            mock_detect.return_value = MagicMock(free_vram_gb=24.0)
            with_gc = estimate_vram(7.0, method="qlora", batch_size=2, seq_length=2048,
                                     gradient_checkpointing=True)
            without_gc = estimate_vram(7.0, method="qlora", batch_size=2, seq_length=2048,
                                       gradient_checkpointing=False)
            assert with_gc.activations_gb < without_gc.activations_gb

    def test_larger_batch_increases_activations(self):
        from finetune_studio.training.vram.estimate import estimate_vram
        with patch("finetune_studio.training.vram.estimate.detect_gpu") as mock_detect:
            mock_detect.return_value = MagicMock(free_vram_gb=24.0)
            small = estimate_vram(7.0, method="qlora", batch_size=1, seq_length=1024)
            large = estimate_vram(7.0, method="qlora", batch_size=4, seq_length=1024)
            assert large.activations_gb > small.activations_gb


# ── Recommend ────────────────────────────────────────────────────────────────

class TestRecommendConfig:
    def test_recommend_returns_list(self):
        from finetune_studio.training.vram.recommend import recommend_config
        with patch("finetune_studio.training.vram.recommend.detect_gpu") as mock_detect:
            mock_detect.return_value = MagicMock(free_vram_gb=24.0)
            configs = recommend_config(model_size_b=7.0, lora_rank=64)
            assert isinstance(configs, list)
            assert len(configs) > 0

    def test_all_configs_fit_true_or_false(self):
        from finetune_studio.training.vram.recommend import recommend_config
        with patch("finetune_studio.training.vram.recommend.detect_gpu") as mock_detect:
            mock_detect.return_value = MagicMock(free_vram_gb=24.0)
            configs = recommend_config(model_size_b=7.0, lora_rank=64)
            for c in configs:
                assert isinstance(c.fits, bool)

    def test_fits_configs_are_under_budget(self):
        from finetune_studio.training.vram.recommend import recommend_config
        with patch("finetune_studio.training.vram.recommend.detect_gpu") as mock_detect:
            mock_detect.return_value = MagicMock(free_vram_gb=24.0)
            configs = recommend_config(model_size_b=7.0, lora_rank=64)
            fitting = [c for c in configs if c.fits]
            for c in fitting:
                assert c.estimated_vram_gb <= 24.0

    def test_results_sorted_by_estimated_vram(self):
        from finetune_studio.training.vram.recommend import recommend_config
        with patch("finetune_studio.training.vram.recommend.detect_gpu") as mock_detect:
            mock_detect.return_value = MagicMock(free_vram_gb=24.0)
            configs = recommend_config(model_size_b=7.0, lora_rank=64)
            # All configs should have valid estimated_vram_gb values
            for c in configs:
                assert c.estimated_vram_gb > 0
                # Fits configs should be under the 24GB budget
                if c.fits:
                    assert c.estimated_vram_gb <= 24.0

    def test_method_in_results(self):
        from finetune_studio.training.vram.recommend import recommend_config
        with patch("finetune_studio.training.vram.recommend.detect_gpu") as mock_detect:
            mock_detect.return_value = MagicMock(free_vram_gb=24.0)
            configs = recommend_config(model_size_b=7.0, lora_rank=64)
            methods = {c.method for c in configs}
            assert methods <= {"qlora", "lora", "full_ft"}
