"""Tests for VRAM profiler module."""
import pytest
from unittest.mock import patch, MagicMock


class TestGPUInfo:
    """Tests for GPU detection."""

    def test_gpu_info_detect_no_cuda(self):
        """When CUDA is unavailable, returns safe defaults."""
        with patch("torch.cuda.is_available", return_value=False):
            from finetune_studio.training.vram import GPUInfo, detect
            gpu = detect()
            assert gpu.total_vram_gb == 0
            assert gpu.supports_flash_attention is False

    def test_gpu_info_detect_with_cuda(self):
        """When CUDA is available, returns real GPU info."""
        mock_props = MagicMock()
        mock_props.name = "NVIDIA GeForce RTX 3090"
        mock_props.total_memory = 24576 * 1024 * 1024  # 24GB
        mock_props.major = 8
        mock_props.minor = 6

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_properties", return_value=mock_props), \
             patch("torch.cuda.mem_get_info", return_value=(22 * 1024**3, 2 * 1024**3)):
            from finetune_studio.training.vram import GPUInfo, detect
            gpu = detect()
            assert gpu.name == "NVIDIA GeForce RTX 3090"
            assert gpu.total_vram_gb > 0
            assert gpu.supports_flash_attention is True
            assert gpu.supports_bf16 is True


class TestEstimateVRAM:
    """Tests for VRAM estimation formula."""

    def test_qlora_7b_fits_24gb(self):
        """QLoRA on 7B should fit in 24GB."""
        from finetune_studio.training.vram_profiler import estimate_vram
        est = estimate_vram(
            model_size_b=7, method="qlora",
            batch_size=2, seq_length=2048,
            available_vram_gb=21.4,
        )
        assert est.fits is True
        assert est.method == "qlora"
        assert est.total_gb < 21.4

    def test_full_ft_7b_doesnt_fit_24gb(self):
        """Full fine-tuning on 7B should NOT fit in 24GB."""
        from finetune_studio.training.vram_profiler import estimate_vram
        est = estimate_vram(
            model_size_b=7, method="full_ft",
            batch_size=2, seq_length=2048,
            available_vram_gb=21.4,
        )
        assert est.fits is False

    def test_qlora_14b_fits_24gb(self):
        """QLoRA on 14B should fit in 24GB."""
        from finetune_studio.training.vram_profiler import estimate_vram
        est = estimate_vram(
            model_size_b=14, method="qlora",
            batch_size=2, seq_length=2048,
            available_vram_gb=21.4,
        )
        assert est.fits is True

    def test_qlora_27b_fits_24gb_tight(self):
        """QLoRA on 27B should fit but tight."""
        from finetune_studio.training.vram_profiler import estimate_vram
        est = estimate_vram(
            model_size_b=27, method="qlora",
            batch_size=2, seq_length=2048,
            available_vram_gb=21.4,
        )
        assert est.fits is True
        assert est.headroom_gb < 3  # Should be tight

    def test_qlora_32b_doesnt_fit_24gb(self):
        """QLoRA on 32B should NOT fit in 24GB."""
        from finetune_studio.training.vram_profiler import estimate_vram
        est = estimate_vram(
            model_size_b=32, method="qlora",
            batch_size=2, seq_length=2048,
            available_vram_gb=21.4,
        )
        assert est.fits is False

    def test_smaller_batch_saves_vram(self):
        """Smaller batch should use less VRAM."""
        from finetune_studio.training.vram_profiler import estimate_vram
        est_large = estimate_vram(
            model_size_b=7, method="qlora",
            batch_size=8, seq_length=2048,
            available_vram_gb=21.4,
        )
        est_small = estimate_vram(
            model_size_b=7, method="qlora",
            batch_size=1, seq_length=2048,
            available_vram_gb=21.4,
        )
        assert est_small.total_gb < est_large.total_gb

    def test_gradient_checkpointing_saves_vram(self):
        """Gradient checkpointing should reduce activation memory."""
        from finetune_studio.training.vram_profiler import estimate_vram
        est_no_gc = estimate_vram(
            model_size_b=7, method="qlora",
            batch_size=2, seq_length=2048,
            gradient_checkpointing=False,
            available_vram_gb=21.4,
        )
        est_gc = estimate_vram(
            model_size_b=7, method="qlora",
            batch_size=2, seq_length=2048,
            gradient_checkpointing=True,
            available_vram_gb=21.4,
        )
        assert est_gc.activations_gb < est_no_gc.activations_gb

    def test_to_dict_roundtrip(self):
        """VRAMEstimate.to_dict should produce serializable dict."""
        from finetune_studio.training.vram_profiler import estimate_vram
        est = estimate_vram(model_size_b=7, method="qlora")
        d = est.to_dict()
        assert isinstance(d, dict)
        assert "total_gb" in d
        assert "fits" in d
        assert isinstance(d["total_gb"], float)


class TestRecommendConfig:
    """Tests for config recommendation."""

    def test_returns_configs_for_7b(self):
        """Should return configs for 7B on 24GB."""
        from finetune_studio.training.vram_profiler import recommend_config
        configs = recommend_config(model_size_b=7, available_vram_gb=21.4)
        assert len(configs) > 0
        assert all(c.fits for c in configs)

    def test_returns_configs_for_14b(self):
        """Should return configs for 14B on 24GB."""
        from finetune_studio.training.vram_profiler import recommend_config
        configs = recommend_config(model_size_b=14, available_vram_gb=21.4)
        assert len(configs) > 0

    def test_no_configs_for_70b_on_24gb(self):
        """Should return no configs for 70B on 24GB."""
        from finetune_studio.training.vram_profiler import recommend_config
        configs = recommend_config(model_size_b=70, available_vram_gb=21.4)
        assert len(configs) == 0

    def test_sorted_by_quality(self):
        """Configs should be sorted by quality (full_ft > lora > qlora)."""
        from finetune_studio.training.vram_profiler import recommend_config
        configs = recommend_config(model_size_b=1, available_vram_gb=21.4)
        if len(configs) >= 2:
            # First config should be highest quality method
            methods = [c.method for c in configs]
            quality_order = {"full_ft": 3, "lora": 2, "qlora": 1}
            for i in range(len(methods) - 1):
                assert quality_order[methods[i]] >= quality_order[methods[i + 1]]


class TestModelPresets:
    """Tests for model presets."""

    def test_known_models_exist(self):
        """Should have presets for common models."""
        from finetune_studio.training.vram_profiler import MODEL_PRESETS
        assert "qwen2.5-7b" in MODEL_PRESETS
        assert "gemma-2-9b" in MODEL_PRESETS
        assert "phi-4" in MODEL_PRESETS

    def test_preset_has_required_fields(self):
        """Each preset should have params_b, hidden, layers."""
        from finetune_studio.training.vram_profiler import MODEL_PRESETS
        for name, preset in MODEL_PRESETS.items():
            assert "params_b" in preset, f"{name} missing params_b"
            assert "hidden" in preset, f"{name} missing hidden"
            assert "layers" in preset, f"{name} missing layers"
            assert preset["params_b"] > 0
            assert preset["hidden"] > 0
            assert preset["layers"] > 0


class TestRecommendForModel:
    """Tests for recommend_for_model helper."""

    def test_known_model(self):
        """Should work for known model names."""
        from finetune_studio.training.vram_profiler import recommend_for_model
        configs = recommend_for_model("qwen2.5-7b", available_vram_gb=21.4)
        assert len(configs) > 0

    def test_unknown_model_raises(self):
        """Should raise for unknown model names."""
        from finetune_studio.training.vram_profiler import recommend_for_model
        with pytest.raises(ValueError, match="Unknown model"):
            recommend_for_model("nonexistent-model-999b")

    def test_partial_match(self):
        """Should match partial names like 'qwen2.5-7b' from 'Qwen2.5-7B-Instruct'."""
        from finetune_studio.training.vram_profiler import recommend_for_model
        configs = recommend_for_model("qwen2.5-7b-instruct", available_vram_gb=21.4)
        assert len(configs) > 0


class TestGenerateReport:
    """Tests for report generation."""

    def test_report_contains_gpu_info(self):
        """Report should contain GPU info."""
        from finetune_studio.training.vram_profiler import generate_vram_report
        report = generate_vram_report(available_vram_gb=21.4)
        assert "VRAM Training Report" in report
        assert "Recommendations" in report

    def test_report_save_to_file(self, tmp_path):
        """Report should save to file when output_path given."""
        from finetune_studio.training.vram_profiler import generate_vram_report
        out = tmp_path / "report.md"
        report = generate_vram_report(available_vram_gb=21.4, output_path=str(out))
        assert out.exists()
        assert len(report) > 100
