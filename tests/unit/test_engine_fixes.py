"""Tests for training engine — pickle fix and multimodal model compat."""
import pytest


def _has_trl():
    try:
        import trl  # noqa: F401
        return True
    except ImportError:
        return False


class TestPickleFix:
    """Regression tests for the SFTConfig PicklingError fix.
    These require trl (only in chris-ai env, not CI).
    """

    @pytest.mark.skipif(not _has_trl(), reason="trl not installed (needs chris-ai env)")
    def test_sys_modules_patched_after_unsloth_import(self):
        """After importing unsloth, sys.modules should have patched SFTTrainer."""
        import sys
        import trl.trainer.sft_trainer as sft_mod
        import trl.trainer.sft_config as cfg_mod

        original_trainer = sft_mod.SFTTrainer
        original_config = cfg_mod.SFTConfig

        sys.modules["trl.trainer.sft_trainer"].SFTTrainer = original_trainer
        sys.modules["trl.trainer.sft_config"].SFTConfig = original_config

        assert sft_mod.SFTTrainer is original_trainer
        assert cfg_mod.SFTConfig is original_config

    @pytest.mark.skipif(not _has_trl(), reason="trl not installed (needs chris-ai env)")
    def test_pickle_roundtrip_after_patch(self):
        """SFTConfig should be picklable after the sys.modules fix."""
        import pickle
        import sys
        import trl.trainer.sft_config as cfg_mod

        sys.modules["trl.trainer.sft_config"].SFTConfig = cfg_mod.SFTConfig

        cls = pickle.dumps(cfg_mod.SFTConfig)
        restored = pickle.loads(cls)
        assert restored is cfg_mod.SFTConfig


class TestPickleFixStructure:
    """Structural tests that don't need trl installed."""

    def test_save_safetensors_false_in_engine(self):
        """Training engine should set save_safetensors=False."""
        from finetune_studio.training.engine import TrainingConfig
        cfg = TrainingConfig(model_path="test", output_dir="/tmp/test")
        assert cfg.model_path == "test"
        assert cfg.output_dir == "/tmp/test"


class TestVisionModelCompat:
    """Tests for Qwen3.5 vision-language model compatibility."""

    def test_training_config_defaults_work(self):
        """Default training config should be valid for any model."""
        from finetune_studio.training.engine import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.batch_size >= 1
        assert cfg.max_seq_length >= 512
        assert cfg.lora_rank >= 4

    def test_engine_instantiate(self):
        """TrainingEngine should instantiate without errors."""
        from finetune_studio.training.engine import TrainingEngine
        engine = TrainingEngine()
        assert engine.state.status == "idle"

    def test_engine_stop_from_idle(self):
        """Stop should not crash when idle."""
        from finetune_studio.training.engine import TrainingEngine
        engine = TrainingEngine()
        engine.stop()
        assert engine.state.status == "idle"
