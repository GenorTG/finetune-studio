"""Tests for training engine class registration and configuration defaults."""
from types import ModuleType, SimpleNamespace


class TestTrlPickleRegistration:
    def test_registers_patched_classes_on_pickle_lookup_modules(self, monkeypatch):
        import sys

        from finetune_studio.training.engine import _register_patched_trl_classes

        trainer_registry = ModuleType("trl.trainer.sft_trainer")
        config_registry = ModuleType("trl.trainer.sft_config")
        monkeypatch.setitem(sys.modules, trainer_registry.__name__, trainer_registry)
        monkeypatch.setitem(sys.modules, config_registry.__name__, config_registry)
        patched_trainer = object()
        patched_config = object()

        _register_patched_trl_classes(
            SimpleNamespace(SFTTrainer=patched_trainer),
            SimpleNamespace(SFTConfig=patched_config),
        )

        assert trainer_registry.SFTTrainer is patched_trainer
        assert config_registry.SFTConfig is patched_config


class TestTrainingConfig:
    def test_config_paths_are_retained(self):
        from finetune_studio.training.engine import TrainingConfig

        cfg = TrainingConfig(model_path="test", output_dir="/tmp/test")
        assert cfg.model_path == "test"
        assert cfg.output_dir == "/tmp/test"

    def test_defaults_are_valid(self):
        from finetune_studio.training.engine import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.batch_size >= 1
        assert cfg.max_seq_length >= 512
        assert cfg.lora_rank >= 4


class TestTrainingEngineLifecycle:
    def test_engine_instantiate(self):
        from finetune_studio.training.engine import TrainingEngine

        engine = TrainingEngine()
        assert engine.state.status == "idle"

    def test_engine_stop_from_idle(self):
        from finetune_studio.training.engine import TrainingEngine

        engine = TrainingEngine()
        engine.stop()
        assert engine.state.status == "idle"
