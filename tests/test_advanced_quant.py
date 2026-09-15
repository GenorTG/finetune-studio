"""Focused tests for GPTQ capability detection and backend dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


class TestGptqCapabilityDetection:
    def test_neither_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from finetune_studio.training import advanced_quant as aq

        monkeypatch.setattr(aq, "is_gptqmodel_available", lambda: False)
        monkeypatch.setattr(aq, "is_auto_gptq_available", lambda: False)
        assert aq.is_gptq_available() is False
        assert aq.preferred_gptq_backend() is None
        msg = aq.gptq_missing_backend_message()
        assert "gptqmodel" in msg
        assert "auto_gptq" in msg

    def test_prefers_gptqmodel_over_auto_gptq(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from finetune_studio.training import advanced_quant as aq

        monkeypatch.setattr(aq, "is_gptqmodel_available", lambda: True)
        monkeypatch.setattr(aq, "is_auto_gptq_available", lambda: True)
        assert aq.is_gptq_available() is True
        assert aq.preferred_gptq_backend() == "gptqmodel"

    def test_falls_back_to_auto_gptq(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from finetune_studio.training import advanced_quant as aq

        monkeypatch.setattr(aq, "is_gptqmodel_available", lambda: False)
        monkeypatch.setattr(aq, "is_auto_gptq_available", lambda: True)
        assert aq.preferred_gptq_backend() == "auto_gptq"


class TestCalibrationExamples:
    def test_shared_calibration_texts(self) -> None:
        from finetune_studio.training.advanced_quant import (
            calibration_example_texts,
        )

        texts = calibration_example_texts(3)
        assert len(texts) == 3
        assert "calibration example number 0" in texts[0]
        assert "GPTQ quantization" in texts[2]


class TestVerifyGptqArtifacts:
    def test_ok_with_config_and_weights(self, tmp_path: Path) -> None:
        from finetune_studio.training.advanced_quant import verify_gptq_artifacts

        gptq = tmp_path / "gptq"
        gptq.mkdir()
        (gptq / "config.json").write_text("{}", encoding="utf-8")
        (gptq / "model.safetensors").write_bytes(b"weights")
        result = verify_gptq_artifacts(str(gptq))
        assert result["ok"] is True
        assert result["size_bytes"] > 0
        assert result["error"] is None

    def test_accepts_quantize_config_json(self, tmp_path: Path) -> None:
        from finetune_studio.training.advanced_quant import verify_gptq_artifacts

        gptq = tmp_path / "gptq"
        gptq.mkdir()
        (gptq / "quantize_config.json").write_text("{}", encoding="utf-8")
        (gptq / "gptq_model-4bit-128g.safetensors").write_bytes(b"w")
        assert verify_gptq_artifacts(str(gptq))["ok"] is True


class TestQuantizeGptqDispatch:
    def test_raises_when_no_backend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from finetune_studio.training import advanced_quant as aq

        monkeypatch.setattr(aq, "preferred_gptq_backend", lambda: None)
        with pytest.raises(ImportError, match="gptqmodel"):
            aq.quantize_gptq(str(tmp_path), str(tmp_path / "out"))

    def test_uses_gptqmodel_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from finetune_studio.training import advanced_quant as aq

        out = tmp_path / "gptq"
        model = tmp_path / "merged"
        model.mkdir()
        (model / "system_prompt.txt").write_text("hi", encoding="utf-8")

        called: dict[str, Any] = {}

        def _fake_gptqmodel(**kwargs: Any) -> dict[str, Any]:
            called["backend"] = "gptqmodel"
            called["texts"] = kwargs["calibration_texts"]
            out.mkdir(parents=True, exist_ok=True)
            (out / "config.json").write_text("{}", encoding="utf-8")
            (out / "model.safetensors").write_bytes(b"gptq")
            return {}

        monkeypatch.setattr(aq, "preferred_gptq_backend", lambda: "gptqmodel")
        monkeypatch.setattr(aq, "_quantize_with_gptqmodel", _fake_gptqmodel)
        monkeypatch.setattr(
            aq,
            "_quantize_with_auto_gptq",
            lambda **_k: (_ for _ in ()).throw(AssertionError("auto")),
        )

        result = aq.quantize_gptq(str(model), str(out), bits=4, group_size=128)
        assert called["backend"] == "gptqmodel"
        assert len(called["texts"]) == 128
        assert result["backend"] == "gptqmodel"
        assert result["method"] == "gptq"
        assert result["bits"] == 4
        assert result["group_size"] == 128
        verified = aq.verify_gptq_artifacts(str(out))
        assert verified["ok"] is True
        assert (out / "system_prompt.txt").is_file()

    def test_uses_auto_gptq_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from finetune_studio.training import advanced_quant as aq

        out = tmp_path / "gptq"
        model = tmp_path / "merged"
        model.mkdir()

        def _fake_auto(**kwargs: Any) -> dict[str, Any]:
            assert len(kwargs["calibration_texts"]) == 128
            out.mkdir(parents=True, exist_ok=True)
            (out / "config.json").write_text("{}", encoding="utf-8")
            (out / "model.bin").write_bytes(b"legacy")
            return {}

        monkeypatch.setattr(aq, "preferred_gptq_backend", lambda: "auto_gptq")
        monkeypatch.setattr(aq, "_quantize_with_auto_gptq", _fake_auto)
        monkeypatch.setattr(
            aq,
            "_quantize_with_gptqmodel",
            lambda **_k: (_ for _ in ()).throw(AssertionError("gptqmodel")),
        )

        result = aq.quantize_gptq(str(model), str(out))
        assert result["backend"] == "auto_gptq"
        assert aq.verify_gptq_artifacts(str(out))["ok"] is True

    def test_gptqmodel_loader_wires_calibration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unit-test the gptqmodel call shape without importing the package."""
        from finetune_studio.training import advanced_quant as aq

        class _StubConfig:
            def __init__(
                self,
                bits: int = 4,
                group_size: int = 128,
                damp_percent: float | None = None,
                desc_act: bool | None = None,
                static_groups: bool = False,
                sym: bool = True,
                true_sequential: bool = True,
                **_kwargs: Any,
            ) -> None:
                self.bits = bits
                self.group_size = group_size
                self.damp_percent = damp_percent

        calls: dict[str, Any] = {}

        class _StubModel:
            def quantize(
                self,
                calibration: list[str],
                batch_size: int = 1,
                calibration_data_min_length: int = 10,
            ) -> dict[str, Any]:
                calls["quantize"] = {
                    "calibration": calibration,
                    "batch_size": batch_size,
                    "calibration_data_min_length": calibration_data_min_length,
                }
                return {}

            def save(self, save_dir: str) -> None:
                calls["save"] = save_dir

        class _StubGPTQModel:
            @staticmethod
            def load(*_a: Any, **_k: Any) -> _StubModel:
                return _StubModel()

        class _FakeGptqModelMod:
            GPTQModel = _StubGPTQModel
            GPTQConfig = _StubConfig

        import sys

        monkeypatch.setitem(sys.modules, "gptqmodel", _FakeGptqModelMod)

        texts = ["This is calibration example number 0 for GPTQ quantization."]
        aq._quantize_with_gptqmodel(
            model_path=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            bits=4,
            group_size=128,
            damp_percent=0.01,
            calibration_texts=texts,
        )
        assert calls["quantize"]["calibration"] == texts
        assert calls["quantize"]["batch_size"] == 1
        assert calls["quantize"]["calibration_data_min_length"] == 1
        assert calls["save"] == str(tmp_path / "out")
