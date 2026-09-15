"""Regression: export responses must be JSON-serializable; UI shows errors."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.encoders import jsonable_encoder


class TestExportResultSerialization:
    """Abliteration historically returned numpy arrays → FastAPI 500."""

    def test_from_raw_drops_numpy_refusal_direction(self) -> None:
        import numpy as np

        from finetune_studio.training.export_response import ExportResult

        raw = {
            "ok": True,
            "status": "exported",
            "format": "abliterated",
            "output_dir": "/tmp/run/abliterated",
            "refusal_direction": np.zeros(16, dtype=np.float32),
            "refusal_magnitude": np.float64(0.42),
            "layers_modified": [28, 29, 30, 31],
            "strength": 1.0,
        }
        payload = ExportResult.from_raw(raw)
        dumped = payload.model_dump()
        # Must survive FastAPI's encoder (the live 500 path).
        encoded = jsonable_encoder(dumped)
        json.dumps(encoded)
        assert payload.ok is True
        assert payload.output_path == "/tmp/run/abliterated"
        assert "refusal_direction" not in dumped
        assert payload.refusal_magnitude == pytest.approx(0.42)
        assert payload.layers_modified == [28, 29, 30, 31]

    def test_sanitize_maps_output_dir(self) -> None:
        from finetune_studio.training.export_response import sanitize_export_dict

        cleaned = sanitize_export_dict(
            {"ok": True, "output_dir": "/x/abliterated", "format": "abliterated"}
        )
        assert cleaned["output_path"] == "/x/abliterated"

    def test_abliterated_export_trained_run_is_json_safe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import numpy as np

        from finetune_studio.training import run_export as re
        from finetune_studio.training.engine import TrainingEngine
        from finetune_studio.training.export_response import ExportResult

        out = tmp_path / "run"
        merged = out / "merged"
        merged.mkdir(parents=True)
        (merged / "model.safetensors").write_bytes(b"x" * 32)

        def _fake_abl(self: TrainingEngine) -> dict:
            abl = os.path.join(self.config.output_dir, "abliterated")
            os.makedirs(abl, exist_ok=True)
            (Path(abl) / "model.safetensors").write_bytes(b"y" * 32)
            return {
                "output_dir": abl,
                "refusal_direction": np.ones(8, dtype=np.float32),
                "refusal_magnitude": np.float32(1.25),
                "layers_modified": [1, 2],
                "strength": 1.0,
            }

        monkeypatch.setattr(TrainingEngine, "_do_abliteration", _fake_abl)
        run = {
            "id": "r1",
            "output_path": str(out),
            "base_model": "Qwen/Qwen3-0.6B",
            "status": "done",
        }
        raw = re.export_trained_run(run, fmt="abliterated", force=True)
        assert raw.get("ok") is True
        assert "refusal_direction" not in raw
        assert raw["output_path"].endswith("abliterated")
        payload = ExportResult.from_raw(raw)
        json.dumps(jsonable_encoder(payload.model_dump()))


class TestAbliteratedExportRoute:
    def test_post_abliterated_returns_200_json_and_registers(
        self, client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Route must JSON-encode success even when raw result has numpy."""
        import numpy as np

        from finetune_studio import db

        pid = client.post(
            "/api/projects", json={"name": "Abliterated Export"}
        ).json()["id"]
        out = tmp_path / "abl-run"
        abl = out / "abliterated"
        abl.mkdir(parents=True)
        (abl / "model.safetensors").write_bytes(b"a" * 64)
        created = db.create_run(
            project_id=pid,
            name="abl",
            base_model="Qwen/Qwen3-0.6B",
            data_path="/d",
        )
        db.update_run(
            created["id"], status="done", output_path=str(out)
        )

        # client fixture mocks TrainingEngine; stub the sync export body
        # with the historical bug shape (numpy refusal_direction).
        def _fake_export(run: dict, **_kwargs: object) -> dict:
            return {
                "ok": True,
                "status": "exported",
                "format": "abliterated",
                "output_dir": str(abl),
                "refusal_direction": np.zeros(4),
                "refusal_magnitude": 0.5,
                "layers_modified": [0, 1],
                "strength": 1.0,
            }

        monkeypatch.setattr(
            "finetune_studio.training.run_export.export_trained_run",
            _fake_export,
        )
        r = client.post(
            f"/api/projects/{pid}/runs/{created['id']}/export",
            json={"format": "abliterated", "force": True},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("ok") is True
        assert body.get("format") == "abliterated"
        assert body.get("output_path") == str(abl)
        assert "refusal_direction" not in body
        assert body.get("export_id")
        row = db.get_export(body["export_id"])
        assert row is not None
        assert row["status"] == "done"
        assert row["format"] == "abliterated"
        assert row["output_path"] == body["output_path"]

    def test_post_merged_registers_artifact(
        self, client, tmp_path: Path
    ) -> None:
        from finetune_studio import db

        pid = client.post(
            "/api/projects", json={"name": "Merged Export Reg"}
        ).json()["id"]
        out = tmp_path / "merged-run"
        merged = out / "merged"
        merged.mkdir(parents=True)
        (merged / "model.safetensors").write_bytes(b"z" * 48)
        (merged / "config.json").write_text("{}", encoding="utf-8")
        created = db.create_run(
            project_id=pid,
            name="m",
            base_model="Qwen/Qwen3-0.6B",
            data_path="/d",
        )
        db.update_run(
            created["id"], status="done", output_path=str(out)
        )
        r = client.post(
            f"/api/projects/{pid}/runs/{created['id']}/export",
            json={"format": "merged"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("ok") is True
        assert body.get("export_id")
        row = db.get_export(body["export_id"])
        assert row["status"] == "done"
        assert row["format"] == "merged"


class TestExportUiReadableErrors:
    def test_export_page_handles_non_json_and_detail(self) -> None:
        html = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "finetune_studio"
            / "webui"
            / "templates"
            / "export_models.html"
        ).read_text(encoding="utf-8")
        assert "non-JSON response" in html
        assert "d.detail" in html
        assert "output_path || d.merged_path || d.output_dir" in html
        assert "size_human" in html
        # GGUF failures stay truthful (prior regression).
        assert "convert_hf_to_gguf" in html or "without artifacts" in html
        assert "non-empty GGUF" in html or "without a non-empty GGUF" in html
