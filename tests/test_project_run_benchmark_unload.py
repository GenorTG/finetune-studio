"""Regression test (2026-10-01 audit fix): projects.py's run_benchmark route
loaded a second InferenceEngine without first freeing the shared, persistent
inference_engine's VRAM — unlike the three equivalent call sites in
webui/routes/benchmarks.py, which all call unload_all_models() first. This
is the exact GH-AAA mixed-GPU-load risk models/llama_loader.py's
no-mixed-offload contract exists to prevent: a model already resident via
the Testing tab stayed loaded while this route tried to load a second one.
"""

from __future__ import annotations

from unittest.mock import patch


class TestRunBenchmarkFreesSharedEngineFirst:
    def test_unload_all_models_called_before_engine_load(
        self, client, mock_settings, monkeypatch,
    ):
        from finetune_studio import db

        pid = db.create_project(name="P", description="")["id"]
        rid = db.create_run(
            project_id=pid, name="r", base_model="m", settings_obj={},
        )["id"]
        db.update_run(rid, output_path="/tmp/does-not-matter")

        calls: list[str] = []

        monkeypatch.setattr(
            "finetune_studio.models.llama_loader.unload_all_models",
            lambda: calls.append("unload_all_models"),
        )

        class FakeEngine:
            def load(self, path):
                calls.append(f"load:{path}")
                raise RuntimeError("stop before a real model load")

            def unload(self):
                calls.append("engine.unload")

        # The `client` fixture already has `testing.inference.InferenceEngine`
        # patched (to a bare MagicMock) for its whole lifetime, via a
        # `unittest.mock.patch` context manager that stays open until fixture
        # teardown. Re-patching the same target with `monkeypatch.setattr`
        # here would be unwound in the wrong order relative to that still-open
        # patch (monkeypatch's fixture tears down *before* `client`'s, since
        # `client` depends on it — LIFO), permanently leaving the attribute
        # pointed at the MagicMock for every test that runs afterward in the
        # same process. Using `patch(...)` as a `with`-scoped context manager
        # here instead guarantees it enters/exits strictly within this test
        # body, restoring `client`'s own MagicMock correctly on exit.
        with patch("finetune_studio.testing.inference.InferenceEngine", FakeEngine):
            r = client.post(
                f"/api/projects/{pid}/runs/{rid}/benchmark",
                json={"suite_name": "default", "suite_path": "data/benchmarks/default.json"},
            )
        assert r.status_code == 200
        body = r.json()
        assert "load failed" in body.get("error", ""), body

        assert calls[0] == "unload_all_models", (
            f"unload_all_models must run before engine.load(); got order {calls}"
        )
        assert any(c.startswith("load:") for c in calls)
