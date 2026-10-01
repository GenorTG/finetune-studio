"""Regression tests for webui/routes/quality.py (2026-10-01 audit fix).

All four of these routes imported classes/functions that never existed
under those names (or called them with the wrong constructor/method
shape), so every call silently returned ``status: "error"`` behind a
broad ``except Exception`` — found independently by two documentation-
audit lanes while reading training/ and benchmarks/compare/. The real
names/shapes are the same ones the equivalent ``fts`` CLI commands
already use correctly (cli/commands/{optimize,augment,
validate_hallucination,convert}.py).
"""

from __future__ import annotations

import json

SAMPLE_ROWS = [
    {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello there"}]},
    {"messages": [{"role": "user", "content": "bye"}, {"role": "assistant", "content": "goodbye"}]},
]


def _write_jsonl(tmp_path, name="training.jsonl"):
    p = tmp_path / name
    with p.open("w") as f:
        for row in SAMPLE_ROWS:
            f.write(json.dumps(row) + "\n")
    return p


class TestQualityRoutes:
    def test_optimize_uses_real_class_and_succeeds(self, client, mock_settings, tmp_path):
        p = _write_jsonl(tmp_path)
        r = client.post("/api/data/optimize", json={"path": str(p)})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok", body
        assert "recommendations" in body["result"]

    def test_augment_uses_real_class_and_succeeds(self, client, mock_settings, tmp_path):
        p = _write_jsonl(tmp_path)
        out = tmp_path / "out.jsonl"
        r = client.post(
            "/api/data/augment", json={"path": str(p), "output": str(out)},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok", body.get("error")
        assert body["result"]["output_count"] >= body["result"]["input_count"]
        assert out.exists()

    def test_hallucination_check_uses_real_class_and_succeeds(self, client, mock_settings, tmp_path):
        p = _write_jsonl(tmp_path)
        r = client.post("/api/data/hallucination-check", json={"path": str(p)})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok", body
        assert "total_risks" in body["result"]

    def test_convert_jsonl_to_json_uses_real_converter(self, client, mock_settings, tmp_path):
        p = _write_jsonl(tmp_path)
        r = client.post(
            "/api/data/convert",
            json={"path": str(p), "target_format": "json"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok", body
        out_path = body["result"]["output_path"]
        assert out_path.endswith(".json")
        with open(out_path) as f:
            data = json.load(f)
        assert len(data) == len(SAMPLE_ROWS)

    def test_convert_unsupported_pair_reports_error_not_crash(self, client, mock_settings, tmp_path):
        p = tmp_path / "x.txt"
        p.write_text("hello")
        r = client.post(
            "/api/data/convert",
            json={"path": str(p), "target_format": "jsonl"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "error"
        assert "Cannot convert" in body["error"]
