"""Full-coverage suite rule (Genor 2026-09-20): the proof instrument tests
EVERY dataset row — N rows → N cases. Sampling is an explicit opt-in
(``max_cases`` / ``sample_size``) and must be labeled in the suite name, the
file metadata, and the audit so a sampled verdict can never masquerade as
full coverage.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from finetune_studio.data.audit import audit_suite_cases
from finetune_studio.testing.generate_suite import generate_suite_from_training_data
from finetune_studio.testing.suite import load_test_suite


def _dataset(tmp_path: Path, n: int, dup: bool = False) -> str:
    p = tmp_path / "train.jsonl"
    lines = []
    for i in range(n):
        q = f"Which sealed vault holds ledger {i % 3 if dup else i}?"
        lines.append(json.dumps({
            "conversations": [
                {"from": "human", "value": q},
                {"from": "gpt", "value": f"Vault {i} holds ledger {i}, certified in annal {i}."},
            ],
            "source_id": f"s{i % 4}",
            "chunk_idx": i,
        }))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


def test_default_is_full_coverage(tmp_path: Path) -> None:
    data = _dataset(tmp_path, 12)
    r = generate_suite_from_training_data(data, str(tmp_path / "out"))
    assert r["case_count"] == 12
    assert r["coverage"] == "full"
    assert r["dataset_count"] == 12
    doc = json.loads(Path(r["suite_path"]).read_text(encoding="utf-8"))
    assert doc["meta"]["coverage"] == "full"
    assert len(doc["cases"]) == 12
    audit = audit_suite_cases(r["suite_path"], data)
    assert audit["ok"] is True and audit["coverage"] == "full"


def test_sampling_is_explicit_deterministic_and_labeled(tmp_path: Path) -> None:
    data = _dataset(tmp_path, 12)
    r = generate_suite_from_training_data(data, str(tmp_path / "out"), max_cases=5)
    assert r["case_count"] == 5
    assert r["coverage"] == "sampled"
    assert "-sampled5of12" in r["suite_name"]
    doc = json.loads(Path(r["suite_path"]).read_text(encoding="utf-8"))
    idx1 = [c["row_index"] for c in doc["cases"]]
    assert sorted(idx1) != [0, 1, 2, 3, 4], "sampling must be uniform, not head-truncation"
    # deterministic: same seed → same rows
    r2 = generate_suite_from_training_data(data, str(tmp_path / "out2"), max_cases=5)
    doc2 = json.loads(Path(r2["suite_path"]).read_text(encoding="utf-8"))
    assert [c["row_index"] for c in doc2["cases"]] == idx1
    # audit accepts an honest sampled suite
    audit = audit_suite_cases(r["suite_path"], data)
    assert audit["ok"] is True and audit["coverage"] == "sampled"


def test_duplicate_questions_get_unique_names(tmp_path: Path) -> None:
    data = _dataset(tmp_path, 6, dup=True)
    r = generate_suite_from_training_data(data, str(tmp_path / "out"))
    assert r["case_count"] == 6
    audit = audit_suite_cases(r["suite_path"], data)
    assert audit["duplicate_names"] == []
    assert audit["ok"] is True


def test_row_index_roundtrips_through_loader(tmp_path: Path) -> None:
    data = _dataset(tmp_path, 4)
    r = generate_suite_from_training_data(data, str(tmp_path / "out"))
    cases = load_test_suite(r["suite_path"])
    assert [c.row_index for c in cases] == [0, 1, 2, 3]
    assert cases[2].source_id == "s2"


# ── route surface: POST /api/training/runs/{id}/auto-suites/generate ──


@pytest.fixture
def run_with_data(client, tmp_path: Path):
    from finetune_studio import db

    pid = client.post("/api/projects", json={"name": "CovSuite"}).json()["id"]
    data = _dataset(tmp_path, 9)
    out = tmp_path / "runout"
    out.mkdir()
    run = db.create_run(project_id=pid, name="cov", base_model="x", data_path=data,
                        rag_ids=[], settings_obj={})
    db.update_run(run["id"], output_path=str(out))
    return pid, run["id"], data


def test_route_default_full_coverage(client, run_with_data) -> None:
    _, rid, _ = run_with_data
    r = client.post(f"/api/training/runs/{rid}/auto-suites/generate", json={})
    body = r.json()
    assert body.get("ok") is True, body
    assert body["case_count"] == 9 and body["coverage"] == "full"


def test_route_sample_size_opt_in(client, run_with_data) -> None:
    _, rid, _ = run_with_data
    r = client.post(f"/api/training/runs/{rid}/auto-suites/generate",
                    json={"sample_size": 4})
    body = r.json()
    assert body.get("ok") is True, body
    assert body["case_count"] == 4 and body["coverage"] == "sampled"
    assert "-sampled4of9" in body["suite_name"]


def test_route_rejects_bad_sample_size(client, run_with_data) -> None:
    _, rid, _ = run_with_data
    assert "error" in client.post(f"/api/training/runs/{rid}/auto-suites/generate",
                                  json={"sample_size": 0}).json()
    assert "error" in client.post(f"/api/training/runs/{rid}/auto-suites/generate",
                                  json={"sample_size": "many"}).json()
