"""Smart Brain passages keep a structured record whole when it fits (follow-up to #85).

Measured 2026-11-01 on the Quillmoor S xml/json/yaml sources with the multilingual-e5-large tokenizer at the store's
400/80-token defaults: 2 of the 20 XML records that fit a passage were cut across two passages; 0/20 after.
"""
from __future__ import annotations

from finetune_studio.smart_brain.chunking import chunk_rows, estimate_tokens


def _split(records: list[str], chunks: list[str]) -> int:
    return sum(not any(r in c for c in chunks) for r in records)


def _xml(i: int) -> str:
    return (f"<shipment>\n  <id>SH-{i:04d}</id>\n  <origin>Depot {i}</origin>\n  <destination>Customer site {i}</destination>\n"
            f"  <note>Temperature log {i} attached, reefer checked at gate {i % 7}</note>\n</shipment>")


def test_xml_records_are_never_cut_between_passages() -> None:
    records = [_xml(i) for i in range(40)]
    text = "<shipments>\n" + "\n".join("  " + r.replace("\n", "\n  ") for r in records) + "\n</shipments>"
    indented = ["  " + r.replace("\n", "\n  ") for r in records]
    chunks = chunk_rows(text, 120, 24)
    assert len(chunks) > 3
    assert _split(indented, chunks) == 0
    assert all(estimate_tokens(c) <= 120 + 8 for c in chunks)


def test_json_and_yaml_records_stay_whole() -> None:
    json_recs = [f'{{\n  "id": {i},\n  "name": "record {i}",\n  "value": "value number {i} for the test"\n}}' for i in range(30)]
    yaml_recs = [f"- id: {i}\n  name: record {i}\n  value: value number {i} for the test" for i in range(30)]
    assert _split(json_recs, chunk_rows(",\n".join(json_recs), 60, 10)) == 0
    assert _split(yaml_recs, chunk_rows("\n".join(yaml_recs), 60, 10)) == 0


def test_an_oversized_record_still_loses_nothing() -> None:
    body = " ".join(f"field{i}: value{i}." for i in range(150))
    text = f"<item>\n  {body}\n</item>\n<item>\n  short\n</item>"
    joined = "\n".join(chunk_rows(text, 80, 10))
    assert all(f"field{i}: value{i}." in joined for i in range(150))
    assert "short" in joined


def test_table_rows_and_markdown_bullets_keep_their_rules() -> None:
    rows = "name | city\n--- | ---\n" + "\n".join(f"Company {i} | City {i}" for i in range(60))
    chunks = chunk_rows(rows, 40, 8)
    assert all(c.startswith("name | city\n--- | ---") for c in chunks[1:])
    bullets = "\n".join(f"- ordinary bullet {i}" for i in range(30))
    assert "\n".join(chunk_rows(bullets, 400, 80)) == bullets
