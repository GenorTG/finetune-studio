"""CSV / TSV parser."""

from __future__ import annotations

import csv
from pathlib import Path

from ._base import cli_run, make_result


def parse(path: Path) -> dict:
    # Sniff the dialect
    sample = path.read_text(encoding="utf-8", errors="replace")[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except csv.Error:
        # Fall back based on extension
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        class _D(csv.excel):  # type: ignore[misc]
            pass
        dialect = _D()
        dialect.delimiter = delimiter
    has_header = False
    try:
        has_header = csv.Sniffer().has_header(sample)
    except csv.Error:
        pass
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, dialect=dialect)
        rows = list(reader)
    headers = rows[0] if rows and has_header else None
    body = rows[1:] if has_header else rows
    # Plain text — markdown table for human reading; perfect for Q&A chunks
    lines = []
    if headers:
        lines.append(" | ".join(headers))
        lines.append(" | ".join(["---"] * len(headers)))
    for row in body:
        lines.append(" | ".join("" if v is None else str(v) for v in row))
    text = "\n".join(lines)
    structured = {
        "type": "csv",
        "delimiter": dialect.delimiter,
        "has_header": has_header,
        "headers": headers,
        "row_count": len(body),
        "column_count": len(headers) if headers else (max((len(r) for r in body), default=0)),
        "rows": [list(r) for r in body[:200]],  # sample
    }
    return make_result(text, structured, parser="csv_v1")


if __name__ == "__main__":
    cli_run(parse)
