"""XLSX (modern Excel) parser."""

from __future__ import annotations

from pathlib import Path

from ._base import cli_run, make_result


def parse(path: Path) -> dict:
    try:
        from openpyxl import load_workbook
    except ImportError as e:
        return make_result("", {"type": "xlsx", "error": str(e)}, parser="xlsx_v1",
                           warnings=["install openpyxl (pip install openpyxl)"])
    wb = load_workbook(str(path), data_only=True, read_only=True)
    text_lines = []
    sheets = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = []
        for row in ws.iter_rows(values_only=True):
            values = [str(v) if v is not None else "" for v in row]
            if any(v.strip() for v in values):
                rows.append(values)
        sheets.append({"name": sheet_name, "row_count": len(rows), "rows": rows[:200]})
        text_lines.append(f"=== Sheet: {sheet_name} ({len(rows)} rows) ===")
        for row in rows:
            text_lines.append(" | ".join(row))
    text = "\n".join(text_lines)
    structured = {
        "type": "xlsx",
        "sheet_count": len(sheets),
        "sheet_names": wb.sheetnames,
        "sheets": sheets,
    }
    return make_result(text, structured, parser="xlsx_v1")


if __name__ == "__main__":
    cli_run(parse)
