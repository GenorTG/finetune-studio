"""XLS (legacy Excel) parser."""

from __future__ import annotations

from pathlib import Path

from ._base import cli_run, make_result


def parse(path: Path) -> dict:
    try:
        import xlrd
    except ImportError as e:
        return make_result("", {"type": "xls", "error": str(e)}, parser="xls_v1",
                           warnings=["install xlrd (pip install xlrd)"])
    wb = xlrd.open_workbook(str(path))
    text_lines = []
    sheets = []
    for sheet in wb.sheets():
        rows = []
        for r in range(sheet.nrows):
            values = [str(sheet.cell_value(r, c)) for c in range(sheet.ncols)]
            if any(v.strip() for v in values):
                rows.append(values)
        sheets.append({"name": sheet.name, "row_count": len(rows), "rows": rows[:200]})
        text_lines.append(f"=== Sheet: {sheet.name} ===")
        for row in rows:
            text_lines.append(" | ".join(row))
    text = "\n".join(text_lines)
    structured = {"type": "xls", "sheet_count": len(sheets), "sheets": sheets}
    return make_result(text, structured, parser="xls_v1")


if __name__ == "__main__":
    cli_run(parse)
