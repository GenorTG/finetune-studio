"""XLS (legacy Excel) parser."""

from __future__ import annotations

from pathlib import Path

from ._base import cell_text, cli_run, make_result
from .ooxml_charts import charts_in_package, sheet_resolver


def _charts(path: Path, grid: dict[str, list[list[str]]], warnings: list[str]) -> list[str]:
    """xlrd does not read charts: convert the workbook to xlsx with LibreOffice and read the chart parts there."""
    import tempfile

    from finetune_studio.data.render import RenderError, convert, soffice

    if not soffice():
        warnings.append("charts not read: LibreOffice is not installed")
        return []
    with tempfile.TemporaryDirectory(prefix="_xls_") as tmp:
        src = Path(tmp) / "book.xls"
        src.write_bytes(path.read_bytes())
        try:
            xlsx = convert(src, "xlsx", Path(tmp))
        except RenderError as e:
            warnings.append(f"charts not read: {e}")
            return []
        return charts_in_package(xlsx, sheet_resolver(grid))


def _cell(wb, sheet, r: int, c: int) -> str:
    import xlrd

    kind = sheet.cell_type(r, c)
    value = sheet.cell_value(r, c)
    if kind == xlrd.XL_CELL_DATE:
        try:
            return cell_text(xlrd.xldate.xldate_as_datetime(value, wb.datemode))
        except (ValueError, xlrd.xldate.XLDateError):
            pass
    if kind == xlrd.XL_CELL_BOOLEAN:
        return cell_text(bool(value))
    return cell_text(value)


def parse(path: Path) -> dict:
    try:
        import xlrd
    except ImportError as e:
        return make_result("", {"type": "xls", "error": str(e)}, parser="xls_v1",
                           warnings=["install xlrd (pip install xlrd)"])
    wb = xlrd.open_workbook(str(path))
    try:
        text_lines = []
        sheets = []
        grid: dict[str, list[list[str]]] = {}
        warnings: list[str] = []
        for sheet in wb.sheets():
            rows = []
            grid[sheet.name] = []
            for r in range(sheet.nrows):
                values = [_cell(wb, sheet, r, c) for c in range(sheet.ncols)]
                grid[sheet.name].append(values)  # by position, for chart references
                if any(v.strip() for v in values):
                    rows.append(values)
            sheets.append({"name": sheet.name, "row_count": len(rows), "rows": rows[:200]})
            text_lines.append(f"=== Sheet: {sheet.name} ===")
            for row in rows:
                text_lines.append(" | ".join(row))
        charts = _charts(path, grid, warnings)
        if charts:
            text_lines.append("=== Charts ===")
            text_lines.extend(charts)
        text = "\n".join(text_lines)
        structured = {"type": "xls", "sheet_count": len(sheets), "sheets": sheets, "chart_count": len(charts)}
    finally:
        wb.release_resources()
    return make_result(text, structured, parser="xls_v1", warnings=warnings)


if __name__ == "__main__":
    cli_run(parse)
