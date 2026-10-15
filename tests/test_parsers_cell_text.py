"""Table cells read the way a person reads them: no ``.0`` on whole numbers, no newline splitting a row (found by reading the
mined pairs of the depot directory: "412.0 reefer plugs", "ext 2100.0", and a policy number cut off into the next chunk)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from finetune_studio.data.parsers import xls as xls_parser
from finetune_studio.data.parsers import xlsx as xlsx_parser
from finetune_studio.data.parsers._base import cell_text


@pytest.mark.parametrize("value,expected", [
    (412.0, "412"), (38.5, "38.5"), (0.05, "0.05"), (-3.0, "-3"), (None, ""), (True, "TRUE"),
    ("quote policy no.\nSB-KCC-7710-24", "quote policy no. SB-KCC-7710-24"), ("  a \t b ", "a b"),
    (dt.datetime(2024, 9, 17), "2024-09-17"), (dt.datetime(2024, 9, 17, 14, 5), "2024-09-17 14:05:00"),
])
def test_cell_text(value: object, expected: str) -> None:
    assert cell_text(value) == expected


def test_xls_numbers_and_multiline_cells(tmp_path: Path) -> None:
    import xlwt

    wb = xlwt.Workbook()
    ws = wb.add_sheet("Depots")
    for c, v in enumerate(["GDY-1", 412, 2100, "quote policy no.\nSB-KCC-7710-24"]):
        ws.write(0, c, v)
    path = tmp_path / "d.xls"
    wb.save(str(path))
    text = xls_parser.parse(path)["text"]
    assert "GDY-1 | 412 | 2100 | quote policy no. SB-KCC-7710-24" in text and "412.0" not in text


def test_xlsx_dates_and_multiline_cells(tmp_path: Path) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(["RC-01", 41.0, dt.datetime(2024, 3, 1), "line one\nline two"])
    path = tmp_path / "r.xlsx"
    wb.save(path)
    assert "RC-01 | 41 | 2024-03-01 | line one line two" in xlsx_parser.parse(path)["text"]
