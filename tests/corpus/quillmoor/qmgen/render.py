"""Format writers: lay a ``Doc`` out as a real file of its extension.

Writers never change a block's text: every string in ``doc.blocks`` is written once, whole, exactly as given (no smart
quotes, no hyphenation, no number/date reformatting, nothing split inside a token). What each format keeps is stated by
``kept_text(doc)``: csv/tsv keep only the first table, every other format keeps every block (an email also keeps its
headers). Page-like formats wrap long lines, so compare against parsed text with whitespace collapsed
(``norm_ws``) -- a line break inside "net 47 days" is still the same phrase.

Direct writers (text formats, docx, xlsx, xls, pptx, rtf, eml, epub, images, scanned pdf) run in a process pool;
pdf/doc/odt/ods/odp are written as docx/xlsx/pptx and converted by LibreOffice in batches (one private profile per
soffice call, so it never meets another LibreOffice on the host, e.g. the live app's).
"""
from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import math
import os
import queue
import random
import re
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

from .model import Block, Doc

TEXT_EXTS = frozenset({".txt", ".md", ".markdown", ".log", ".py", ".js", ".ts", ".jsx", ".tsx", ".css", ".yaml", ".yml",
                       ".ini", ".cfg", ".conf", ".csv", ".tsv", ".json", ".jsonl", ".xml", ".html", ".htm"})
IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"})
DIRECT_BINARY = frozenset({".rtf", ".docx", ".xlsx", ".xls", ".pptx", ".eml", ".epub"})
# target ext -> (source ext written first, LibreOffice --convert-to filter)
LO_TARGETS: dict[str, tuple[str, str]] = {
    ".pdf": (".docx", "pdf:writer_pdf_Export"),
    ".doc": (".docx", "doc:MS Word 97"),
    ".odt": (".docx", "odt"),
    ".ods": (".xlsx", "ods"),
    ".odp": (".pptx", "odp"),
}
ALL_EXTS = TEXT_EXTS | IMAGE_EXTS | DIRECT_BINARY | frozenset(LO_TARGETS)
OCR_EXTS = IMAGE_EXTS  # plus any .pdf with meta scan=True

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
SOFFICE = shutil.which("soffice") or "/usr/bin/soffice"
LO_CHUNK = 40
LO_TIMEOUT = 600


def is_ocr(doc: Doc) -> bool:
    return doc.ext.lower() in IMAGE_EXTS or (doc.ext.lower() == ".pdf" and bool(doc.meta.get("scan")))


def norm_ws(s: str) -> str:
    return " ".join(s.split())


def _seed(doc: Doc) -> int:
    return int.from_bytes(hashlib.sha256(doc.doc_id.encode()).digest()[:8], "big")


def _first_table(doc: Doc) -> Block | None:
    return next((b for b in doc.blocks if b.kind == "table"), None)


def kept_text(doc: Doc) -> str:
    """The text this doc's format promises to carry (what the benchmark may expect after parsing)."""
    ext = doc.ext.lower()
    if doc.raw is not None:
        return doc.raw
    if ext in (".csv", ".tsv"):
        t = _first_table(doc)
        if t is None:
            return ""
        return "\n".join([" | ".join(t.columns)] + [" | ".join(r) for r in t.rows])
    out = doc.plain_text()
    if ext == ".eml" and doc.meta.get("email"):
        e = doc.meta["email"]
        out = "\n".join([*(str(e.get(k, "")) for k in ("from", "to", "cc", "subject") if e.get(k)), out])
    return out


# ----------------------------------------------------------------------------------------------------- flat text
def _table_lines(b: Block, pad: bool = True) -> list[str]:
    rows = [b.columns, *b.rows]
    ncol = max((len(r) for r in rows), default=0)
    rows = [list(r) + [""] * (ncol - len(r)) for r in rows]
    if pad:
        widths = [max(len(r[i]) for r in rows) for i in range(ncol)]
        lines = [" | ".join(c.ljust(widths[i]) for i, c in enumerate(r)).rstrip() for r in rows]
        sep = "-+-".join("-" * w for w in widths)
        return [lines[0], sep, *lines[1:]]
    return [" | ".join(r) for r in rows]


def _plain_lines(doc: Doc) -> list[str]:
    out = [doc.title, "=" * max(3, len(doc.title)), ""]
    for b in doc.blocks:
        k = b.kind
        if k == "heading":
            out += [b.text, ("=" if b.level == 1 else "-") * max(3, len(b.text)), ""]
        elif k == "para":
            out += [b.text, ""]
        elif k == "bullets":
            out += [f"- {i}" for i in b.items] + [""]
        elif k == "table":
            if b.caption:
                out.append(b.caption)
            out += _table_lines(b) + [""]
        elif k == "kv":
            out += [f"{lab}: {val}" for lab, val in b.pairs] + [""]
        elif k == "quote":
            out += [f"> {ln}" for ln in b.text.split("\n")] + [""]
        elif k == "note":
            out += [f"Note -- {b.text}", ""]
        elif k == "pagebreak":
            out += ["", ""]
    for key in ("footer",):
        if doc.meta.get(key):
            out.append(str(doc.meta[key]))
    return out


def _txt(doc: Doc) -> str:
    return "\n".join(_plain_lines(doc)).rstrip() + "\n"


def _md(doc: Doc) -> str:
    out = [f"# {doc.title}", ""]
    for b in doc.blocks:
        k = b.kind
        if k == "heading":
            out += [f"{'#' * min(6, b.level + 1)} {b.text}", ""]
        elif k == "para":
            out += [b.text, ""]
        elif k == "bullets":
            out += [f"- {i}" for i in b.items] + [""]
        elif k == "table":
            if b.caption:
                out += [f"**{b.caption}**", ""]
            out.append("| " + " | ".join(b.columns) + " |")
            out.append("|" + "|".join("---" for _ in b.columns) + "|")
            out += ["| " + " | ".join(r) + " |" for r in b.rows]
            out.append("")
        elif k == "kv":
            out += [f"- **{lab}:** {val}" for lab, val in b.pairs] + [""]
        elif k == "quote":
            out += [f"> {ln}" for ln in b.text.split("\n")] + [""]
        elif k == "note":
            out += [f"_Note: {b.text}_", ""]
        elif k == "pagebreak":
            out += ["---", ""]
    if doc.meta.get("footer"):
        out += ["---", "", str(doc.meta["footer"])]
    return "\n".join(out).rstrip() + "\n"


def _html(doc: Doc) -> str:
    e = html.escape
    parts = ["<!DOCTYPE html>", '<html lang="en">', "<head>", '<meta charset="utf-8">', f"<title>{e(doc.title)}</title>",
             ("<style>body{font-family:sans-serif;max-width:60em;margin:2em auto;line-height:1.45}"
              "table{border-collapse:collapse;margin:1em 0}th,td{border:1px solid #999;padding:.25em .6em;text-align:left}"
              "th{background:#eee}blockquote{color:#444;border-left:3px solid #ccc;margin-left:0;padding-left:1em}"
              ".note{font-size:.85em;color:#555}</style>"), "</head>", "<body>", "<main>", f"<h1>{e(doc.title)}</h1>"]
    for b in doc.blocks:
        k = b.kind
        if k == "heading":
            lv = min(6, b.level + 1)
            parts.append(f"<h{lv}>{e(b.text)}</h{lv}>")
        elif k == "para":
            parts.append(f"<p>{e(b.text)}</p>")
        elif k == "bullets":
            parts.append("<ul>" + "".join(f"<li>{e(i)}</li>" for i in b.items) + "</ul>")
        elif k == "table":
            t = ["<table>"]
            if b.caption:
                t.append(f"<caption>{e(b.caption)}</caption>")
            t.append("<thead><tr>" + "".join(f"<th>{e(c)}</th>" for c in b.columns) + "</tr></thead><tbody>")
            t += ["<tr>" + "".join(f"<td>{e(c)}</td>" for c in r) + "</tr>" for r in b.rows]
            t.append("</tbody></table>")
            parts.append("\n".join(t))
        elif k == "kv":
            parts.append("<dl>" + "".join(f"<dt>{e(lab)}</dt><dd>{e(val)}</dd>" for lab, val in b.pairs) + "</dl>")
        elif k == "quote":
            parts.append("<blockquote>" + "".join(f"<p>{e(ln)}</p>" for ln in b.text.split("\n")) + "</blockquote>")
        elif k == "note":
            parts.append(f'<p class="note">{e(b.text)}</p>')
        elif k == "pagebreak":
            parts.append("<hr>")
    parts.append("</main>")
    if doc.meta.get("footer"):
        parts.append(f"<footer><p>{e(str(doc.meta['footer']))}</p></footer>")
    parts += ["</body>", "</html>"]
    return "\n".join(parts) + "\n"


def _csv(doc: Doc, delim: str) -> str:
    t = _first_table(doc)
    if t is None:
        raise ValueError("csv/tsv needs a table block")
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=delim, quoting=csv.QUOTE_ALL, lineterminator="\n")
    w.writerow(t.columns)
    w.writerows(t.rows)
    return buf.getvalue()


def _row_obj(columns: list[str], row: list[str]) -> dict[str, str]:
    keys: list[str] = []
    for i, c in enumerate(columns):
        k = c or f"column {i + 1}"
        n = 2
        base = k
        while k in keys:
            k = f"{base} ({n})"
            n += 1
        keys.append(k)
    keys += [f"column {i + 1}" for i in range(len(keys), len(row))]
    return {k: (row[i] if i < len(row) else "") for i, k in enumerate(keys)}


def _block_obj(b: Block) -> dict:
    if b.kind == "heading":
        return {"type": "heading", "level": b.level, "text": b.text}
    if b.kind == "bullets":
        return {"type": "bullets", "items": list(b.items)}
    if b.kind == "table":
        return {"type": "table", "caption": b.caption, "columns": list(b.columns),
                "rows": [_row_obj(b.columns, r) for r in b.rows]}
    if b.kind == "kv":
        return {"type": "fields", "fields": [{"label": k, "value": v} for k, v in b.pairs]}
    if b.kind == "pagebreak":
        return {"type": "pagebreak"}
    return {"type": b.kind, "text": b.text}


def _json(doc: Doc) -> str:
    obj: dict = {"title": doc.title}
    for k in ("author", "date", "company"):
        if doc.meta.get(k):
            obj[k] = doc.meta[k]
    obj["blocks"] = [_block_obj(b) for b in doc.blocks]
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


def _jsonl(doc: Doc) -> str:
    # One record per table row; any non-table block is a record too, so nothing the doc says is dropped.
    lines = []
    for b in doc.blocks:
        if b.kind == "table":
            for r in b.rows:
                rec = _row_obj(b.columns, r)
                rec["_table"] = b.caption
                lines.append(json.dumps(rec, ensure_ascii=False))
        elif b.kind != "pagebreak":
            rec = _block_obj(b)
            lines.append(json.dumps({"_block": rec.pop("type"), **rec}, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def _xml(doc: Doc) -> str:
    root = ET.Element("document", {"id": doc.doc_id})
    ET.SubElement(root, "title").text = doc.title
    for k in ("author", "date", "company"):
        if doc.meta.get(k):
            ET.SubElement(root, k).text = str(doc.meta[k])
    sec = ET.SubElement(root, "section")
    for b in doc.blocks:
        k = b.kind
        if k == "heading":
            if b.level == 1 and len(sec):
                sec = ET.SubElement(root, "section")
            ET.SubElement(sec, "heading", {"level": str(b.level)}).text = b.text
        elif k in ("para", "quote", "note"):
            ET.SubElement(sec, k).text = b.text
        elif k == "bullets":
            lst = ET.SubElement(sec, "list")
            for i in b.items:
                ET.SubElement(lst, "item").text = i
        elif k == "table":
            t = ET.SubElement(sec, "table", {"caption": b.caption} if b.caption else {})
            for r in b.rows:
                row = ET.SubElement(t, "row")
                for name, val in _row_obj(b.columns, r).items():
                    ET.SubElement(row, "cell", {"name": name}).text = val
        elif k == "kv":
            f = ET.SubElement(sec, "fields")
            for lab, val in b.pairs:
                fe = ET.SubElement(f, "field")
                ET.SubElement(fe, "label").text = lab
                ET.SubElement(fe, "value").text = val
        elif k == "pagebreak":
            ET.SubElement(sec, "pagebreak")
    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def _yaml(doc: Doc) -> str:
    import yaml

    class _Dumper(yaml.SafeDumper):
        pass

    def _str(dumper: yaml.SafeDumper, s: str):  # exact strings: block scalar for multi-line, never '' doubling
        if "\n" in s:
            return dumper.represent_scalar("tag:yaml.org,2002:str", s, style="|")
        if "'" in s:
            return dumper.represent_scalar("tag:yaml.org,2002:str", s, style='"')
        return dumper.represent_scalar("tag:yaml.org,2002:str", s)

    _Dumper.add_representer(str, _str)
    sections: list[dict] = []
    cur: dict = {"heading": None, "content": []}
    for b in doc.blocks:
        if b.kind == "heading" and b.level == 1:
            if cur["heading"] is not None or cur["content"]:
                sections.append(cur)
            cur = {"heading": b.text, "content": []}
            continue
        if b.kind == "pagebreak":
            continue
        if b.kind == "heading":
            cur["content"].append({"subheading": b.text})
        elif b.kind == "table":
            cur["content"].append({"table": b.caption, "rows": [_row_obj(b.columns, r) for r in b.rows]})
        elif b.kind == "kv":
            cur["content"].append({"fields": [{lab: val} for lab, val in b.pairs]})
        elif b.kind == "bullets":
            cur["content"].append({"list": list(b.items)})
        else:
            cur["content"].append({b.kind: b.text})
    if cur["heading"] is not None or cur["content"]:
        sections.append(cur)
    data = {"title": doc.title, **{k: doc.meta[k] for k in ("author", "date", "company") if doc.meta.get(k)},
            "sections": sections}
    return yaml.dump(data, Dumper=_Dumper, allow_unicode=True, default_flow_style=False, width=10**6, sort_keys=False)


_COMMENT = {".py": "# ", ".js": "// ", ".ts": "// ", ".jsx": "// ", ".tsx": "// ", ".ini": "; ", ".cfg": "# ",
            ".conf": "# "}


def _commented(doc: Doc, ext: str) -> str:
    lines = _plain_lines(doc)
    if ext == ".css":
        return "/*\n" + "\n".join(f" * {ln}".rstrip() for ln in lines) + "\n */\n"
    if ext == ".log":
        stamp = str(doc.meta.get("date") or "2020-01-01")[:10]
        return "\n".join(f"{stamp} 00:00:{i % 60:02d} INFO {ln}" for i, ln in enumerate(ln for ln in lines if ln)) + "\n"
    c = _COMMENT[ext]
    return "\n".join(f"{c}{ln}".rstrip() for ln in lines) + "\n"


def _text_native(doc: Doc, ext: str) -> str:
    if ext in (".txt",):
        return _txt(doc)
    if ext in (".md", ".markdown"):
        return _md(doc)
    if ext in (".html", ".htm"):
        return _html(doc)
    if ext == ".csv":
        return _csv(doc, ",")
    if ext == ".tsv":
        return _csv(doc, "\t")
    if ext == ".json":
        return _json(doc)
    if ext == ".jsonl":
        return _jsonl(doc)
    if ext == ".xml":
        return _xml(doc)
    if ext in (".yaml", ".yml"):
        return _yaml(doc)
    return _commented(doc, ext)


# ------------------------------------------------------------------------------------------------------------ docx
def _doc_date(doc: Doc) -> datetime:
    d = doc.meta.get("date")
    if d:
        try:
            return datetime.fromisoformat(str(d)[:19])
        except ValueError:
            pass
    return datetime(2020, 1, 1)  # noqa: DTZ001 - document dates are local, naive like the meta ISO dates


def write_docx(doc: Doc, path: Path, *, for_pdf: bool = False) -> None:
    from docx import Document
    from docx.enum.text import WD_BREAK
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt

    d = Document()
    normal = d.styles["Normal"]
    normal.font.name = "DejaVu Sans"
    normal.font.size = Pt(10)
    rpr = normal.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(a), "DejaVu Sans")
    for sname in ("Title", "Heading 1", "Heading 2", "Heading 3"):
        st = d.styles[sname]
        st.font.name = "DejaVu Sans"
    sec = d.sections[0]
    sec.left_margin = sec.right_margin = Inches(0.8)
    if doc.meta.get("header"):
        sec.header.paragraphs[0].text = str(doc.meta["header"])
    if doc.meta.get("footer"):
        sec.footer.paragraphs[0].text = str(doc.meta["footer"])
    cp = d.core_properties
    cp.title = doc.title
    cp.author = str(doc.meta.get("author") or "")
    cp.created = cp.modified = _doc_date(doc)
    cp.last_modified_by = cp.author

    d.add_heading(doc.title, 0)

    def table(columns: list[str], rows: list[list[str]], header: bool = True) -> None:
        ncol = max([len(columns)] + [len(r) for r in rows]) or 1
        allrows = ([columns] if header else []) + rows
        t = d.add_table(rows=len(allrows), cols=ncol)
        t.style = "Table Grid"
        for ri, r in enumerate(allrows):
            cells = t.rows[ri].cells
            for ci in range(ncol):
                val = r[ci] if ci < len(r) else ""
                p = cells[ci].paragraphs[0]
                run = p.add_run(val)
                if header and ri == 0:
                    run.bold = True
        d.add_paragraph()

    for b in doc.blocks:
        k = b.kind
        if k == "heading":
            d.add_heading(b.text, min(3, max(1, b.level)))
        elif k == "para":
            d.add_paragraph(b.text)
        elif k == "bullets":
            for i in b.items:
                d.add_paragraph(i, style="List Bullet")
        elif k == "table":
            if b.caption:
                d.add_paragraph().add_run(b.caption).bold = True
            table(b.columns, b.rows)
        elif k == "kv":
            table([], [[lab, val] for lab, val in b.pairs], header=False)
        elif k == "quote":
            p = d.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.5)
            p.add_run(b.text).italic = True
        elif k == "note":
            p = d.add_paragraph()
            p.add_run(b.text).font.size = Pt(8)
        elif k == "pagebreak":
            d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    d.save(str(path))


# ------------------------------------------------------------------------------------------------------ xlsx / xls
_SHEET_BAD = re.compile(r"[\[\]:*?/\\]")


def _sheet_names(n_needed: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for i, cap in enumerate(n_needed):
        base = _SHEET_BAD.sub(" ", cap or "").strip().strip("'").strip() or f"Table {i + 1}"
        base = " ".join(base.split())[:31].strip() or f"Table {i + 1}"
        name, n = base, 2
        while name.lower() in seen:
            suf = f" ({n})"
            name = base[: 31 - len(suf)].rstrip() + suf
            n += 1
        seen.add(name.lower())
        out.append(name)
    return out


def _cover_rows(doc: Doc) -> list[tuple[list[str], bool]]:
    rows: list[tuple[list[str], bool]] = []
    for b in doc.blocks:
        k = b.kind
        if k == "heading":
            rows.append(([b.text], True))
        elif k in ("para", "quote", "note"):
            rows.append(([b.text], False))
        elif k == "bullets":
            rows += [(["- " + i], False) for i in b.items]
        elif k == "kv":
            rows += [([lab, val], False) for lab, val in b.pairs]
    return rows


def _workbook_plan(doc: Doc) -> list[tuple[str, list[tuple[list[str], bool]]]]:
    """[(sheet name, [(cells, bold)])] -- shared by the xlsx and xls writers."""
    cover = _cover_rows(doc)
    tables = [b for b in doc.blocks if b.kind == "table"]
    sheets: list[tuple[str, list[tuple[list[str], bool]]]] = []
    names = _sheet_names((["Cover"] if cover or not tables else []) + [t.caption for t in tables])
    ni = 0
    if cover or not tables:
        sheets.append((names[0], [([doc.title], True), ([""], False), *cover]))
        ni = 1
    for t in tables:
        rows: list[tuple[list[str], bool]] = []
        if doc.meta.get("title_row"):
            rows.append(([doc.title], True))
        rows.append((list(t.columns), True))
        rows += [(list(r), False) for r in t.rows]
        sheets.append((names[ni], rows))
        ni += 1
    return sheets


def write_xlsx(doc: Doc, path: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    wb.remove(wb.active)
    wb.properties.title = doc.title
    wb.properties.creator = str(doc.meta.get("author") or "")
    wb.properties.created = wb.properties.modified = _doc_date(doc)
    bold = Font(bold=True)
    for name, rows in _workbook_plan(doc):
        ws = wb.create_sheet(name)
        widths: dict[int, int] = {}
        for ri, (cells, is_bold) in enumerate(rows, 1):
            for ci, val in enumerate(cells, 1):
                if val == "":
                    continue
                c = ws.cell(row=ri, column=ci)
                c.value = val
                c.data_type = "s"  # text stays text: "=x", "1,250.00", "03.04.2019" are never reinterpreted
                c.number_format = "@"
                if is_bold:
                    c.font = bold
                widths[ci] = max(widths.get(ci, 8), min(60, len(val) + 2))
        for ci, w in widths.items():
            ws.column_dimensions[get_column_letter(ci)].width = w
    wb.save(str(path))


def write_xls(doc: Doc, path: Path) -> None:
    import xlwt

    wb = xlwt.Workbook(encoding="utf-8")
    bold = xlwt.easyxf("font: bold on")
    plain = xlwt.easyxf("", num_format_str="@")
    for name, rows in _workbook_plan(doc):
        ws = wb.add_sheet(name)
        widths: dict[int, int] = {}
        for ri, (cells, is_bold) in enumerate(rows):
            for ci, val in enumerate(cells):
                if val == "":
                    continue
                ws.write(ri, ci, val[:32767], bold if is_bold else plain)
                widths[ci] = max(widths.get(ci, 8), min(60, len(val) + 2))
        for ci, w in widths.items():
            ws.col(ci).width = 256 * w
    wb.save(str(path))


# ------------------------------------------------------------------------------------------------------------ pptx
def write_pptx(doc: Doc, path: Path) -> None:
    from pptx import Presentation
    from pptx.oxml.ns import qn
    from pptx.util import Emu, Inches, Pt

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    prs.core_properties.title = doc.title
    prs.core_properties.author = str(doc.meta.get("author") or "")
    W, H = prs.slide_width, prs.slide_height
    left, width = Inches(0.6), W - Inches(1.2)
    top0, bottom = Inches(1.5), H - Inches(0.4)
    chars_per_line = 120  # 16 pt over ~12 in

    title_slide = prs.slides.add_slide(prs.slide_layouts[0])
    title_slide.shapes.title.text = doc.title
    sub = " - ".join(str(doc.meta[k]) for k in ("company", "author", "date") if doc.meta.get(k))
    if len(title_slide.placeholders) > 1:
        title_slide.placeholders[1].text = sub
    state = {"slide": title_slide, "y": Inches(4.8), "title": doc.title, "notes": []}

    def flush_notes() -> None:
        if state["notes"]:
            state["slide"].notes_slide.notes_text_frame.text = "\n".join(state["notes"])
            state["notes"] = []

    def new_slide(title: str) -> None:
        flush_notes()
        s = prs.slides.add_slide(prs.slide_layouts[5])
        s.shapes.title.text = title
        state.update(slide=s, y=top0, title=title)

    def room(h: int) -> None:
        if state["y"] + h > bottom and state["y"] > top0:
            new_slide(state["title"] + " (continued)")

    def text_box(paras: list[str], size: int = 16, italic: bool = False, bullet: bool = False) -> None:
        lines = sum(max(1, math.ceil(len(p) * 16 / size / chars_per_line)) for p in paras)
        h = Emu(int(Pt(size * 1.35) * lines + Inches(0.2)))
        room(h)
        tb = state["slide"].shapes.add_textbox(left, state["y"], width, h)
        tf = tb.text_frame
        tf.word_wrap = True
        for i, ptxt in enumerate(paras):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            r = p.add_run()
            r.text = ptxt
            r.font.size = Pt(size)
            r.font.italic = italic
            if bullet:
                pPr = p._p.get_or_add_pPr()
                pPr.set("marL", "342900")
                pPr.set("indent", "-342900")
                bu = pPr.makeelement(qn("a:buChar"), {"char": "•"})
                pPr.append(bu)
        state["y"] += h + Inches(0.1)

    def table(columns: list[str], rows: list[list[str]], header: bool = True) -> None:
        ncol = max([len(columns)] + [len(r) for r in rows]) or 1
        chunks = [rows[i:i + 12] for i in range(0, len(rows), 12)] or [[]]
        for chunk in chunks:
            allrows = ([columns] if header else []) + chunk
            rh = Inches(0.4)
            h = Emu(int(rh * len(allrows)))
            room(h)
            if state["y"] > top0 and chunk is not chunks[0]:
                new_slide(state["title"] + " (continued)")
            gf = state["slide"].shapes.add_table(len(allrows), ncol, left, state["y"], width, h)
            tbl = gf.table
            for ri, r in enumerate(allrows):
                for ci in range(ncol):
                    cell = tbl.cell(ri, ci)
                    cell.text = r[ci] if ci < len(r) else ""
                    for p in cell.text_frame.paragraphs:
                        for run in p.runs:
                            run.font.size = Pt(12)
                            run.font.bold = bool(header and ri == 0)
            state["y"] += h + Inches(0.2)

    for b in doc.blocks:
        k = b.kind
        if k == "heading" and b.level == 1:
            new_slide(b.text)
        elif k == "pagebreak":
            new_slide(state["title"])
        elif k == "heading":
            text_box([b.text], size=20)
        elif k == "para":
            text_box([b.text])
        elif k == "bullets":
            text_box(list(b.items), bullet=True)
        elif k == "quote":
            text_box(b.text.split("\n"), italic=True)
        elif k == "note":
            state["notes"].append(b.text)
        elif k == "table":
            if b.caption:
                text_box([b.caption], size=14)
            table(b.columns, b.rows)
        elif k == "kv":
            table([], [[lab, val] for lab, val in b.pairs], header=False)
    flush_notes()
    prs.save(str(path))


# ------------------------------------------------------------------------------------------------------------- rtf
def _rtf_esc(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if ch in "\\{}":
            out.append("\\" + ch)
        elif ch == "\n":
            out.append("\\line ")
        elif ch == "\t":
            out.append("\\tab ")
        elif o < 128:
            out.append(ch)
        else:
            for unit in (ch.encode("utf-16-le")[i:i + 2] for i in range(0, len(ch.encode("utf-16-le")), 2)):
                n = int.from_bytes(unit, "little")
                out.append(f"\\u{n - 65536 if n > 32767 else n}?")
    return "".join(out)


def write_rtf(doc: Doc, path: Path) -> None:
    e = _rtf_esc
    o = [r"{\rtf1\ansi\ansicpg1252\deff0\uc1{\fonttbl{\f0\fswiss DejaVu Sans;}{\f1\fmodern DejaVu Sans Mono;}}",
         r"{\info{\title " + e(doc.title) + "}" + (r"{\author " + e(str(doc.meta["author"])) + "}" if doc.meta.get("author")
                                                   else "") + "}",
         r"\f0\fs20", r"\pard\b\fs36 " + e(doc.title) + r"\b0\fs20\par"]

    def table(rows: list[list[str]], header: bool) -> None:
        ncol = max((len(r) for r in rows), default=1) or 1
        cw = 9600 // ncol
        for ri, r in enumerate(rows):
            o.append(r"\trowd\trgaph108" + "".join(rf"\clbrdrt\brdrs\clbrdrl\brdrs\clbrdrb\brdrs\clbrdrr\brdrs\cellx{cw * (i + 1)}"
                                                   for i in range(ncol)))
            cells = []
            for ci in range(ncol):
                v = e(r[ci] if ci < len(r) else "")
                cells.append(r"\pard\intbl " + (r"\b " + v + r"\b0" if header and ri == 0 else v) + r"\cell")
            o.append(" ".join(cells) + r"\row")
        o.append(r"\pard\par")

    for b in doc.blocks:
        k = b.kind
        if k == "heading":
            size = {1: 30, 2: 26}.get(b.level, 22)
            o.append(rf"\pard\sb120\b\fs{size} " + e(b.text) + r"\b0\fs20\par")
        elif k == "para":
            o.append(r"\pard " + e(b.text) + r"\par")
        elif k == "bullets":
            o += [r"\pard\li360\fi-240 \bullet  " + e(i) + r"\par" for i in b.items]
        elif k == "table":
            if b.caption:
                o.append(r"\pard\b " + e(b.caption) + r"\b0\par")
            table([b.columns, *b.rows], True)
        elif k == "kv":
            o += [r"\pard " + e(lab) + ": " + e(val) + r"\par" for lab, val in b.pairs]
        elif k == "quote":
            o.append(r"\pard\li720\i " + e(b.text) + r"\i0\par")
        elif k == "note":
            o.append(r"\pard\fs16 " + e(b.text) + r"\fs20\par")
        elif k == "pagebreak":
            o.append(r"\page")
    o.append("}")
    path.write_text("\n".join(o) + "\n", encoding="ascii")


# ------------------------------------------------------------------------------------------------------------- eml
def write_eml(doc: Doc, path: Path) -> None:
    import email.policy
    from email.message import EmailMessage
    from email.utils import format_datetime

    m = doc.meta.get("email") or {}
    msg = EmailMessage(policy=email.policy.SMTP)
    msg["From"] = m.get("from") or "noreply@example.invalid"
    msg["To"] = m.get("to") or "undisclosed-recipients:;"
    if m.get("cc"):
        msg["Cc"] = m["cc"]
    msg["Subject"] = m.get("subject") or doc.title
    msg["Date"] = m.get("date") or format_datetime(_doc_date(doc))
    msg["Message-ID"] = m.get("message_id") or f"<{doc.doc_id}@quillmoor.invalid>"
    if m.get("in_reply_to"):
        msg["In-Reply-To"] = m["in_reply_to"]
        msg["References"] = m["in_reply_to"]
    lines: list[str] = []
    for b in doc.blocks:
        k = b.kind
        if k in ("heading", "para", "note"):
            lines += [b.text, ""]
        elif k == "bullets":
            lines += [f"- {i}" for i in b.items] + [""]
        elif k == "table":
            if b.caption:
                lines.append(b.caption)
            lines += _table_lines(b, pad=False) + [""]
        elif k == "kv":
            lines += [f"{lab}: {val}" for lab, val in b.pairs] + [""]
        elif k == "quote":
            lines += [f"> {ln}" for ln in b.text.split("\n")] + [""]
    plain = "\n".join(lines).rstrip() + "\n"
    msg.set_content(plain, charset="utf-8", cte="quoted-printable")
    body_html = _html(doc).split("<main>", 1)[1].split("</main>", 1)[0]
    body_html = re.sub(r"<h1>.*?</h1>", "", body_html, count=1, flags=re.DOTALL)  # the subject is not repeated as a heading
    msg.add_alternative(f'<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>{body_html}</body></html>\n',
                        subtype="html", charset="utf-8", cte="quoted-printable")
    msg.set_boundary(f"qm-{hashlib.sha256(doc.doc_id.encode()).hexdigest()[:24]}")  # deterministic MIME boundary
    path.write_bytes(msg.as_bytes())


# ------------------------------------------------------------------------------------------------------------ epub
def write_epub(doc: Doc, path: Path) -> None:
    e = html.escape
    chapters: list[tuple[str, list[Block]]] = []
    cur_title, cur = doc.title, []
    for b in doc.blocks:
        if b.kind == "heading" and b.level == 1:
            if cur or not chapters:
                chapters.append((cur_title, cur))
            cur_title, cur = b.text, []
        else:
            cur.append(b)
    chapters.append((cur_title, cur))
    uid = f"urn:uuid:{uuid.UUID(bytes=hashlib.sha256(doc.doc_id.encode()).digest()[:16])}"
    lang = "en"

    def xhtml(title: str, blocks: list[Block], first: bool) -> str:
        sub = Doc(doc_id=doc.doc_id, kind=doc.kind, ext=".html", title=title, blocks=blocks)
        body = _html(sub).split("<main>", 1)[1].split("</main>", 1)[0]
        body = body.replace("<hr>", "<hr/>")
        if first and title != doc.title:
            body = f"<h1>{e(doc.title)}</h1>\n" + body
        return ('<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n'
                f'<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="{lang}" '
                f'xml:lang="{lang}"><head><meta charset="utf-8"/><title>{e(title)}</title></head><body><section>'
                f"{body}</section></body></html>\n")

    files = [(f"text/ch{i + 1:03d}.xhtml", t, xhtml(t, bl, i == 0)) for i, (t, bl) in enumerate(chapters)]
    nav = ('<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n<html xmlns="http://www.w3.org/1999/xhtml" '
           f'xmlns:epub="http://www.idpf.org/2007/ops" lang="{lang}"><head><meta charset="utf-8"/><title>Contents</title>'
           '</head><body><nav epub:type="toc" id="toc"><h1>Contents</h1><ol>'
           + "".join(f'<li><a href="{n}">{e(t)}</a></li>' for n, t, _ in files) + "</ol></nav></body></html>\n")
    modified = _doc_date(doc).strftime("%Y-%m-%dT%H:%M:%SZ")
    opf = ('<?xml version="1.0" encoding="utf-8"?>\n<package xmlns="http://www.idpf.org/2007/opf" version="3.0" '
           'unique-identifier="uid"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
           f'<dc:identifier id="uid">{uid}</dc:identifier><dc:title>{e(doc.title)}</dc:title><dc:language>{lang}</dc:language>'
           + (f"<dc:creator>{e(str(doc.meta['author']))}</dc:creator>" if doc.meta.get("author") else "")
           + f'<meta property="dcterms:modified">{modified}</meta></metadata><manifest>'
           '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
           + "".join(f'<item id="c{i + 1}" href="{n}" media-type="application/xhtml+xml"/>' for i, (n, _, _) in enumerate(files))
           + "</manifest><spine>" + "".join(f'<itemref idref="c{i + 1}"/>' for i in range(len(files)))
           + "</spine></package>\n")
    container = ('<?xml version="1.0" encoding="utf-8"?>\n<container version="1.0" '
                 'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                 '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
                 "</rootfiles></container>\n")
    fixed = (2020, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(zipfile.ZipInfo("mimetype", fixed), "application/epub+zip", compress_type=zipfile.ZIP_STORED)

        def add(name: str, data: str) -> None:
            z.writestr(zipfile.ZipInfo(name, fixed), data.encode("utf-8"), compress_type=zipfile.ZIP_DEFLATED)

        add("META-INF/container.xml", container)
        add("OEBPS/content.opf", opf)
        add("OEBPS/nav.xhtml", nav)
        for n, _, data in files:
            add(f"OEBPS/{n}", data)


# ---------------------------------------------------------------------------------------------------------- images
def _font(name: str, size: int):
    from PIL import ImageFont

    return ImageFont.truetype(str(FONT_DIR / name), size)


WORD_GAP = 0.6  # extra space width between words: "€ 2 340" and "11.5 h" lost their narrow spaces in OCR


def _tw(font, s: str) -> int:
    """Drawn width of ``s`` with the widened word gaps ``_draw`` uses."""
    return int(font.getlength(s) + s.count(" ") * font.getlength(" ") * WORD_GAP)


def _wrap(text: str, font, max_w: int) -> list[str]:
    """Wrap at spaces only; a word wider than the line stays whole (the canvas grows instead)."""
    out: list[str] = []
    for para in text.split("\n"):
        words = para.split(" ")
        line = ""
        for w in words:
            cand = w if not line else f"{line} {w}"
            if line and _tw(font, cand) > max_w:
                out.append(line)
                line = w
            else:
                line = cand
        out.append(line)
    return out


def _visual_lines(doc: Doc) -> list[tuple[str, object]]:
    """[(style, text | table Block)] in reading order; style picks the font."""
    out: list[tuple[str, object]] = [("title", doc.title), ("gap", "")]
    for b in doc.blocks:
        k = b.kind
        if k == "heading":
            out += [("gap", ""), ("head", b.text)]
        elif k in ("para", "quote"):
            out.append(("body", b.text))
        elif k == "note":
            out.append(("small", b.text))
        elif k == "bullets":
            out += [("bullet", i) for i in b.items]
        elif k == "kv":
            out += [("kv", f"{lab}: {val}") for lab, val in b.pairs]
        elif k == "table":
            if b.caption:
                out.append(("head", b.caption))
            out.append(("table", b))
        elif k == "pagebreak":
            out.append(("gap", ""))
        if k != "heading":
            out.append(("gap", ""))
    return out


def _style_fonts(body: int) -> dict:
    # Serif: tesseract reads its I/l/1 apart ("Kestrel-II" came back "Kestrel-Il" in DejaVu Sans)
    return {"title": _font("DejaVuSerif-Bold.ttf", int(body * 1.4)), "head": _font("DejaVuSerif-Bold.ttf", int(body * 1.12)),
            "body": _font("DejaVuSerif.ttf", body), "bullet": _font("DejaVuSerif.ttf", body),
            "kv": _font("DejaVuSerif-Bold.ttf", body), "small": _font("DejaVuSerif.ttf", max(28, int(body * 0.9))),
            "cell": _font("DejaVuSerif.ttf", max(28, int(body * 0.9))),
            "cellhead": _font("DejaVuSerif-Bold.ttf", max(28, int(body * 0.9))), "gap": _font("DejaVuSerif.ttf", body // 2)}


Placed = tuple[str, object, int]  # (style, text or [(x, cell text)], line height)


def _layout(lines: list[tuple[str, object]], fonts: dict, max_w: int, *, grow: bool = True) -> tuple[list[Placed], int, int]:
    """Wrap every line to ``max_w`` (at spaces only); tables become rows of positioned cells. When a table is wider
    than ``max_w`` and the canvas may not ``grow``, each row is written as "column: value; ..." lines instead."""
    placed: list[Placed] = []
    widest = 0

    def text_lines(style: str, text: str) -> None:
        nonlocal widest
        f = fonts[style]
        lh = int(f.size * 1.45)
        prefix = "\u2022 " if style == "bullet" else ""
        indent = _tw(f, prefix)
        for i, ln in enumerate(_wrap(text, f, max_w - indent)):
            if i == 0 and prefix:
                placed.append((style, [(0, prefix.strip()), (indent, ln)], lh))
            else:
                placed.append((style, [(indent, ln)], lh))
            widest = max(widest, indent + _tw(f, ln))

    for style, item in lines:
        if style == "gap":
            placed.append((style, [], int(fonts["gap"].size * 1.45)))
        elif style == "table":
            b = item
            rows = [b.columns, *b.rows]
            ncol = max(len(r) for r in rows)
            gap = 48
            fh, fc = fonts["cellhead"], fonts["cell"]
            widths = [max(_tw(fh if ri == 0 else fc, r[c] if c < len(r) else "") for ri, r in enumerate(rows))
                      for c in range(ncol)]
            total = sum(widths) + gap * (ncol - 1)
            if total > max_w and not grow:
                for r in b.rows:
                    text_lines("body", "; ".join(f"{h}: {v}" for h, v in zip(b.columns, r, strict=False)))
                continue
            xs = [sum(widths[:c]) + gap * c for c in range(ncol)]
            widest = max(widest, total)
            for ri, r in enumerate(rows):
                cells = [(xs[c], r[c]) for c in range(min(ncol, len(r))) if r[c]]
                placed.append(("cellhead" if ri == 0 else "cell", cells, int(fc.size * 1.6)))
        else:
            text_lines(style, str(item))
    return placed, widest, sum(p[2] for p in placed)


def _draw(d, placed: list[Placed], fonts: dict, x0: int, y0: int, ink) -> None:
    y = y0
    for st, cells, lh in placed:
        f = fonts.get(st)
        for x, text in cells:
            step = f.getlength(" ") * (1 + WORD_GAP)
            cx = x0 + x
            for w in text.split(" "):
                if w:
                    d.text((cx, y), w, font=f, fill=ink)
                cx += f.getlength(w) + step
        y += lh


def _render_image(doc: Doc) -> object:
    from PIL import Image, ImageDraw, ImageFilter

    rnd = random.Random(_seed(doc))
    style = doc.meta.get("image_style") or ("nameplate" if any(b.kind == "kv" for b in doc.blocks)
                                            and all(b.kind in ("kv", "heading") for b in doc.blocks) else "notice")
    body = 34 if style != "whiteboard" else 38
    fonts = _style_fonts(body)
    margin = 110 if style == "nameplate" else 80
    content_w = 1400 - 2 * margin
    placed, widest, total_h = _layout(_visual_lines(doc), fonts, content_w)
    content_w = max(content_w, widest)
    W, H = content_w + 2 * margin, total_h + 2 * margin
    if style == "nameplate":
        img = Image.new("RGB", (W, H), (196, 199, 203))
        px = img.load()
        for y in range(H):  # brushed metal: faint horizontal streaks
            v = rnd.randint(-6, 6)
            for x in range(0, W, 7):
                r, g, b = px[x, y]
                for dx in range(min(7, W - x)):
                    px[x + dx, y] = (r + v, g + v, b + v + 1)
        d = ImageDraw.Draw(img)
        d.rectangle([12, 12, W - 13, H - 13], outline=(70, 72, 76), width=6)
        d.rectangle([24, 24, W - 25, H - 25], outline=(150, 152, 156), width=2)
        for cx, cy in ((48, 48), (W - 48, 48), (48, H - 48), (W - 48, H - 48)):
            d.ellipse([cx - 16, cy - 16, cx + 16, cy + 16], fill=(120, 122, 126), outline=(60, 60, 64), width=3)
            d.line([cx - 10, cy - 10, cx + 10, cy + 10], fill=(50, 50, 54), width=4)
        ink = (18, 18, 22)
    elif style == "whiteboard":
        img = Image.new("RGB", (W, H), (250, 251, 252))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W - 1, H - 1], outline=(160, 165, 170), width=14)
        for _ in range(W * H // 6000):  # faint ghost smudges
            x, y = rnd.randrange(W), rnd.randrange(H)
            d.ellipse([x, y, x + rnd.randint(20, 90), y + rnd.randint(4, 14)], fill=(238, 240, 243))
        ink = (20, 40, 120)
    else:
        img = Image.new("RGB", (W, H), (250, 247, 238))
        small = Image.frombytes("L", (W // 4 + 1, H // 4 + 1), rnd.randbytes((W // 4 + 1) * (H // 4 + 1)))
        noise = small.resize((W, H), Image.BILINEAR).filter(ImageFilter.GaussianBlur(1)).convert("RGB")
        img = Image.blend(img, noise, 0.04)  # light paper grain, seeded from the doc id
        d = ImageDraw.Draw(img)
        d.rectangle([6, 6, W - 7, H - 7], outline=(200, 195, 180), width=3)
        ink = (15, 15, 15)
    _draw(ImageDraw.Draw(img), placed, fonts, margin, margin, ink)
    return img


def write_image(doc: Doc, path: Path) -> None:
    from PIL import Image

    img = _render_image(doc)
    ext = doc.ext.lower() if doc.ext else path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        bg = img.getpixel((2, 2))
        img = img.rotate(0.8, resample=Image.BICUBIC, expand=True, fillcolor=bg)
        img.save(path, "JPEG", quality=90, optimize=True)
    elif ext == ".png":
        img.save(path, "PNG", optimize=True)
    elif ext in (".tif", ".tiff"):
        img.save(path, "TIFF", compression="tiff_lzw")
    elif ext == ".bmp":
        img.save(path, "BMP")
    elif ext == ".webp":
        img.save(path, "WEBP", quality=92, method=4)
    elif ext == ".gif":
        img.convert("P", palette=Image.ADAPTIVE, colors=64).save(path, "GIF")
    else:
        raise ValueError(f"not an image extension: {ext}")


# ----------------------------------------------------------------------------------------------------- scanned pdf
def write_scan_pdf(doc: Doc, path: Path) -> None:
    from PIL import Image, ImageDraw

    rnd = random.Random(_seed(doc))
    dpi = 170
    W, H = int(8.27 * dpi), int(11.69 * dpi)
    margin = int(0.75 * dpi)
    body = 30
    fonts = _style_fonts(body)
    content_w = W - 2 * margin
    placed, _, _ = _layout(_visual_lines(doc), fonts, content_w, grow=False)
    pages_lines: list[list[Placed]] = [[]]
    y = margin
    for p in placed:
        if y + p[2] > H - margin and pages_lines[-1]:
            pages_lines.append([])
            y = margin
        pages_lines[-1].append(p)
        y += p[2]
    pages = []
    for pl in pages_lines:
        img = Image.new("L", (W, H), 246)
        d = ImageDraw.Draw(img)
        for _ in range(W * H // 2500):  # scanner speckle
            x, yy = rnd.randrange(W), rnd.randrange(H)
            d.point((x, yy), fill=rnd.randint(150, 215))
        _draw(d, pl, fonts, margin, margin, 25)
        angle = rnd.uniform(-0.5, 0.5)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=246)
        pages.append(img)
    when = _doc_date(doc)
    pages[0].save(path, "PDF", save_all=True, append_images=pages[1:], resolution=dpi, title=doc.title,
                  creationDate=when.timetuple(), modDate=when.timetuple())


# ------------------------------------------------------------------------------------------------------- dispatcher
def _check_raw(doc: Doc, ext: str) -> None:
    if doc.raw is not None and ext not in TEXT_EXTS:
        raise ValueError(f"raw text given for a non-text format {ext}")


def render_doc(doc: Doc, path: Path) -> None:
    """Write ``doc`` to ``path`` (its extension decides the format)."""
    ext = doc.ext.lower()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _check_raw(doc, ext)
    if ext in TEXT_EXTS:
        text = doc.raw if doc.raw is not None else _text_native(doc, ext)
        path.write_text(text, encoding="utf-8", newline="")
    elif ext == ".pdf" and doc.meta.get("scan"):
        write_scan_pdf(doc, path)
    elif ext in IMAGE_EXTS:
        write_image(doc, path)
    elif ext == ".docx":
        write_docx(doc, path)
    elif ext == ".xlsx":
        write_xlsx(doc, path)
    elif ext == ".xls":
        write_xls(doc, path)
    elif ext == ".pptx":
        write_pptx(doc, path)
    elif ext == ".rtf":
        write_rtf(doc, path)
    elif ext == ".eml":
        write_eml(doc, path)
    elif ext == ".epub":
        write_epub(doc, path)
    elif ext in LO_TARGETS:
        with tempfile.TemporaryDirectory(prefix="qmgen-one-") as tmp:
            src_ext, flt = LO_TARGETS[ext]
            src = Path(tmp) / f"{path.stem}{src_ext}"
            _write_source(doc, src)
            outdir = Path(tmp) / "out"
            outdir.mkdir()
            _soffice(flt, [src], outdir, Path(tmp) / "profile")
            got = outdir / f"{path.stem}{ext}"
            if not got.is_file() or got.stat().st_size == 0:
                raise RuntimeError(f"LibreOffice produced no {ext}")
            shutil.move(str(got), path)
    else:
        raise ValueError(f"unsupported extension {ext}")


def _write_source(doc: Doc, src: Path) -> None:
    if src.suffix == ".docx":
        write_docx(doc, src, for_pdf=doc.ext.lower() == ".pdf")
    elif src.suffix == ".xlsx":
        write_xlsx(doc, src)
    elif src.suffix == ".pptx":
        write_pptx(doc, src)
    else:
        raise ValueError(src.suffix)


def _soffice(flt: str, files: list[Path], outdir: Path, profile: Path) -> None:
    cmd = [SOFFICE, "--headless", "--norestore", "--nolockcheck", f"-env:UserInstallation=file://{profile}",
           "--convert-to", flt, "--outdir", str(outdir), *map(str, files)]
    subprocess.run(cmd, capture_output=True, timeout=LO_TIMEOUT, check=False,
                   env={**os.environ, "SAL_USE_VCLPLUGIN": "svp"})


# ------------------------------------------------------------------------------------------------------ batch mode
def _direct_job(args: tuple[Doc, str]) -> tuple[str, str]:
    doc, dest = args
    try:
        render_doc(doc, Path(dest))
        return doc.relpath, "ok"
    except Exception as e:  # noqa: BLE001 - one bad doc is reported, never stops the corpus
        return doc.relpath, f"error: {type(e).__name__}: {e}"


def _source_job(args: tuple[Doc, str]) -> tuple[str, str]:
    doc, src = args
    try:
        _check_raw(doc, doc.ext.lower())
        Path(src).parent.mkdir(parents=True, exist_ok=True)
        _write_source(doc, Path(src))
        return doc.relpath, "ok"
    except Exception as e:  # noqa: BLE001
        return doc.relpath, f"error: {type(e).__name__}: {e}"


def render_all(docs: list[Doc], out_dir: Path, *, workers: int = 4, log: Callable[[str], None] = print) -> dict[str, str]:
    """Write every doc under ``out_dir/doc.relpath``; returns {relpath: "ok" | "error: <reason>"}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, str] = {}
    lo_docs = [d for d in docs if d.ext.lower() in LO_TARGETS and not (d.ext.lower() == ".pdf" and d.meta.get("scan"))]
    lo_ids = {id(d) for d in lo_docs}
    direct = [d for d in docs if id(d) not in lo_ids]
    stage = Path(tempfile.mkdtemp(prefix=f"qmgen-stage-{os.getpid()}-"))
    try:
        # the staged source name is the doc's index: unique within a batch whatever the relpaths are
        lo_src = {i: stage / "src" / d.ext.lower().lstrip(".") / f"d{i:06d}{LO_TARGETS[d.ext.lower()][0]}"
                  for i, d in enumerate(lo_docs)}
        with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
            jobs = [(d, str(out_dir / d.relpath)) for d in direct]
            for n, (rel, status) in enumerate(pool.map(_direct_job, jobs, chunksize=8), 1):
                result[rel] = status
                if n % 1000 == 0:
                    log(f"render: {n}/{len(jobs)} direct files")
            src_status = list(pool.map(_source_job, [(d, str(lo_src[i])) for i, d in enumerate(lo_docs)], chunksize=4))
        log(f"render: {len(direct)} direct files written; {len(lo_docs)} to convert with LibreOffice")
        tasks: list[tuple[str, list[int]]] = []
        for target in LO_TARGETS:
            idx = [i for i, d in enumerate(lo_docs) if d.ext.lower() == target and src_status[i][1] == "ok"]
            tasks += [(target, idx[k:k + LO_CHUNK]) for k in range(0, len(idx), LO_CHUNK)]
        for i, (rel, status) in enumerate(src_status):
            if status != "ok":
                result[rel] = status

        def convert(task: tuple[int, tuple[str, list[int]]]) -> list[tuple[str, str]]:
            n, (target, idx) = task
            flt = LO_TARGETS[target][1]
            outdir = stage / "out" / f"{n:05d}"
            outdir.mkdir(parents=True, exist_ok=True)
            slot = slots.get()
            profile = Path(tempfile.gettempdir()) / f"qmgen-lo-{os.getpid()}-{slot}"
            done: list[tuple[str, str]] = []
            try:
                return _convert_batch(target, flt, idx, outdir, profile, done)
            finally:
                slots.put(slot)

        def _convert_batch(target, flt, idx, outdir, profile, done):
            try:
                _soffice(flt, [lo_src[i] for i in idx], outdir, profile)
            except subprocess.TimeoutExpired:
                pass
            for i in idx:
                d = lo_docs[i]
                got = outdir / f"{lo_src[i].stem}{target}"
                if not got.is_file() or got.stat().st_size == 0:  # retry alone once
                    try:
                        _soffice(flt, [lo_src[i]], outdir, profile)
                    except subprocess.TimeoutExpired:
                        pass
                if got.is_file() and got.stat().st_size > 0:
                    dest = out_dir / d.relpath
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(got), dest)
                    done.append((d.relpath, "ok"))
                else:
                    done.append((d.relpath, f"error: LibreOffice produced no {target}"))
            return done

        # one soffice per worker slot at a time; each slot has its own profile dir
        slots: queue.SimpleQueue[int] = queue.SimpleQueue()
        for n in range(max(1, workers)):
            slots.put(n)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as tp:
            for k, batch in enumerate(tp.map(convert, list(enumerate(tasks))), 1):
                result.update(batch)
                if k % 10 == 0:
                    log(f"render: {k}/{len(tasks)} LibreOffice batches")
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        for n in range(max(1, workers)):
            shutil.rmtree(Path(tempfile.gettempdir()) / f"qmgen-lo-{os.getpid()}-{n}", ignore_errors=True)
    bad = sum(1 for v in result.values() if v != "ok")
    log(f"render: {len(result) - bad} ok, {bad} errors")
    return result
