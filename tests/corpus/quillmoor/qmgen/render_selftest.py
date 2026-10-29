"""Self-test of the format writers: render sample docs to every extension, parse each file with the app's own parser,
and check that every distinctive token survived.

    .venv/bin/python tests/corpus/quillmoor/qmgen/render_selftest.py [--bench N]   # from the repo root

Non-OCR formats must keep 100% of their tokens (exit 1 otherwise); OCR formats (images, scanned PDF) report a
recovery rate. A parser exception fails every format, including OCR. Tokens are compared with whitespace collapsed:
page formats wrap lines, a wrapped phrase is still the same phrase.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                    # tests/corpus/quillmoor -> `import qmgen`
sys.path.insert(0, str(HERE.parents[3] / "src"))        # repo src -> `import finetune_studio`

from qmgen.model import Block, Doc
from qmgen.render import ALL_EXTS, IMAGE_EXTS, is_ocr, norm_ws, render_all

REPORT_TOKENS = ["QZ-77412-K", "SN40318827", "1,284.75 EUR", "14.03.2021", "net 47 days", "Ørsund Wharf", "11.5 h",
                 "€ 2 340", "port 18443", "Brakkenfjell", "R-2209/7", "07:45", "Halvdan Tørrisen", "0.042 mm",
                 "PO-55130", "rev. C4", "Nørre Slip 3", "48.6 kWh", "Q3-2021", "3,500 kg", "Line 9 / Bay 14",
                 "pressure 6.2 bar", "ticket #88214", "Kestrel-II", "CSV-ALPHA-31"]
REGISTER_TOKENS = ["QZ-77412-K", "SN40318827", "1,284.75 EUR", "14.03.2021", "Ørsund Wharf", "€ 2 340", "11.5 h",
                   "R-2209/7", "0.042 mm", "=SUM(A1)", "03.04.2019", "1,250.00", "007431", "Brakkenfjell"]
PLATE_TOKENS = ["QZ-77412-K", "SN40318827", "14.03.2021", "11.5 h", "port 18443", "400 V", "3,500 kg", "IP54"]
EMAIL_TOKENS = ["QZ-77412-K", "1,284.75 EUR", "net 47 days", "Ørsund Wharf", "14.03.2021", "€ 2 340", "PO-55130",
                "Halvdan Tørrisen", "ticket #88214"]


def sample_report(doc_id: str = "selftest-report", ext: str = ".txt") -> Doc:
    return Doc(
        doc_id=doc_id, kind="incident_report", ext=ext, title="Incident report Kestrel-II at Ørsund Wharf",
        meta={"author": "Halvdan Tørrisen", "date": "2021-03-14", "company": "Quillmoor Marine",
              "header": "Quillmoor Marine - internal", "footer": "Form QM-IR rev. C4"},
        blocks=[
            Block("heading", text="Summary", level=1),
            Block("para", text="On 14.03.2021 at 07:45 the winch QZ-77412-K (serial SN40318827) at Ørsund Wharf stopped "
                               "after 11.5 h of continuous duty. The supplier invoice of 1,284.75 EUR is payable net 47 "
                               "days under order PO-55130."),
            Block("bullets", items=["Reported by Halvdan Tørrisen under ticket #88214.",
                                    "Telemetry gateway listens on port 18443.",
                                    "Spare part R-2209/7 ordered for Nørre Slip 3."]),
            Block("heading", text="Measurements", level=2),
            Block("table", caption="Readings Q3-2021", columns=["Item", "Value", "Where"],
                  rows=[["Shaft wear", "0.042 mm", "Line 9 / Bay 14"], ["Energy used", "48.6 kWh", "Brakkenfjell"],
                        ["Hydraulics", "pressure 6.2 bar", "Ørsund Wharf"], ["Repair quote", "€ 2 340", "CSV-ALPHA-31"]]),
            Block("pagebreak"),
            Block("heading", text="Asset data", level=1),
            Block("kv", pairs=[("Asset", "Kestrel-II"), ("Rated load", "3,500 kg"), ("Drawing", "rev. C4")]),
            Block("quote", text="We saw the brake slip twice before 07:45."),
            Block("note", text="Figures checked against the Q3-2021 register."),
        ])


def sample_register(doc_id: str = "selftest-register", ext: str = ".csv") -> Doc:
    return Doc(
        doc_id=doc_id, kind="asset_register", ext=ext, title="Asset register Brakkenfjell", meta={"title_row": True},
        blocks=[Block("table", caption="Assets: Brakkenfjell [2021]", columns=["Asset tag", "Serial", "Cost", "Commissioned", "Site", "Note"],
                      rows=[["QZ-77412-K", "SN40318827", "1,284.75 EUR", "14.03.2021", "Ørsund Wharf", "€ 2 340"],
                            ["R-2209/7", "007431", "1,250.00", "03.04.2019", "Brakkenfjell", "11.5 h"],
                            ["QZ-10001-A", "0.042 mm", "=SUM(A1)", "03.04.2019", "Brakkenfjell", "11.5 h"]])])


def sample_plate(doc_id: str = "selftest-plate", ext: str = ".png") -> Doc:
    return Doc(doc_id=doc_id, kind="nameplate", ext=ext, title="QUILLMOOR WINCH", meta={"image_style": "nameplate"},
               blocks=[Block("kv", pairs=[("Model", "QZ-77412-K"), ("Serial", "SN40318827"), ("Built", "14.03.2021"),
                                          ("Duty", "11.5 h"), ("Supply", "400 V"), ("Max load", "3,500 kg"),
                                          ("Protection", "IP54"), ("Service", "port 18443")])])


def sample_email(doc_id: str = "selftest-email", ext: str = ".eml") -> Doc:
    return Doc(
        doc_id=doc_id, kind="email", ext=ext, title="Re: invoice QZ-77412-K",
        meta={"email": {"from": "Halvdan Tørrisen <halvdan@quillmoor.invalid>", "to": "ops@quillmoor.invalid",
                        "cc": "finance@quillmoor.invalid", "subject": "Re: invoice QZ-77412-K",
                        "date": "Sun, 14 Mar 2021 07:45:00 +0100", "message_id": "<qm-1@quillmoor.invalid>",
                        "in_reply_to": "<qm-0@quillmoor.invalid>"}},
        blocks=[Block("para", text="The invoice of 1,284.75 EUR for Ørsund Wharf is payable net 47 days (PO-55130)."),
                Block("para", text="Repair quote: € 2 340, see ticket #88214."),
                Block("quote", text="On 14.03.2021 Halvdan Tørrisen wrote:\nCan you confirm the amount?")])


def cases() -> list[tuple[Doc, list[str]]]:
    out: list[tuple[Doc, list[str]]] = []
    for ext in sorted(ALL_EXTS):
        if ext in (".csv", ".tsv"):
            out.append((sample_register(f"register{ext.replace('.', '-')}", ext), REGISTER_TOKENS))
            continue
        if ext in IMAGE_EXTS:
            out.append((sample_plate(f"plate{ext.replace('.', '-')}", ext), PLATE_TOKENS))
            out.append((sample_report(f"report{ext.replace('.', '-')}", ext), REPORT_TOKENS))
            continue
        out.append((sample_report(f"report{ext.replace('.', '-')}", ext), REPORT_TOKENS))
        if ext in (".xlsx", ".xls", ".ods", ".json", ".jsonl", ".xml", ".yaml", ".docx", ".pdf", ".pptx"):
            out.append((sample_register(f"register{ext.replace('.', '-')}", ext), REGISTER_TOKENS))
    out.append((sample_email("email-eml", ".eml"), EMAIL_TOKENS))
    scan = sample_report("report-scan-pdf", ".pdf")
    scan.meta["scan"] = True
    out.append((scan, REPORT_TOKENS))
    return out


def run_selftest() -> int:
    from finetune_studio.data.parsers import parse

    cs = cases()
    with tempfile.TemporaryDirectory(prefix="qmgen-selftest-") as tmp:
        out = Path(tmp)
        t0 = time.time()
        status = render_all([d for d, _ in cs], out, workers=4, log=lambda m: None)
        print(f"rendered {len(cs)} files in {time.time() - t0:.1f}s")
        rows = []
        failed = False
        for doc, tokens in cs:
            st = status.get(doc.relpath, "missing")
            label = doc.ext + (" (scan)" if doc.meta.get("scan") else "") + f" [{doc.kind}]"
            if st != "ok":
                rows.append((label, 0, 0, len(tokens), st, 0.0))
                failed = True
                continue
            t1 = time.time()
            try:
                text = parse(out / doc.relpath).get("text") or ""
            except Exception as e:  # noqa: BLE001
                text = ""
                st = f"parse error {e}"
                failed = True
            dt = time.time() - t1
            nt = norm_ws(text)
            missing = [t for t in tokens if norm_ws(t) not in nt]
            rows.append((label, len(text), len(tokens) - len(missing), len(tokens), ", ".join(missing) or "-", dt))
            if missing and not is_ocr(doc):
                failed = True
        print(f"{'ext':34} {'chars':>7} {'found':>9}  {'secs':>5}  missing")
        for label, n, f, tot, miss, dt in rows:
            pct = 100 * f / tot if tot else 0
            print(f"{label:34} {n:7d} {f:3d}/{tot:<3d}{pct:4.0f}% {dt:5.1f}  {miss}")
        ocr = [(f, tot) for (label, _, f, tot, _, _), (d, _) in zip(rows, cs, strict=True) if is_ocr(d)]
        if ocr:
            print(f"OCR formats: {sum(f for f, _ in ocr)}/{sum(t for _, t in ocr)} tokens "
                  f"({100 * sum(f for f, _ in ocr) / sum(t for _, t in ocr):.0f}%)")
    print("FAIL" if failed else "PASS (all non-OCR formats 100%)")
    return 1 if failed else 0


def bench(n: int) -> None:
    exts = sorted(ALL_EXTS - {".csv", ".tsv"})
    docs = []
    for i in range(n):
        d = sample_report(f"bench-{i:05d}", exts[i % len(exts)])
        d.subdir = f"b{i % 7}"
        docs.append(d)
    with tempfile.TemporaryDirectory(prefix="qmgen-bench-") as tmp:
        t0 = time.time()
        st = render_all(docs, Path(tmp), workers=4, log=print)
        dt = time.time() - t0
    bad = [k for k, v in st.items() if v != "ok"]
    print(f"bench: {n} docs in {dt:.1f}s = {n / dt:.1f} docs/s, {len(bad)} errors {bad[:3]}")


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench(int(sys.argv[sys.argv.index("--bench") + 1]))
        sys.exit(0)
    sys.exit(run_selftest())
