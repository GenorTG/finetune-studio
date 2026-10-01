# data/parsers — file-format extraction layer

Converts uploaded/library files (office docs, text/code, spreadsheets, markup,
images, email, ebooks) into plain text plus structured metadata for RAG
ingestion, Q&A chunking, and training-data conversion. The live implementation
is the `finetune_studio.data.parsers` **package** (`src/finetune_studio/data/parsers/`,
one module per format) behind a single dispatcher in its `__init__.py`.

## `data/parsers/__init__.py` — the dispatcher (THE canonical entry point)

- `PARSERS: dict[str, tuple[module, fn]]` — extension → `(module_name, "parse")`,
  covering 33+ extensions (text/code, csv/tsv, json/jsonl, xml, html, images,
  pdf/docx/doc/xlsx/xls/pptx/odt/ods/odp, rtf, epub, eml/msg).
- `get_parser_for(path) -> callable | None` — resolves and lazily imports the
  per-format `parse` function via `importlib.import_module`.
- `parse(path) -> dict` — the one true entry point. Looks up the extension,
  calls the format's `parse(path)`, and returns the standard envelope
  (see `_base.make_result`). Unknown extensions fall back to reading the file
  as UTF-8 text with a `warnings` note. **Gotcha:** this function does not
  wrap the call to the per-format parser in try/except — if a format parser
  itself raises (most now guard their own failure modes, see below), the
  exception propagates to the caller. Callers that loop over many files
  (e.g. `data/audit.py:audit_project_sources`) do not catch per-file either,
  so one bad file used to abort the whole batch — fixed for JSON in this
  audit, see Discrepancies below.
- `parse_bytes(filename, data) -> dict` — writes `data` to a temp file (so
  every parser can stay path-based) and delegates to `parse()`; always
  cleans up the temp file in a `finally`.
- `list_parsers() -> list[dict]` — introspection helper (extension/module/fn
  triples), used for e.g. documenting supported formats, not required by the
  parse path itself.

**Output envelope** (built by every format's `parse()` via `_base.make_result`):
```json
{"text": "...", "structured": {...}, "metadata": {"parser": "...", "version": "1",
 "parsed_at": "...", "char_count": N, "warnings": [...]}}
```
**Load-bearing invariant:** only `result["text"]` is read by any caller in
this repo (`rag/ingest.py:extract_text`, `data/rag_portable/store.py`,
`data/audit.py`, `webui/routes/file_library.py` reparse flow). `result["structured"]`
is populated by every parser but **no caller in the codebase reads it** —
anything a format parser puts only into `structured` and not into `text` is
effectively invisible to RAG ingestion, Q&A mining, and training-data export.
This was the root cause of the DOCX table-loss bug fixed below; keep it in
mind before adding a new parser that computes rich structured data — if it
matters for training, it must also be folded into `text`.

## `data/parsers/_base.py` — shared helpers
- `make_result(text, structured, parser, warnings=None, **extra) -> dict` —
  builds the envelope above; `char_count` is always `len(text)`, so a parser
  that sets `text` wrong (e.g. the email bug below) silently corrupts this
  field too.
- `cli_run(parse_fn)` — the common `if __name__ == "__main__":` entry every
  format module wires up, so each parser is independently runnable:
  `python -m finetune_studio.data.parsers.pdf file.pdf [--pretty]`.
- `now_iso()` — UTC ISO-8601 timestamp for `metadata.parsed_at`.

## Per-format parsers

Each file below exposes exactly one function, `parse(path: Path) -> dict`
(images/pdf also take `languages: str = DEFAULT_LANGS`), always going through
`_base.make_result`.

- **`text.py`** (`.txt .md .markdown .log .py .js .ts .jsx .tsx .css .yaml .yml .ini .cfg .conf`) —
  reads as UTF-8 (replace errors), guesses a `language` tag for syntax hints,
  computes `line_count`. No failure modes to speak of.
- **`csv.py`** (`.csv .tsv`) — sniffs the dialect (delimiter) and header
  presence via `csv.Sniffer`, falls back to extension-based delimiter guess
  on sniff failure. Renders a markdown table into `text` using **every** row
  (not just the `structured.rows[:200]` sample) — so CSV content never gets
  silently truncated in the text actually used downstream.
- **`json.py`** (`.json`) — pretty-prints parsed JSON into `text`.
  **Fixed in this audit:** previously `json.loads()` was unguarded, so a
  malformed `.json` file raised `JSONDecodeError` straight out of `parse()`,
  which (since the dispatcher also has no try/except) propagated into
  `data/audit.py:audit_project_sources`'s list comprehension and crashed the
  *entire project's* audit over one bad file. Now mirrors the `xml.py`/`jsonl.py`
  pattern: catches `JSONDecodeError`, returns the raw text with a `warnings`
  entry and `structured.parse_error` instead of raising.
- **`jsonl.py`** (`.jsonl`) — parses line-by-line; a bad line is kept verbatim
  in `text` and recorded in `warnings` with its line number — the reference
  pattern other formats should follow for partial-failure handling.
- **`xml.py`** (`.xml`) — `ET.parse`; on `ParseError` falls back to raw text
  with a warning (same good pattern as jsonl). `structured.elements` is
  capped at 200 (sample only; full content lives in `text` via `ET.tostring`).
- **`html.py`** (`.html .htm`) — BeautifulSoup-based text extraction, strips
  script/style/nav/footer/header/noscript/aside/form before `get_text()`.
  Falls back to a regex tag-stripper when `bs4` isn't installed. Captures
  title/headings/link_count in `structured` (links themselves not retained
  beyond count — acceptable since anchor text is already inline in `text`).
- **`pdf.py`** (`.pdf`) — fallback chain pypdf → PyPDF2 → `pdftotext` CLI →
  OCR (via `data/ocr.py:ocr_pdf`, tesseract). `text` joins **all** extracted
  pages; `structured.pages` is capped at 500 for size but that cap never
  affects `text`. `metadata.parser` encodes which method actually won (e.g.
  `pdf_v1_pypdf`, `pdf_v1_ocr_eng+pol`) — useful for debugging why a given
  PDF came out empty.
- **`docx.py`** (`.docx`) — paragraphs + headings (by `Heading N` style) +
  tables via `python-docx`. **Fixed in this audit:** table rows were
  collected into `structured.tables` but never appended to `text` — since
  `structured` is write-only (see invariant above), every table in every
  DOCX ever ingested was silently dropped from RAG/training content. Table
  rows (`cell | cell | ...`) are now appended to `text` after the paragraphs,
  matching the pattern `pptx.py`/`xlsx.py`/`xls.py` already used correctly.
- **`doc.py`** (`.doc`, legacy Word 97-2003) — fallback chain antiword →
  catdoc → `textract` → a pure-Python OLE2 scan (`_olefile_extract`, scans
  the `WordDocument` stream for 16-bit printable runs ≥12 chars) as a last
  resort before a placeholder string. `structured.extraction_method` records
  which step won.
- **`xlsx.py`** / **`xls.py`** — one sheet section per sheet in `text` with
  all non-blank rows; `structured.sheets[].rows` sample-capped at 200 but,
  as with CSV, the cap is structured-only — `text` has the full data.
- **`pptx.py`** (`.pptx`) — per-slide section headers, shape text + table
  rows folded into `text` correctly (this file is the reference example that
  revealed the docx.py gap).
- **`odf.py`** (`.odt .ods .odp`) — unzips, reads `content.xml`, extracts all
  `<text:p>`/`<text:h>` element text (also catches ODS cell paragraphs, which
  wrap their content the same way). Any exception (bad zip, missing member)
  is caught and reported via `structured.error` + `warnings`, never raises.
- **`rtf.py`** (`.rtf`) — `striprtf` if installed, else a minimal regex strip
  fallback.
- **`epub.py`** (`.epub`) — unzips, converts every `.xhtml/.html/.htm`
  chapter via `bs4` (or regex fallback) into `text`; `structured.chapters`
  carries per-chapter char counts only (chapter text itself lives in `text`).
- **`email.py`** (`.eml .msg`) — RFC 822 parse via `email.policy.default`.
  Walks non-multipart parts, skips attachment-disposition parts (recorded in
  `structured.attachments` with filename/type/size instead), joins remaining
  body parts into `text`. **Fixed in this audit:** the no-body fallback path
  called `msg.get_body(preferencelist=("plain",))` and used its return value
  directly as `text` — but `get_body()` returns an `email.message.Message`
  *part object*, not a string. Because `Message` objects are truthy and
  support `len()` (counting headers/payload items, not characters), this bug
  did not crash immediately but silently stored a non-string object into the
  `text` field: `make_result`'s `char_count = len(text)` became a bogus
  number, and any JSON-serializing caller (`cli_run`, anything that persists
  `parsed.txt` as JSON) would have raised `TypeError: Object of type
  EmailMessage is not JSON serializable`. Fixed to call `.get_content()` on
  the returned part (with an `("plain", "html")` preference list and a guard
  for `None`/decode failure), falling through to the raw source as last resort.
- **`image.py`** (`.png .jpg .jpeg .tif .tiff .bmp .webp .gif`) — OCR via
  `data/ocr.py:ocr_image`; best-effort width/height/format via Pillow.
  `languages` defaults to `ocr.DEFAULT_LANGS` (`"eng+pol"`); the dispatcher's
  `parse()`/`parse_bytes()` always call with the default — there is currently
  no way to pass a custom OCR language through the package-level entry
  points, only by importing `parsers.image.parse` directly.

## The removed `data/parsers.py` module (dead code)

Prior to this audit, `src/finetune_studio/data/parsers.py` (428 lines) sat
alongside the `data/parsers/` package and reimplemented **every single
format** as a flat function (`parse_pdf`, `parse_docx`, `parse_xlsx`, …) plus
its own `PARSERS` dict, `parse_document()`, `parse_bytes()`, and
`ingest_bytes()`. In Python, when a package directory (`parsers/` with
`__init__.py`) and a module file (`parsers.py`) share a name in the same
parent package, the package always wins the import — `parsers.py` could
**never** be reached via `finetune_studio.data.parsers` and had zero other
import paths into it (confirmed via `importlib.util.find_spec` and a repo-wide
grep for any `spec_from_file_location`-style path import). It was pure dead
code: a second, divergent implementation of the "one true way" format-parsing
concern that could not run and would only mislead a future reader who tried
to "fix a parser bug" in it. **Deleted** in this audit; the live package
(`data/parsers/`) is the sole parser implementation, already covered end to
end by `tests/test_parsers_coverage.py`.

Note: `src/finetune_studio/data/prep/parsers.py` is unrelated despite the
name — it parses raw LLM text output into `{"q":..., "a":...}` Q&A pairs, not
file formats. It belongs to a different lane and was not read/edited here.

## How this module is wired into the app

- `rag/ingest.py:extract_text()` — the main RAG-ingestion entry, calls
  `data.parsers.parse(path)` and takes only `result["text"]`. Also exposes
  `extract_pdf()` / `extract_docx()` as backward-compat single-format helpers
  delegating to `parsers.pdf.parse` / `parsers.docx.parse` directly.
- `data/rag_portable/store.py` — same `parse()` entry point for the portable
  RAG store's own ingestion path; also reads `PARSERS` to list supported
  extensions.
- `data/audit.py:audit_source()` — calls `parsers.parse_bytes(filename, raw)`
  to deterministically re-derive `text` from stored raw bytes and compare
  against the persisted `parsed.txt`, to catch any drift between what was
  stored and what the current parser would produce.
- `data/prep/ingest.py` — calls `parsers.parse_bytes` for project-file
  ingestion into the Q&A prep pipeline.
- `webui/routes/data_prep.py`, `webui/routes/file_library.py` — read
  `PARSERS` (extension-membership checks and the `promotable_extensions`
  list surfaced to the UI); do not call `parse()` directly.
- `data/ocr.py` — not a parser itself; `parsers/pdf.py` and `parsers/image.py`
  both depend on it (`ocr_pdf`, `ocr_image`) for the OCR fallback/primary path.

## `data/ocr.py`

Tesseract wrapper shared by `parsers/pdf.py` (OCR fallback when no
extractable text layer) and `parsers/image.py` (primary extraction path).

- `is_available(languages)` — sanity-checks the `tesseract` binary plus
  `--list-langs` output against the requested language codes.
- `_ensure_tessdata(languages)` — self-heals by downloading missing
  `tessdata_fast` `.traineddata` files into `~/.local/share/tessdata/` on
  first use per process (guarded by a lock + one-shot flag); raises
  `RuntimeError` if the download itself fails. Opt out with
  `FTS_OCR_AUTOINSTALL=0` (e.g. air-gapped CI).
- `ocr_image(path, languages, psm=3, oem=1)` — shells out to `tesseract`,
  writing output to a `/tmp/_ocr_{pid}_{hash}.txt` file it then reads and
  deletes. **Gotcha:** the temp filename is derived from `hash(str(path))`,
  which is stable within one process — two threads OCR-ing the *same* image
  path concurrently in the same process could collide on the same temp file.
  Not hit by current callers (sequential per-file parsing), but worth
  knowing before adding concurrent OCR.
- `ocr_image_object(img, ...)` — in-memory PIL image OCR via `pytesseract`,
  avoids the disk round-trip; used internally by `ocr_pdf`.
- `ocr_pdf(path, languages, dpi=200)` — rasterizes every page via
  `pdf2image.convert_from_path`, OCRs each page independently; a page-level
  OCR failure is caught, logged, and recorded as `{"text": "", "error": ...}`
  for that page only — does not abort the whole document.
- `install_hint()` — returns a platform-correct apt/brew/pacman/dnf command
  string, read from `/etc/os-release` on Linux.

## `data/converter.py`

CLI-facing format converters (`cli/commands/convert.py`) that turn
CSV/plain-text/JSON into the JSONL chat-message format the training pipeline
expects (`{"messages": [{"role": ..., "content": ...}]}`).
- `csv_to_jsonl` — maps one CSV column (`text_column`) into a single
  user-message per row.
- `simple_to_chat` — splits a text file on blank lines into Q:/A: blocks,
  each becoming one `{"messages": [...]}` JSONL record.
- `jsonl_to_json` / `json_to_jsonl` — straight format round-trips.
- **Gotcha:** unlike every `data/parsers/*.py` file, these `open()` calls
  don't pass `encoding="utf-8"` or `newline=""`; on a non-UTF-8-locale box or
  a CSV with embedded newlines this can behave differently from the `csv.py`
  parser in the main pipeline. Low risk (training data is typically produced
  on the same box it's consumed on) but inconsistent with the rest of this
  module's conventions — worth tightening if this path sees more use.

## `data/validator.py`

Pre-training JSONL/JSON/TXT sanity checker (`cli/commands/validate.py`,
`webui/routes/data.py`). `validate_jsonl` walks every line, flags invalid
JSON, missing `role`/`content` keys, and non-list `messages`, accumulating
`errors`/`warnings`/`stats` rather than raising — a good pattern this audit
used as a cross-check for the `json.py` parser fix above (validator already
treated malformed-JSON-per-line as a recoverable warning, not a crash).

## `data/sentence_transformer_local.py`

Repairs/normalizes locally-cached `sentence-transformers` save directories
so `SentenceTransformer(path)` can load them under ST ≥5, which requires
`embedding_dimension` in `1_Pooling/config.json` where older saves only had
`word_embedding_dimension` or omitted the pooling config dir entirely.
Explicitly designed to **raise**, never silently substitute a different
embedder, when a dir can't be made loadable — see the docstring note about
"never a silent switch to a different embedder."
- `copy_sentence_transformer_tree` / `move_sentence_transformer_tree` —
  tree copy/move that preserves module subdirectories (`1_Pooling/`,
  `2_Normalize/`), skipping `META.json` (written separately by the shared
  model store).
- `is_complete_sentence_transformer_dir` — read-only check: every module in
  `modules.json` has its directory, and any Pooling module has a
  `config.json` with `embedding_dimension` or `word_embedding_dimension`.
- `prepare_local_sentence_transformer_dir` — the repair entry point: creates
  missing module dirs, writes a minimal Pooling config when absent (inferring
  dimension from sibling configs via `_infer_embedding_dimension`), and
  normalizes legacy pooling keys in every module's config.
- `_apply_pooling_compat` — migrates `word_embedding_dimension` →
  `embedding_dimension` and legacy boolean `pooling_mode_*_token` flags into
  the single `pooling_mode` string ST ≥5 expects. **Fixed in this audit:**
  had a third `elif` branch (`pooling_mode_mean_tokens is False and
  pooling_mode_cls_token`) that could never execute — it required
  `pooling_mode_cls_token` to be truthy, but the first `if` in the same
  chain already catches every truthy `pooling_mode_cls_token` case, so the
  `elif`'s own condition could never be reached. Removed as unreachable dead
  code; behavior is unchanged (the branch never fired).

Callers: `data/shared_models.py` (model-store registration), `data/rag_portable/embedders.py`
(embedder loading for the portable RAG store).

## Cross-module findings — NOT fixed, needs parent coordination

- **Naming collision, not a real duplicate:** `src/finetune_studio/data/prep/parsers.py`
  (owned by another lane) shares the name "parsers.py" with this lane's now-removed
  `data/parsers.py`, but implements a completely different concern (LLM
  JSON-output → Q&A pair extraction, not file-format parsing). No action
  needed — flagging only because the task brief asked to check for exactly
  this kind of name collision.
- No other cross-module conflicts found. The "two code paths could parse the
  same file type differently" risk named in the task brief is fully resolved
  now that the dead `data/parsers.py` duplicate is removed — `data/parsers/`
  (the package) is the only live implementation for every format.
