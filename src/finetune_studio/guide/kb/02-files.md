---
id: files
title: Files page (step 1)
page: files
keywords: upload files documents parse parsed pdf docx doc xls pptx rtf html csv txt md folder tag reparse trash library parser status
---
## Purpose
Step 1 of both flows: upload documents, organise them in folders, and confirm each one parsed to readable text. Pairs and the RAG index both read from this library.

## Workflow
1. Open **Upload**, choose files with the file picker, then click **Upload** — choosing files alone does not start the upload.
2. Wait for every row to show a green parsed pill (about 10 s for small files). A row stuck on queued means a parser problem.
3. Open the eye icon on a row: parsed text must be complete and readable (tables flattened, no HTML tags).
4. Fix bad parses with **Re-parse selected**; delete or restore files via the trash views.
5. Next: `/projects/{pid}/data-prep` (pairs) or `/projects/{pid}/rag`.

## Key controls
- `#fb-upload-toggle` — shows/hides the upload card.
- `#fb-upload-input` — file picker (txt, md, docx, doc, pdf, html, csv, xlsx, xls, pptx, rtf).
- `#fb-upload-form` — the upload form; its Upload button actually sends.
- `#fb-files-table` — the library table (type, size, parse status).
- `#fb-bulk-reparse` — re-run the parser on the selected rows.
- `#fb-search` — search the file names.
- `#fb-view-trash` — switch to the trash view.

## What to check
Each file: status ready, non-zero character count, a sensible parser name. Empty text for `.xls`, `.pptx` or `.rtf` means a document-parser module is missing (`bash install.sh --check`).

## Common mistakes
Uploading scans without text (they parse to nothing); expecting a file to be usable before it is parsed.
