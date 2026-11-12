---
id: files
title: Files page (step 1)
page: files
keywords: upload files documents parse parsed pdf docx doc xls pptx rtf html csv txt md folder tag reparse trash library parser status
---
## Purpose
Step 1 · Files & Knowledge: upload documents, organise them in folders, and check (or edit) the text each one parsed into. Every readable file is parsed once on arrival; Q&A pairs (training) and the Smart Brain both read that one parse.

## Workflow
1. Open **Upload**, choose files with the file picker, then click **Upload** — choosing files alone does not start the upload.
2. Wait for every row to show a green parsed pill (about 10 s for small files). A row stuck on queued means a parser problem.
3. Open the eye icon on a row: parsed text must be complete and readable (tables flattened, no HTML tags).
4. Fix bad parses with **Re-parse selected**; delete or restore files via the trash views.
5. Next: `/projects/{pid}/data-prep` (pairs) or `/projects/{pid}/brain`.

## Key controls
- `#fb-upload-toggle` — shows/hides the upload card.
- `#fb-upload-input` — file picker (txt, md, docx, doc, pdf, html, csv, xlsx, xls, pptx, rtf).
- `#fb-upload-form` — the upload form; its Upload button actually sends. Drop files **or folders** on it: a dropped folder keeps its sub-folders.
- `#fb-upload-folder-input` — upload a whole folder: its tree is kept (`public/faq/returns.md` lands in the folder `public/faq`).
- `#fb-upload-into` — the folder to upload into (`public/faq`, created with the folders above it when missing); empty = the folder chosen in the filter, or none.
- Folders nest: a folder's name is its path (`public/faq`); the chips show sub-folders under their parent, picking a folder also shows its sub-folders' files, ＋ Folder takes a path, and renaming or deleting a folder takes its sub-folders along (files are never deleted). The Smart Brain index, the Export card's "Documents in the package" (choose `public/` or `public/faq/` at any depth), package manifests and an opened package's restored originals all keep the full path. API: `POST …/files/upload` with `folder_path` and one `paths` entry per file (or a file name with `/`); CLI: `fts files upload DIR --project P [--folder public]` keeps the directory tree.
- `#fb-files-table` — the library table (type, size, parse status).
- `#fb-bulk-reparse` — re-run the parser on the selected rows.
- `#fb-search` — search the file names.
- `#fb-view-trash` — switch to the trash view.

## What to check
Each file: status ready, non-zero character count, a sensible parser name. Empty text for `.xls`, `.pptx` or `.rtf` means a document-parser module is missing (`bash install.sh --check`).

## Common mistakes
Uploading scans without text (they parse to nothing); expecting a file to be usable before it is parsed.
