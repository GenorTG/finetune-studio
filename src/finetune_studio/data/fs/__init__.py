"""Per-project storage on disk (under ``$FTS_ROOT/projects/<pid>/``).

LAYOUT
------
  files/raw/<kind>/<file_id>_<name>  # the file library: uploaded bytes, one per version (DB: project_files, file_versions)
  files/<sha12>/                     # one source's parse (DB: file_conversions format='parsed', see knowledge.sources)
    parsed.txt                       #   the text every consumer reads
    parsed.json                      #   the parser's full result
    chunks/NNNN.txt + manifest.json  #   the passages (mining keys items and pairs by their index)
  qa/pairs/<qa-id>.json              # mined Q&A pairs

MODULES
-------
  paths.py        — root(), project_dir(), file_dir(), path fences (resolve_in_project / resolve_within)
  chunks.py       — passage writer + manifest
  qa.py           — qa/pairs on disk
  qa_review.py    — pair review helpers (source passage lookup)
  file_library.py — DB-backed file library (raw/ + trash dirs, folders, versions)
  workbench.py    — bulk actions / export over the library
  archive_library.py — file library rows in project export/import
"""
