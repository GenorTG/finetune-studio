"""Files & Knowledge: one route for every project file, whatever it is used for (Genor 2026-10-14).

    upload ─► file library (``data.fs.file_library``: bytes, versions, folders, trash)
               │ parsed ONCE ─► ``sources``: parsed text + passages (+ Vision readings, hand edits)
               ▼
    mining ─► inventory + Q&A pairs (``data.mining``)  ─► Train: dataset ─► LoRA adapter / merged model
    passages ─► Smart Brain index (semantic + keyword search, rank fusion, reranking) ─► standalone server

Modules:
  sources.py — the one parse of each file: state, text, passages, edits, reparse, forget on purge
  status.py  — per-file pipeline flags for the file browser
  migrate.py — one-time move from the old ``qa/sources`` manifests
"""
