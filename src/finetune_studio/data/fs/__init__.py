"""Structured per-project filesystem.

LAYOUT (under $FTS_ROOT/projects/<pid>/, default ~/.finetune-studio):

  project.json                     # project metadata
  files/<sha256-12>/               # content-addressed file store (dedupes)
    <original-filename>            # immutable copy of uploaded bytes
    parsed.txt                     # plain text (for chunking into Q&A)
    parsed.json                    # structured + metadata
    metadata.json                  # {sha256, original_filename, mime, char_count, ...}
    chunks/                        # semantic chunks (one file per chunk)
      0000.txt
      0001.txt
      ...
      manifest.json                # [{index, char_count, source_section}, ...]
  qa/                              # Q&A pairs (one file per pair)
    pairs/<qa-id>.json
    sources/<source-id>.json
  logs/                            # append-only audit trail
    ingestions.jsonl

DESIGN
------
- Content-addressed (sha256) storage means uploading the same file twice
  doesn't create duplicate copies.
- The raw copy is never modified; parsed.*, metadata.json and chunks/ are
  rewritten on re-parse (write_chunks clears old chunks first).
- The audit log is the source of truth for "what happened to this project".
- The original filename is preserved on disk under files/<sha>/<name>.
  When the same content arrives under a different name, the old name is
  moved into metadata.aliases[].

LAYOUT
------
data/fs/
  __init__.py     — re-exports the public API (back-compat with the old project_filesystem.py)
  paths.py        — root path helpers, root(), project_dir(), file_dir()
  project.py      — project.json read/write
  files.py        — content-addressed file storage (store_file, list_files, delete)
  parsed.py       — parsed.txt / parsed.json writers
  chunks.py       — chunk writer + manifest
  metadata.py     — FileMetadata dataclass + read/update + safe filenames
  ingestion.py    — logs/ingestions.jsonl append + read
  qa.py           — qa/pairs and qa/sources on disk
  file_library.py — DB-backed file library (raw/ + converted/ + trash dirs)
  workbench.py    — bulk actions / export over the library
"""
from finetune_studio.data.fs.chunks import write_chunks  # noqa: F401, I001
from finetune_studio.data.fs.files import delete_file, list_files, store_file  # noqa: F401
from finetune_studio.data.fs.ingestion import log_ingestion, read_ingestion_log  # noqa: F401
from finetune_studio.data.fs.metadata import (  # noqa: F401
    FileMetadata,
    _safe_filename,
    hash_bytes,
    read_file_metadata,
    update_file_metadata,
)
from finetune_studio.data.fs.parsed import write_parsed_outputs  # noqa: F401
from finetune_studio.data.fs.paths import file_dir, project_dir, root  # noqa: F401
from finetune_studio.data.fs.project import read_project_json, write_project_json  # noqa: F401
from finetune_studio.data.fs.qa import (  # noqa: F401
    delete_qa_source,
    list_qa_pairs,
    list_qa_sources,
    read_qa_source,
    register_qa_source,
    update_qa_pair,
    write_qa_pair,
    write_qa_source,
)
