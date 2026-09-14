"""Back-compat shim — the real code lives in finetune_studio.data.fs.

This module re-exports the public API from `data/fs/` so older imports
(`from finetune_studio.data.project_filesystem import store_file`) keep working.
"""

from finetune_studio.data.fs import (  # noqa: F401
    FileMetadata,
    delete_file,
    delete_qa_source,
    file_dir,
    hash_bytes,
    list_files,
    list_qa_pairs,
    list_qa_sources,
    log_ingestion,
    project_dir,
    read_file_metadata,
    read_ingestion_log,
    read_project_json,
    read_qa_source,
    register_qa_source,
    root,
    store_file,
    update_file_metadata,
    update_qa_pair,
    write_chunks,
    write_parsed_outputs,
    write_project_json,
    write_qa_pair,
    write_qa_source,
)
