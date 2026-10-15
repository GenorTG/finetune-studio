"""Data prep building blocks: parse + chunk files, validate and store Q&A pairs, export datasets.

Mining itself (inventory, mine, check, fix per file) lives in ``finetune_studio.data.mining``.

LAYOUT
======
data/prep/
  __init__.py      — public API re-exports
  chunker.py       — paragraph/sentence-aware text splitter
  parsers.py       — parse raw model JSON / line-list Q&A output
  qa_validate.py   — strict deterministic Q&A gates + the on-disk pair record
  scorer.py        — heuristic quality score for a single Q&A pair
  generator.py     — resolve the seated helper's chat callable (helper only, never loads a local GGUF)
  dedupe.py        — near-duplicate pending pairs
  export.py        — JSONL exporters (sharegpt / alpaca / openai) + dedupe
  dataset_build.py — completeness gate → export → registered dataset
"""

from finetune_studio.data.prep.chunker import chunk_text
from finetune_studio.data.prep.export import export_qa_jsonl
from finetune_studio.data.prep.parsers import (
    coerce_pairs,
    find_matching_bracket,
    parse_qa_json,
    parse_qa_lines,
)
from finetune_studio.data.prep.qa_validate import (
    VALIDATION_VERSION,
    BatchValidationResult,
    PairValidation,
    Provenance,
    RejectionCounters,
    build_qa_record,
    validate_qa_batch,
    validate_qa_pair,
)
from finetune_studio.data.prep.scorer import heuristic_score

__all__ = [
    "VALIDATION_VERSION",
    "BatchValidationResult",
    "PairValidation",
    "Provenance",
    "RejectionCounters",
    "build_qa_record",
    "chunk_text",
    "coerce_pairs",
    "export_qa_jsonl",
    "find_matching_bracket",
    "heuristic_score",
    "parse_qa_json",
    "parse_qa_lines",
    "validate_qa_batch",
    "validate_qa_pair",
]
