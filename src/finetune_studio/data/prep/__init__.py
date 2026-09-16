"""AI-powered data prep pipeline.

PIPELINE
========
upload bytes -> store content-addressed -> parse via dedicated script ->
write parsed.txt / parsed.json / chunks/ -> model generates Q&A per chunk ->
strict validate -> persist accepted Q&A to qa/pairs/<id>.json + log ingestions

LAYOUT
======
data/prep/
  __init__.py    — public API (re-exports the back-compat names)
  chunker.py     — paragraph/sentence-aware text splitter
  prompts.py     — system + user prompts for Q&A generation
  parsers.py     — parse raw model JSON / line-list Q&A output
  qa_validate.py — strict deterministic post-parse Q&A gates
  scorer.py      — heuristic quality score for a single Q&A pair
  generator.py   — resolve chat callable (manager or inference_engine)
  ingest.py      — shared parse+chunk (promote path + DataPrepRunner)
  runner.py      — DataPrepRunner + PrepProgress (the orchestrator)
  export.py      — JSONL exporters (sharegpt / alpaca / openai)
"""

from finetune_studio.data.prep.chunker import chunk_text
from finetune_studio.data.prep.export import export_qa_jsonl
from finetune_studio.data.prep.ingest import (
    IngestResult,
    ensure_qa_source_parsed,
    parse_and_chunk,
)
from finetune_studio.data.prep.parsers import (
    coerce_pairs,
    find_matching_bracket,
    parse_qa_json,
    parse_qa_lines,
)
from finetune_studio.data.prep.prompts import (
    QA_SYSTEM_PROMPT,
    QA_USER_TEMPLATE,
    style_hint,
)
from finetune_studio.data.prep.qa_validate import (
    VALIDATION_VERSION,
    BatchValidationResult,
    CoverageTracker,
    PairValidation,
    Provenance,
    RejectionCounters,
    build_qa_record,
    validate_qa_batch,
    validate_qa_pair,
)
from finetune_studio.data.prep.runner import DataPrepRunner, PrepProgress
from finetune_studio.data.prep.scorer import heuristic_score

__all__ = [
    "QA_SYSTEM_PROMPT",
    "QA_USER_TEMPLATE",
    "VALIDATION_VERSION",
    "BatchValidationResult",
    "CoverageTracker",
    "DataPrepRunner",
    "IngestResult",
    "PairValidation",
    "PrepProgress",
    "Provenance",
    "RejectionCounters",
    "build_qa_record",
    "chunk_text",
    "coerce_pairs",
    "ensure_qa_source_parsed",
    "export_qa_jsonl",
    "find_matching_bracket",
    "heuristic_score",
    "parse_and_chunk",
    "parse_qa_json",
    "parse_qa_lines",
    "style_hint",
    "validate_qa_batch",
    "validate_qa_pair",
]
