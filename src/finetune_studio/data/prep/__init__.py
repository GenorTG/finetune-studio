"""AI-powered data prep pipeline.

PIPELINE
========
upload bytes -> store content-addressed -> parse via dedicated script ->
write parsed.txt / parsed.json / chunks/ -> model generates Q&A per chunk ->
persist Q&A to qa/pairs/<id>.json + log to logs/ingestions.jsonl

LAYOUT
======
data/prep/
  __init__.py    — public API (re-exports the back-compat names)
  chunker.py     — paragraph/sentence-aware text splitter
  prompts.py     — system + user prompts for Q&A generation
  parsers.py     — parse raw model JSON / line-list Q&A output
  scorer.py      — heuristic quality score for a single Q&A pair
  runner.py      — DataPrepRunner + PrepProgress (the orchestrator)
  export.py      — JSONL exporters (sharegpt / alpaca / openai)
"""

from finetune_studio.data.prep.chunker import chunk_text
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
from finetune_studio.data.prep.scorer import heuristic_score
from finetune_studio.data.prep.runner import DataPrepRunner, PrepProgress
from finetune_studio.data.prep.export import export_qa_jsonl
