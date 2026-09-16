"""Back-compat shim — the real code lives in finetune_studio.data.prep.

This module re-exports the public API so older imports
(`from finetune_studio.data.prep import DataPrepRunner`) keep working.
"""

from finetune_studio.data.prep import (  # noqa: F401
    DataPrepRunner,
    PrepProgress,
    chunk_text,
    coerce_pairs,
    export_qa_jsonl,
    find_matching_bracket,
    heuristic_score,
    parse_qa_json,
    parse_qa_lines,
    style_hint,
    validate_qa_batch,
    validate_qa_pair,
)
from finetune_studio.data.prep.prompts import (  # noqa: F401
    QA_SYSTEM_PROMPT,
    QA_USER_TEMPLATE,
)
