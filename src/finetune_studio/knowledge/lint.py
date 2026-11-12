"""Knowledge check: what is wrong with a project's documents before they become a Smart Brain package.

Deterministic, no model and no network: the parsed text, the file names and, once an index exists, its stored vectors.
Every finding names its files and says what to do. ``error`` means the package would be wrong or would leak something;
``warning`` means a person should look. A finding can be ignored ("we keep both on purpose"); the ignore is stored per
project and survives a re-run, keyed on the finding's rule and files.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from finetune_studio.data.fs.paths import brain_index_dir, project_dir
from finetune_studio.knowledge import index as knowledge_index
from finetune_studio.knowledge.sources import Source, list_sources, parsed_text

THIN_CHARS = 200
SKETCH = 256
NEAR_DUPLICATE = 0.8
SIMILAR_PASSAGES = 0.86          # cosine: "these two passages are about the same thing"
MAX_PASSAGES_COMPARED = 20_000
MAX_CONFLICT_EXAMPLES = 3
MAX_LABEL_VALUES = 3
TEMPLATE_DOCS = 4                # a label with its own value in this many documents is a form field

_TABULAR = re.compile(r"\.(csv|tsv|xlsx?|ods|json|jsonl)$", re.IGNORECASE)
_OLD_NAME = re.compile(r"(?:^|[\W_])(superseded|obsolete|deprecated|outdated|old|archived?|draft|backup|copy)(?:[\W_]|$)", re.IGNORECASE)
_OLD_TEXT = re.compile(r"\b(superseded by|this (?:document|version|policy|handbook) is (?:superseded|obsolete|no longer)"
                       r"|no longer in force|obsolete version)\b", re.IGNORECASE)
_VERSION = re.compile(r"(?:^|[\W_])v(?:er(?:sion)?)?[\W_]?(\d{1,3})(?:[\W_]|$)", re.IGNORECASE)
# A year (or a date starting with one) in a file name: "handbook_v1_2023", "terms_v4_2025-05-01"
_NAME_YEAR = re.compile(r"(?<![^\W_])((?:19|20)\d{2})(?:[-_.]?\d{2}){0,2}(?![^\W_])")
_SECRETS = (
    ("a private key", re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----")),
    ("an AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("an API key", re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{30,}|xox[abp]-[A-Za-z0-9-]{20,})\b")),
)
_CREDENTIAL = re.compile(r"\b(?:password|passwd|pwd|api[ _-]?key|secret[ _-]?key|access[ _-]?token)\b\s*[:=]\s*[\"']?([^\s\"',;]{6,})", re.IGNORECASE)
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
# A value with the words right before it: "annual leave entitlement 21 days" -> ("annual leave entitlement", "21")
_LABELLED_VALUE = re.compile(r"((?:[^\W\d_]{3,}[ \t]+){2})(?:is|of|at|to|are|was|for)?[ \t]*(?:EUR|PLN|USD|€|\$)?[ \t]*(\d[\d.,]*\d|\d)(?![\d/:-])", re.UNICODE)
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_DATE_LIKE = re.compile(rf"\d{{1,4}}[./-]\d{{1,2}}[./-]\d{{1,4}}|\d{{1,2}}\s+{_MONTH}\b", re.IGNORECASE)
# A label is two content words; anything else ("the button", "writing within") matches unrelated sentences.
_LABEL_NOISE = frozenset(["the", "and", "for", "with", "from", "that", "this", "than", "per", "page", "section", "clause", "version", "figure", "table", "item", "step", "see", "annex", "appendix", "rev", "revision", "within", "above", "below", "before", "after", "over", "under", "into", "about", "least", "most", "only", "first", "last", "each", "every", "more", "less", "their", "your", "our", "has", "have", "been", "will", "must", "shall", "may", "can", "not", "are", "was", "were", "its", "all", "any", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "days", "day", "hours", "hour", "weeks", "week", "months", "month", "years", "year", "up", "off", "down", "out"])


@dataclass
class Finding:
    rule: str
    severity: str                  # error | warning
    title: str
    fix: str
    files: list[dict[str, str]]
    detail: list[str] = field(default_factory=list)
    key: str = ""
    ignored: bool = False


def _file(s: Source) -> dict[str, str]:
    return {"id": s.id, "file_id": s.file_id, "name": s.filename}


def _finding(rule: str, severity: str, title: str, fix: str, sources: list[Source], detail: list[str] | None = None) -> Finding:
    ids = sorted(s.id for s in sources)
    return Finding(rule, severity, title, fix, [_file(s) for s in sources], detail or [],
                   hashlib.sha256("|".join([rule, *ids]).encode()).hexdigest()[:16])


def _garbled(text: str) -> bool:
    sample = text[:20_000]
    if not sample:
        return False
    bad = sum(c == "�" or (ord(c) < 32 and c not in "\n\r\t") for c in sample)
    letters = sum(c.isalpha() for c in sample)
    return bad / len(sample) > 0.01 or letters / len(sample) < 0.35


def _sketch(text: str) -> frozenset[int]:
    """Bottom-k hashes of the word 5-grams: comparing two documents costs a set intersection, whatever their length."""
    words = [w.lower() for w in _WORD.findall(text)]
    grams = {hash(tuple(words[i:i + 5])) for i in range(max(0, len(words) - 4))}
    return frozenset(sorted(grams)[:SKETCH])


def _similarity(a: frozenset[int], b: frozenset[int]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _version_key(name: str) -> tuple[str, int, int | None] | None:
    """(family, version, year in the name) of a versioned file name. The year is not part of the family: a new
    version is usually dated too ("handbook_v1_2023" -> "handbook_v2_2025")."""
    stem = re.sub(r"\.[^.]+$", "", name)   # first: the same document can be a .docx in one version and a .pdf in the next
    match = _VERSION.search(stem)
    if not match:
        return None
    rest = _VERSION.sub("_", stem, count=1)
    year = _NAME_YEAR.search(rest)
    family = re.sub(r"[\W_]+", "", _NAME_YEAR.sub("_", rest)).lower()
    return family, int(match.group(1)), int(year.group(1)) if year else None


def _older(version: int, year: int | None, newest: tuple[int, int | None]) -> bool:
    """A lower version is older unless its name carries a later year (budget_2025_v1 is not older than budget_2024_v3)."""
    return version < newest[0] and (year is None or newest[1] is None or year <= newest[1])


def _check_files(sources: list[Source], texts: dict[str, str]) -> list[Finding]:
    out: list[Finding] = []
    for s in sources:
        if s.status == "error" or (s.status == "ready" and s.chunk_count == 0):
            out.append(_finding("unreadable", "error", f"{s.filename} could not be read",
                                "Open it in Files, fix or replace it, or remove it. Nothing from it is in the Smart Brain.",
                                [s], [s.error] if s.error else []))
    ready = [s for s in sources if s.ready]
    for s in ready:
        text = texts[s.id]
        if len(text.strip()) < THIN_CHARS:
            out.append(_finding("thin", "warning", f"{s.filename} has almost no text ({len(text.strip())} characters)",
                                "If it is a scan or a picture, have it read in Files. Otherwise remove it.", [s]))
        elif not _TABULAR.search(s.filename) and _garbled(text):
            out.append(_finding("garbled", "warning", f"{s.filename} looks garbled",
                                "Check its text in Files: the parser may have read it badly. Edit it or reparse.", [s]))
        leaks = [what for what, rx in _SECRETS if rx.search(text)]
        if leaks:
            out.append(_finding("secret", "error", f"{s.filename} contains {' and '.join(leaks)}",
                                "Remove it from the text in Files before you package this: whoever gets the package can read it.", [s]))
        else:
            hits = [m.group(0).split(m.group(1))[0].strip() + " " + m.group(1)[:2] + "…" for m in _CREDENTIAL.finditer(text)]
            if hits:
                out.append(_finding("credential", "warning", f"{s.filename} seems to contain a password or key",
                                    "Check it, and remove it in Files if it is real.", [s], hits[:3]))

    by_text: dict[str, list[Source]] = defaultdict(list)
    for s in ready:
        by_text[s.text_sha].append(s)
    exact_groups = [group for sha, group in by_text.items() if sha and len(group) > 1]
    for group in exact_groups:
        names = ", ".join(s.filename for s in group)
        out.append(_finding("duplicate", "warning", f"Same text in {len(group)} files: {names}",
                            "Keep one: duplicates crowd out other passages in the search results.", group))
    exact = {s.id for group in exact_groups for s in group}
    candidates = [s for s in ready if s.id not in exact]
    sketches = {s.id: _sketch(texts[s.id]) for s in candidates}
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            score = _similarity(sketches[a.id], sketches[b.id])
            if score >= NEAR_DUPLICATE:
                out.append(_finding("near_duplicate", "warning", f"{a.filename} and {b.filename} are nearly the same",
                                    "If one replaces the other, remove the old one; otherwise ignore this.", [a, b],
                                    [f"{round(score * 100)} % of the wording matches"]))

    flagged: set[str] = set()
    for s in ready:
        name_hit = _OLD_NAME.search(s.filename)
        text_hit = _OLD_TEXT.search(texts[s.id][:3_000])
        if name_hit or text_hit:
            flagged.add(s.id)
            why = f'the file name says "{name_hit.group(1)}"' if name_hit else f'the text says "{text_hit.group(0)}"'
            out.append(_finding("old_version", "warning", f"{s.filename} looks out of date ({why})",
                                "Remove it unless answers should use it: old facts will otherwise be quoted as current.", [s]))
    versions: dict[str, list[tuple[int, int | None, Source]]] = defaultdict(list)
    for s in ready:
        key = _version_key(s.filename)
        if key:
            versions[key[0]].append((key[1], key[2], s))
    for family in versions.values():
        newest = max(((v, y) for v, y, _ in family), key=lambda vy: vy[0])
        for version, year, s in family:
            if _older(version, year, newest) and s.id not in flagged:
                out.append(_finding("old_version", "warning", f"{s.filename} has a newer version (v{newest[0]})",
                                    "Remove it unless answers should use it: old facts will otherwise be quoted as current.", [s]))
    return out


def _labelled_values(text: str) -> dict[str, set[tuple[str, str]]]:
    """Extract a label, value and nearby unit; values in unlike units are not conflicts."""
    out: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for match in _LABELLED_VALUE.finditer(text):
        words = match.group(1).lower().split()
        if set(words) & _LABEL_NOISE or _DATE_LIKE.match(text, match.start(2)):
            continue
        prefix = match.group(0).upper()
        suffix = text[match.end(2):match.end(2) + 24]
        unit_match = re.match(r"\s*(months?|days?|weeks?|years?|hours?|minutes?|eur|pln|usd|%|percent|employees?)(?=\W|$)",
                              suffix, re.IGNORECASE)
        unit = unit_match.group(1).lower().rstrip("s") if unit_match else ""
        currency = next((code.lower() for code in ("EUR", "PLN", "USD") if code in prefix), "")
        currency = currency or ("eur" if "€" in prefix else "usd" if "$" in prefix else "")
        unit = unit or currency.lower()
        out[" ".join(words)].add((match.group(2).rstrip(".,").replace(",", ""), unit))
    return out


def _template_labels(texts: list[str], docs: list[str]) -> set[tuple[str, str]]:
    """Labels that are a field of a repeated form ("station raised 932 EUR"), not a fact: many documents state them,
    each with its own value. Read over the whole index, not only the pairs that clash."""
    values: dict[tuple[str, str], set[str]] = defaultdict(set)
    holders: dict[tuple[str, str], set[str]] = defaultdict(set)
    for text, doc in zip(texts, docs, strict=True):
        for label, found in _labelled_values(text).items():
            for value, unit in found:
                values[(label, unit)].add(value)
                holders[(label, unit)].add(doc)
    return {key for key, docs_with in holders.items()
            if len(docs_with) >= TEMPLATE_DOCS and len(values[key]) > MAX_LABEL_VALUES}


def _check_conflicts(pid: str, sources: list[Source], outdated: set[str] | None = None) -> list[Finding]:
    """Passages of different files that say the same thing with a different number for the same label.

    A pair with a file already reported as out of date is left out: that finding says what to do about it."""
    corpus = brain_index_dir(pid)
    if not ((corpus / "vectors.npy").is_file() and (corpus / "chunks.parquet").is_file()):
        return []
    import numpy as np
    import pandas as pd

    chunks = pd.read_parquet(corpus / "chunks.parquet", columns=["id", "document_id", "text"])
    idx = json.loads((corpus / "vectors.idx.json").read_text(encoding="utf-8"))
    vectors = np.load(corpus / "vectors.npy", mmap_mode="r")
    if len(chunks) < 2 or len(chunks) > MAX_PASSAGES_COMPARED or not len(vectors):
        return []
    rows = np.array([idx[c] for c in chunks["id"]])
    matrix = np.asarray(vectors[rows], dtype=np.float32)
    docs = chunks["document_id"].to_numpy()
    texts = chunks["text"].tolist()
    by_id = {s.id: s for s in sources if s.id not in (outdated or set())}
    template = _template_labels(texts, [str(d) for d in docs])
    found: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    label_values: dict[tuple[str, str], set[str]] = defaultdict(set)
    for start in range(0, len(matrix), 512):
        block = matrix[start:start + 512] @ matrix.T
        for offset, scores in enumerate(block):
            i = start + offset
            for j in np.nonzero(scores[i + 1:] >= SIMILAR_PASSAGES)[0] + i + 1:
                if docs[i] == docs[j] or docs[i] not in by_id or docs[j] not in by_id:
                    continue
                left, right = _labelled_values(texts[i]), _labelled_values(texts[j])
                for label in left.keys() & right.keys():
                    left_by_unit: dict[str, set[str]] = defaultdict(set)
                    right_by_unit: dict[str, set[str]] = defaultdict(set)
                    for value, unit in left[label]:
                        left_by_unit[unit].add(value)
                    for value, unit in right[label]:
                        right_by_unit[unit].add(value)
                    for unit in left_by_unit.keys() & right_by_unit.keys():
                        if left_by_unit[unit] & right_by_unit[unit] or (label, unit) in template:
                            continue
                        pair = tuple(sorted((str(docs[i]), str(docs[j]))))
                        a, b = (left, right) if pair[0] == docs[i] else (right, left)
                        av, bv = left_by_unit[unit], right_by_unit[unit]
                        found[pair].append((label, unit,
                                            f'"{label}": {"/".join(sorted(av))} vs {"/".join(sorted(bv))}'))
                        label_values[(label, unit)] |= av | bv
    out = []
    for (a, b), examples in found.items():
        # A label with many different values is a template field ("station raised": each site's own figure), not a fact.
        unique = list(dict.fromkeys(text for label, unit, text in examples
                                    if len(label_values[(label, unit)]) <= MAX_LABEL_VALUES))
        if not unique:
            continue
        out.append(_finding("conflict", "warning", f"{by_id[a].filename} and {by_id[b].filename} may disagree",
                            "Check both. If one is out of date, correct or remove it; if both are right, ignore this.",
                            [by_id[a], by_id[b]], unique[:MAX_CONFLICT_EXAMPLES]))
    return out


def _check_index(pid: str, sources: list[Source]) -> list[Finding]:
    st = knowledge_index.state(pid)
    if not st.built or st.fresh:
        return []
    by_id = {s.id: s for s in sources}
    changed = [by_id[i] for i in [*st.stale, *st.missing] if i in by_id]
    if not changed and not st.removed:
        return []
    return [_finding("index_stale", "warning", "The index does not match your files",
                     "Press Update Smart Brain: answers still use the old index.", changed,
                     [f"{len(changed)} new or changed", f"{len(st.removed)} removed"])]


def _ignored_path(pid: str) -> Path:
    return project_dir(pid) / "brain_lint_ignored.json"


def ignored_keys(pid: str) -> set[str]:
    try:
        return set(json.loads(_ignored_path(pid).read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return set()


def set_ignored(pid: str, key: str, ignored: bool) -> None:
    keys = ignored_keys(pid)
    (keys.add if ignored else keys.discard)(key)
    path = _ignored_path(pid)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(sorted(keys)), encoding="utf-8")
    os.replace(tmp, path)


def excluded_documents(pid: str) -> dict[str, str]:
    """Documents a quiz should not ask about: out of date, or a duplicate of another file (unless ignored).
    {source id: why}. The first file of a duplicate group is kept."""
    report = lint(pid)
    out: dict[str, str] = {}
    for f in report["findings"]:
        if f["ignored"]:
            continue
        if f["rule"] == "old_version":
            out.update({x["id"]: "out of date" for x in f["files"]})
        elif f["rule"] in ("duplicate", "near_duplicate"):
            out.update({x["id"]: "duplicate" for x in f["files"][1:]})
    return out


def lint(pid: str) -> dict[str, Any]:
    """Run every check. Blocking (reads every parsed file): async callers use a thread."""
    sources = [s for s in list_sources(pid) if not s.deleted]
    texts = {s.id: parsed_text(pid, s) for s in sources if s.ready}
    ignored = ignored_keys(pid)
    file_findings = _check_files(sources, texts)
    outdated = {f["id"] for x in file_findings if x.rule == "old_version" and x.key not in ignored for f in x.files}
    findings = [*file_findings, *_check_conflicts(pid, sources, outdated), *_check_index(pid, sources)]
    for f in findings:
        f.ignored = f.key in ignored
    findings.sort(key=lambda f: (f.ignored, f.severity != "error", f.rule))
    active = [f for f in findings if not f.ignored]
    return {
        "files": len(sources), "reading": sum(s.status in ("queued", "parsing") for s in sources),
        "errors": sum(f.severity == "error" for f in active), "warnings": sum(f.severity == "warning" for f in active),
        "ignored": len(findings) - len(active), "findings": [asdict(f) for f in findings],
    }
