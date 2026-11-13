"""Knowledge check: what is wrong with a project's documents before they become a Smart Brain package.

Deterministic, no model and no network: the parsed text, the file names and, once an index exists, its stored vectors.
Every finding names its files and says what to do. ``error`` means the package would be wrong or would leak something;
``warning`` means a person should look. A finding can be ignored ("we keep both on purpose"); the ignore is stored per
project and survives a re-run, keyed on the finding's rule and files.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import lru_cache
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
    severity: str                  # error | warning | info (worth knowing, nothing to fix: a draft with no final yet)
    title: str
    fix: str
    files: list[dict[str, str]]
    detail: list[str] = field(default_factory=list)
    key: str = ""
    ignored: bool = False
    link: dict[str, str] = field(default_factory=dict)    # {"href", "label"}: where the fix is
    action: str = ""                                       # a button the page offers ("read_pictures")


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


def _check_files(sources: list[Source], texts: dict[str, str],
                 pairs: set[frozenset[str]] | None = None) -> list[Finding]:
    """``pairs``: documents linked as base + amendment (``_check_versions``): not reported as near duplicates."""
    from finetune_studio.knowledge import lint_versions as lv

    pairs = pairs or set()
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
        # A labelled password is its own finding, also next to a key the check above found: it reported the API key
        # and stayed silent about "password: hunter2hunter2" in the same file.
        hits = [m.group(0).split(m.group(1))[0].strip() + " " + m.group(1)[:2] + "…" for m in _CREDENTIAL.finditer(text)
                if not any(rx.search(m.group(1)) for _what, rx in _SECRETS)]
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
            if score >= NEAR_DUPLICATE and frozenset((a.id, b.id)) not in pairs:
                out.append(_finding("near_duplicate", "warning", f"{a.filename} and {b.filename} are nearly the same",
                                    "If one replaces the other, remove the old one; otherwise ignore this.", [a, b],
                                    [f"{round(score * 100)} % of the wording matches"]))

    flagged: set[str] = set()
    for s in ready:
        name_hit = _OLD_NAME.search(s.filename)
        text_hit = _OLD_TEXT.search(texts[s.id][:3_000])
        if name_hit and not text_hit and name_hit.group(1).lower() == "draft":
            # A draft is out of date only when its final (or a later version) is here too; a lone draft is just a draft.
            family = lv.draft_family(s.filename)
            newer = [o for o in ready if o.id != s.id and lv.draft_family(o.filename) == family
                     and not _OLD_NAME.search(o.filename)]
            if not newer:
                out.append(_finding("draft", "info", f"{s.filename} is a draft",
                                    "Nothing newer is here, so answers use it as it stands. If a final version exists, "
                                    "add it and remove the draft; if the draft is what applies, ignore this.", [s]))
                continue
            flagged.add(s.id)
            out.append(_finding("old_version", "warning", f"{s.filename} is a draft of {newer[0].filename}",
                                "Remove the draft unless answers should use it: its facts will otherwise be quoted as "
                                "current.", [s]))
            continue
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
                flagged.add(s.id)
                out.append(_finding("old_version", "warning", f"{s.filename} has a newer version (v{newest[0]})",
                                    "Remove it unless answers should use it: old facts will otherwise be quoted as current.", [s]))
    by_year: dict[str, list[tuple[int, Source]]] = defaultdict(list)   # salary_bands_2024 next to salary_bands_2025
    for s in ready:
        key = lv.year_key(s.filename) if not _version_key(s.filename) else None
        if key:
            by_year[key[0]].append((key[1], s))
    for family in by_year.values():
        newest_year, newest = max(family, key=lambda ys: ys[0])
        for year, s in family:
            if year < newest_year and s.id not in flagged:
                flagged.add(s.id)
                out.append(_finding("old_version", "warning", f"{s.filename} has a newer version ({newest.filename})",
                                    "Remove it unless answers should use it: old facts will otherwise be quoted as current.",
                                    [s]))
    return out


def _check_versions(sources: list[Source], texts: dict[str, str]) -> tuple[list[Finding], set[frozenset[str]]]:
    """Amendments linked to what they amend, and older sheets kept in a workbook (``knowledge.lint_versions``).
    Returns the findings and the (base, amendment) pairs, which the duplicate and conflict checks then leave alone."""
    from finetune_studio.knowledge import lint_versions as lv

    ready = [s for s in sources if s.ready]
    out: list[Finding] = []
    pairs: set[frozenset[str]] = set()
    for am in lv.amendments(ready, texts):
        pairs.add(frozenset((am.base.id, am.amendment.id)))
        out.append(_finding("amended", "warning", f"{am.base.filename} is amended by {am.amendment.filename}",
                            "Amended: check the amendment's terms are used. Keep both files (the amendment changes "
                            "only some terms); ask a question about a changed term and check the answer quotes the "
                            "amendment, then ignore this.", [am.base, am.amendment], am.evidence))
    for s in ready:
        if not _TABULAR.search(s.filename):
            continue
        old = lv.older_sheets(texts.get(s.id, ""))
        if old:
            out.append(_finding("old_sheet", "warning",
                                f"{s.filename} keeps an older sheet: " + ", ".join(f'"{o}"' for o, _ in old),
                                "Answers may quote the older sheet's figures as current. Delete it from the workbook "
                                "(or put the year or 'superseded' in its title) and upload it again, or ignore this "
                                "if both are kept on purpose.", [s],
                                [f'"{o}" looks older than "{n}"' for o, n in old]))
    return out, pairs


@lru_cache(maxsize=8192)
def _labelled_values(text: str) -> dict[str, frozenset[tuple[str, str]]]:
    """Extract a label, value and nearby unit; values in unlike units are not conflicts.

    Cached per passage text: the conflict check compares every similar pair of passages, and re-reading the same
    passage for each pair cost ~4 s on a 512-passage index (Package page render). Callers only read the result."""
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
    return {label: frozenset(found) for label, found in out.items()}


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


def _check_conflicts(pid: str, sources: list[Source], outdated: set[str] | None = None,
                     pairs: set[frozenset[str]] | None = None) -> list[Finding]:
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
        if frozenset((a, b)) in (pairs or set()):     # an amendment changes values of what it amends: reported as amended
            continue
        # A label with many different values is a template field ("station raised": each site's own figure), not a fact.
        unique = list(dict.fromkeys(text for label, unit, text in examples
                                    if len(label_values[(label, unit)]) <= MAX_LABEL_VALUES))
        if not unique:
            continue
        out.append(_finding("conflict", "warning", f"{by_id[a].filename} and {by_id[b].filename} may disagree",
                            "Check both. If one is out of date, correct or remove it; if both are right, ignore this.",
                            [by_id[a], by_id[b]], unique[:MAX_CONFLICT_EXAMPLES]))
    return out


def _check_pictures(pid: str, sources: list[Source]) -> list[Finding]:
    """Pictures (charts, table images, scanned pages) that only OCR read: a chart's numbers lose their labels."""
    from finetune_studio.knowledge import pictures

    st = pictures.project_status(pid)
    files = list(st["files"])
    if not files:
        return []
    by_id = {s.id: s for s in sources}
    srcs = [by_id[f["id"]] for f in files if f["id"] in by_id]
    n = sum(f["pictures_ocr"] + f["scanned_ocr"] for f in files)
    detail = [f"{f['name']}: {f['pictures_ocr']} picture(s), {f['scanned_ocr']} scanned page(s)" for f in files[:8]]
    job = st.get("job") or {}
    if st["reader"]["mode"] == "off":
        return [Finding("pictures_ocr", "warning", pictures.OCR_ONLY_NOTE.format(n=n),
                        "Set a vision reader (a local model with an mmproj, or an API vision model): pictures are then "
                        "read when files arrive, or press Read pictures here. Then Update Smart Brain.",
                        [_file(s) for s in srcs], detail, _finding("pictures_ocr", "", "", "", srcs).key,
                        link={"href": pictures.SETTINGS_LINK, "label": "Set a vision reader"})]
    fixable = [f for f in files if not f["mined"]]
    fix = (f"{st['reader']['label']} is reading them now." if job.get("running")
           else f"Press Read pictures to read them with {st['reader']['label']}, then Update Smart Brain."
           if fixable else "These files were mined: Reset them in Mining to have their pictures read.")
    return [Finding("pictures_ocr", "warning", f"{n} picture(s) not read by the vision reader yet", fix,
                    [_file(s) for s in srcs], detail, _finding("pictures_ocr", "", "", "", srcs).key,
                    action="read_pictures" if fixable and not job.get("running") else "")]


def _check_images(pid: str, sources: list[Source]) -> list[Finding]:
    """The image layer (Image search): pictures that could not be embedded, and pictures nothing describes (no OCR
    text, no Vision-reader reading, no caption or page text): only their look can find them."""
    from finetune_studio.data.fs.paths import brain_index_dir
    from finetune_studio.smart_brain import image_layer

    records, vectors, meta = image_layer.read_layer(brain_index_dir(pid))
    if not meta:
        return []
    by_id = {s.id: s for s in sources}
    out: list[Finding] = []
    failed = list(meta.get("failed") or [])
    if failed:
        srcs = [by_id[i] for i in dict.fromkeys(str(f.get("document_id") or "") for f in failed) if i in by_id]
        detail = [f"{f.get('filename') or f.get('document_id')}"
                  f"{' · ' + str(f['locator']) if f.get('locator') else ''}: {str(f.get('error') or '')[:160]}"
                  for f in failed[:8]]
        out.append(_finding("image_failed", "warning",
                            f"{int(meta.get('failed_count') or len(failed))} picture(s) could not be embedded for Image search",
                            "Open the file: a damaged or unsupported picture (e.g. EMF/WMF, HEIC without pillow-heif) is "
                            "left out of Image search. Save it as PNG or JPEG, replace the file and Update Smart Brain.",
                            srcs, detail))
    if vectors is not None:
        bare = [r for r in records if not str(r.get("description") or "").strip()
                and not str(r.get("context") or "").strip()]
        if bare:
            srcs = [by_id[i] for i in dict.fromkeys(str(r.get("document_id") or "") for r in bare) if i in by_id]
            detail = [f"{r.get('filename')} · {r.get('locator')}" for r in bare[:8]]
            out.append(_finding("image_undescribed", "info",
                                f"{len(bare)} picture(s) have no description: only their look can find them",
                                "No OCR text, Vision-reader reading or caption came with these pictures, so a question "
                                "that names what is in them (a model, a person, a number) cannot match their words. Set "
                                "a Vision reader and press Read pictures, or add a caption next to the picture.",
                                srcs, detail))
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


SUPERSEDED_HINT = "Remove them or explicitly accept each finding in the knowledge check."


def superseded_files(report: dict[str, Any]) -> list[str]:
    """Names of the documents an export must not ship unreviewed: out-of-date versions nobody accepted (ignored)."""
    return sorted({f["name"] for x in report["findings"] if x["rule"] == "old_version" and not x["ignored"]
                   for f in x["files"]})


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


# The text checks (names, versions, duplicates, conflicts) depend only on the sources' parsed text, the index files and
# the ignored findings: they are kept per project until one of those changes (the Package page and the Smart Brain page
# both run the check on every load; 120 files took 4 s).
_TEXT_CHECKS: dict[str, tuple[tuple, list[Finding]]] = {}
_CACHE_LOCK = threading.Lock()
_INDEX_FILES = ("vectors.npy", "chunks.parquet", "vectors.idx.json", "manifest.json")


def _stamp(path: Path) -> tuple[int, int]:
    try:
        st = path.stat()
        return st.st_mtime_ns, st.st_size
    except OSError:
        return 0, 0


def _fingerprint(pid: str, sources: list[Source], ignored: set[str]) -> tuple:
    corpus = brain_index_dir(pid)
    return (tuple((s.id, s.file_id, s.filename, s.status, s.text_sha, s.chunk_count, s.edited_at, s.size_bytes)
                  for s in sources),
            tuple(_stamp(corpus / name) for name in _INDEX_FILES), tuple(sorted(ignored)))


def forget_cached(pid: str | None = None) -> None:
    """Drop the cached text checks (tests; a caller that changed text outside the sources' own records)."""
    with _CACHE_LOCK:
        if pid is None:
            _TEXT_CHECKS.clear()
        else:
            _TEXT_CHECKS.pop(pid, None)


def _check_unparsable(pid: str) -> list[Finding]:
    """Files in the library no parser reads (``.exe``, ``.bin``, …): listed in Files, never in the Smart Brain. The
    check said nothing about them (walkthrough 2026-11-13 m10)."""
    from finetune_studio.data.fs import file_library as fl
    from finetune_studio.knowledge import sources as ks

    try:
        names = [f for f in fl.list_files(pid) if not ks.is_parseable(str(f.get("original_name") or ""))]
    except Exception:  # noqa: BLE001 - the library is unreadable: the other checks report what they can
        return []
    if not names:
        return []
    shown = ", ".join(str(f.get("original_name")) for f in names[:8]) + (f" (+{len(names) - 8} more)" if len(names) > 8 else "")
    key = hashlib.sha256("|".join(["unparsable", *sorted(str(f.get("id")) for f in names)]).encode()).hexdigest()[:16]
    return [Finding("unparsable", "warning", f"{len(names)} file(s) no parser can read: {shown}",
                    "They are not in the Smart Brain. Remove them in Files, or convert them to PDF, a text or an Office "
                    "file.", [{"id": "", "file_id": str(f.get("id")), "name": str(f.get("original_name"))}
                              for f in names[:20]], [], key)]


def lint(pid: str) -> dict[str, Any]:
    """Run every check. Blocking (reads every parsed file): async callers use a thread."""
    sources = [s for s in list_sources(pid) if not s.deleted]
    ignored = ignored_keys(pid)
    key = _fingerprint(pid, sources, ignored)
    with _CACHE_LOCK:
        hit = _TEXT_CHECKS.get(pid)
    if hit and hit[0] == key:
        text_findings = [copy.deepcopy(f) for f in hit[1]]
    else:
        texts = {s.id: parsed_text(pid, s) for s in sources if s.ready}
        version_findings, pairs = _check_versions(sources, texts)
        file_findings = _check_files(sources, texts, pairs)
        outdated = {f["id"] for x in file_findings if x.rule == "old_version" and x.key not in ignored for f in x.files}
        text_findings = [*file_findings, *version_findings, *_check_conflicts(pid, sources, outdated, pairs)]
        with _CACHE_LOCK:
            _TEXT_CHECKS[pid] = (key, [copy.deepcopy(f) for f in text_findings])
    findings = [*text_findings, *_check_pictures(pid, sources), *_check_images(pid, sources),
                *_check_index(pid, sources), *_check_unparsable(pid)]
    for f in findings:
        f.ignored = f.key in ignored
    findings.sort(key=lambda f: (f.ignored, {"error": 0, "warning": 1}.get(f.severity, 2), f.rule))
    active = [f for f in findings if not f.ignored]
    return {
        "files": len(sources), "reading": sum(s.status in ("queued", "parsing") for s in sources),
        "errors": sum(f.severity == "error" for f in active), "warnings": sum(f.severity == "warning" for f in active),
        "infos": sum(f.severity == "info" for f in active),
        "ignored": len(findings) - len(active), "findings": [asdict(f) for f in findings],
    }
