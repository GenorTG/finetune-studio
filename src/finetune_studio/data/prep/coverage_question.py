"""Self-contained question generation for coverage-fill (extractive) pairs.

A question is only emitted when it can be answered without seeing the chunk:
it must name a *distinctive subject* (a proper noun, a number, or at least two
non-generic content words) and, whenever the document gives us one, the scope
it lives in (section heading, file title, or the chunk's dominant entity).
Pronoun subjects ("It"), clause fragments ("Rinse with fresh water three"),
generic one-word subjects ("The body") without a scope, and near-duplicates
are rejected — the caller then drops the pair and reports the chunk as
``no_specific_question`` instead of emitting a vague question.

Pure functions only: no I/O, no model calls, deterministic.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from pathlib import PurePath

from finetune_studio.data.prep.qa_validate import content_tokens, normalize_question

# ── word lists ───────────────────────────────────────────────────────────

_PRONOUNS = frozenset({
    "it", "its", "this", "that", "these", "those", "they", "them", "their",
    "he", "she", "him", "her", "his", "we", "us", "our", "you", "your", "i",
    "there", "here", "one", "ones", "which", "who", "whom", "what",
})
_DETERMINERS = frozenset({
    "the", "a", "an", "each", "every", "any", "some", "all", "both", "no",
    "another", "such", "their", "its", "his", "her", "our", "your", "my",
})
_SUBORDINATORS = frozenset({
    "because", "if", "when", "while", "although", "though", "unless", "since",
    "whereas", "once", "until", "after", "before", "as", "so", "but", "and",
    "or", "however", "also", "then", "thus", "therefore", "otherwise",
})
_ADVERBS = frozenset({
    "never", "always", "only", "also", "just", "not", "please", "usually",
    "often", "sometimes", "rarely", "still", "even", "ever", "already",
    "afterwards", "afterward", "again", "twice", "once", "thrice", "approximately",
    "about", "around", "roughly", "nearly", "almost", "very", "much", "more",
    "most", "less", "least", "too", "quite", "fully", "mostly", "typically",
    "now", "soon", "currently", "later", "today", "next", "previously",
})
_NUMBER_WORDS = frozenset({
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "twenty", "thirty", "forty", "fifty",
    "hundred", "thousand", "million", "times", "first", "second", "third",
})
_IMPERATIVES = frozenset({
    "use", "rinse", "fill", "boil", "do", "don't", "make", "ensure", "keep",
    "check", "press", "hold", "remove", "insert", "add", "set", "turn", "open",
    "close", "place", "store", "avoid", "replace", "contact", "call", "see",
    "note", "read", "follow", "wait", "let", "allow", "clean", "wash", "dry",
    "unplug", "plug", "connect", "disconnect", "install", "run", "start",
    "stop", "send", "submit", "provide", "include", "choose", "select", "take",
    "put", "pour", "empty", "wipe", "apply", "mix", "stir", "heat", "cool",
    "ask", "register", "report", "return", "verify", "confirm", "sign",
})
_DANGLING_END = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "by", "with", "from",
    "and", "or", "but", "that", "which", "who", "whose", "as", "than", "its",
    "their", "his", "her", "is", "are", "was", "were", "be", "been", "has",
    "have", "had", "will", "would", "can", "could", "into", "over", "under",
    "because", "if", "when", "while", "it", "they", "this", "these", "those",
    "per", "not",
})
_PREPOSITIONS = frozenset({
    "above", "over", "under", "below", "in", "on", "at", "with", "for", "from",
    "by", "to", "into", "within", "without", "between", "during", "after", "before",
})
_AUX_VERBS = frozenset({
    "is", "are", "was", "were", "has", "have", "had", "will", "would", "can",
    "could", "must", "may", "might", "should", "does", "do", "did", "shall",
})
# Nouns that name nothing on their own: acceptable only inside a named scope.
_GENERIC_NOUNS = frozenset({
    "body", "damage", "support", "information", "info", "data", "note", "notes",
    "section", "text", "item", "items", "thing", "things", "process", "system",
    "product", "document", "file", "page", "part", "parts", "way", "case",
    "example", "details", "detail", "result", "results", "time", "times", "use",
    "users", "user", "customer", "customers", "people", "issue", "issues",
    "question", "questions", "answer", "answers", "service", "feature",
    "features", "option", "options", "type", "types", "list", "name", "number",
    "value", "values", "amount", "level", "model", "version", "step", "steps",
    "rule", "rules", "policy", "claim", "claims", "problem", "problems",
    "source", "chapter", "paragraph", "overview", "introduction", "summary",
})
# Headings / titles that do not narrow a scope.
_GENERIC_HEADINGS = frozenset({
    "overview", "introduction", "summary", "notes", "note", "contents",
    "details", "background", "general", "misc", "miscellaneous", "other",
    "appendix", "faq", "faqs", "abstract", "conclusion", "references",
    "table of contents",
})
_GENERIC_FILE_STEMS = frozenset({
    "document", "doc", "file", "chunk", "untitled", "new", "text", "notes",
    "readme", "output", "input", "data", "sample", "test", "draft", "copy",
    "scan", "page", "export", "download", "final", "temp", "tmp",
})

_MIN_SUBJECT_CHARS = 3
_MAX_SUBJECT_WORDS = 7
_MAX_SPAN_WORDS = 4
_DUP_JACCARD = 0.75

_TEMPLATES_SCOPED = (
    "In “{scope}”, what does the source say about {subject}?",
    "In “{scope}”, what is stated regarding {subject}?",
)
_TEMPLATES_BARE = (
    "What does the source say about {subject}?",
    "What is stated in the source regarding {subject}?",
)

_HEX_RE = re.compile(r"^[0-9a-f]{6,}$|^[0-9a-f-]{20,}$", re.IGNORECASE)
_TERMINAL_RE = re.compile(r"[.!?…][\"')”]*$")


# ── sections / headings ──────────────────────────────────────────────────

def _strip_marker(line: str) -> tuple[str, bool]:
    """Strip markdown/numbering markers; second value = line was a markdown heading."""
    s = line.strip()
    md = bool(re.match(r"#{1,6}\s+\S", s))
    s = re.sub(r"^#{1,6}\s+", "", s)
    s = re.sub(r"^(?:\d+(?:\.\d+)*[.)]|[-*•])\s+", "", s)
    return s.strip(), md


def _is_heading(line: str, *, markdown: bool) -> bool:
    if not line or len(line) > 90:
        return False
    if _TERMINAL_RE.search(line):
        return False
    words = line.rstrip(":").split()
    if not words or len(words) > 10:
        return False
    if sum(1 for w in words if re.search(r"[A-Za-z]", w)) < len(words) * 0.8:
        return False  # digits/symbols: table residue, not a title
    if line.count(",") > 1 or "|" in line:
        return False
    if markdown:
        return True
    caps = sum(1 for w in words if w[:1].isupper())
    return caps >= max(1, (len(words) + 1) // 2) and len(words) <= 8


def split_sections(chunk_text: str) -> list[tuple[str, str]]:
    """Split a chunk into ``(heading, paragraph_text)`` blocks.

    Heading lines (short, title-like, no terminal punctuation, or markdown ``#``)
    are removed from the body and attached to the text that follows them, so a
    title glued to the next sentence ("Support Handbook Customer support
    replies …") can no longer leak into a question subject. Text before the
    first heading has heading ``""``.
    """
    blocks: list[tuple[str, list[str]]] = [("", [])]
    for raw in (chunk_text or "").splitlines():
        line, md = _strip_marker(raw)
        if not line:
            continue
        if _is_heading(line, markdown=md):
            blocks.append((line.rstrip(":").strip(), []))
        else:
            blocks[-1][1].append(line)
    return [(h, " ".join(body)) for h, body in blocks if body]


# ── scope (what the chunk is about) ──────────────────────────────────────

def _clean_title(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip(" -_.:“”\"'")
    low = t.lower()
    if len(t) < 4 or low in _GENERIC_HEADINGS:
        return ""
    if not re.search(r"[A-Za-z]{3}", t):
        return ""
    return t


def title_from_filename(filename: str) -> str:
    """Readable topic from a file name; ``""`` for hashes and generic stems."""
    stem = PurePath(filename or "").stem
    if not stem or _HEX_RE.match(stem):
        return ""
    words = [w for w in re.split(r"[\s_\-.]+", stem) if w]
    words = [w for w in words if not w.isdigit() and not _HEX_RE.match(w)]
    if not words or all(w.lower() in _GENERIC_FILE_STEMS for w in words):
        return ""
    title = " ".join(words)
    return _clean_title(title.title() if title.islower() else title)


_CAP_SEQ_RE = re.compile(r"\b[A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+){0,3}")


def dominant_entity(chunk_text: str) -> str:
    """Most repeated capitalised phrase in the chunk ("Aurora Kettle"), else ``""``."""
    counts: Counter[str] = Counter()
    for m in _CAP_SEQ_RE.finditer(chunk_text or ""):
        words = m.group(0).split()
        while words and words[0].lower() in (_DETERMINERS | _SUBORDINATORS | _PRONOUNS):
            words.pop(0)  # sentence-initial "The"/"If"/"It" is not part of a name
        if not words:
            continue
        phrase = " ".join(words)
        if len(words) == 1 and (len(phrase) < 4 or phrase.lower() in _GENERIC_NOUNS):
            continue
        counts[phrase] += 1
    repeated = [(c, len(p.split()), p) for p, c in counts.items() if c >= 2]
    if not repeated:
        return ""
    repeated.sort(reverse=True)
    return repeated[0][2]


def scope_for(heading: str, filename: str, chunk_text: str) -> str:
    """Best available scope: section heading, then file title, then dominant entity."""
    return _clean_title(heading) or title_from_filename(filename) or dominant_entity(chunk_text)


# ── subject extraction ───────────────────────────────────────────────────

def _bare(word: str) -> str:
    return word.strip(",;:.!?\u201c\u201d\"'()[]").lower()


def proper_words(chunk_text: str) -> frozenset[str]:
    """Lowercased words that behave like names in this chunk.

    A word counts when it is capitalised mid-sentence, or capitalised every
    time it occurs (at least twice — "Pip" opening three sentences), never
    lowercase.
    """
    stop = _DETERMINERS | _PRONOUNS | _SUBORDINATORS | _ADVERBS
    out: set[str] = set()
    caps: Counter[str] = Counter()
    lows: Counter[str] = Counter()
    for line in (chunk_text or "").splitlines():
        for sent in re.split(r"(?<=[.!?])\s+", _strip_marker(line)[0]):
            _count_caps(sent, stop, out, caps, lows)
    out |= {b for b, n in caps.items() if n >= 2 and not lows[b] and not _verbish(b)}
    return frozenset(out)


def _count_caps(sent: str, stop: frozenset[str], out: set[str], caps: Counter[str], lows: Counter[str]) -> None:
    for i, w in enumerate(sent.split()):
        b = _bare(w)
        if len(b) < 2 or b in stop:
            continue
        if w.strip(",;:.!?\u201c\u201d\"'()[]")[:1].isupper():
            caps[b] += 1
            if i:
                out.add(b)
        else:
            lows[b] += 1


def _is_proper(word: str, *, first: bool, proper: frozenset[str]) -> bool:
    """Capitalised name word; a sentence-initial one needs evidence beyond the capital."""
    w = word.strip(",;:.!?\u201c\u201d\"'()[]")
    b = w.lower()
    if len(w) < 2 or not w[0].isupper() or b in (_DETERMINERS | _PRONOUNS | _SUBORDINATORS | _ADVERBS):
        return False
    if not first:
        return True
    return b in proper or "-" in w or any(ch.isupper() for ch in w[1:]) or any(ch.isdigit() for ch in w)


def _is_content_word(word: str) -> bool:
    b = _bare(word)
    if not b:
        return False
    if any(ch.isdigit() for ch in b):
        return True
    if len(b) < 3 or not re.fullmatch(r"[a-z][a-z'\-]*", b):
        return False
    banned = _PRONOUNS | _DETERMINERS | _SUBORDINATORS | _ADVERBS | _NUMBER_WORDS | _DANGLING_END | _AUX_VERBS
    return b not in banned and bool(content_tokens(b))  # content_tokens drops the shared stopwords


def _verbish(word: str) -> bool:
    """Participle/gerund/adverb endings: never the edge of a noun phrase."""
    b = _bare(word)
    return b.endswith(("ed", "ing", "ly")) and len(b) > 4 and not b.isupper()


def _trim_edges(words: list[str]) -> list[str]:
    ws = list(words)
    while ws and (_verbish(ws[-1]) or _bare(ws[-1]) in (_DANGLING_END | _ADVERBS | _NUMBER_WORDS | _PRONOUNS)):
        ws.pop()
    # a leading participle is an adjective ("Shared folders") only before a noun
    while ws and (
        _bare(ws[0]) in (_ADVERBS | _IMPERATIVES)
        or (_verbish(ws[0]) and not _is_proper(ws[0], first=True, proper=frozenset())
            and (len(ws) == 1 or _bare(ws[1]) in (_DETERMINERS | _PRONOUNS | _ADVERBS)))
    ):
        ws.pop(0)
    return ws


def _is_distinctive(subject: str, proper: frozenset[str]) -> bool:
    """A subject that names something on its own: a proper noun or a number."""
    words = subject.split()
    if any(any(ch.isdigit() for ch in w) for w in words):
        return True
    return any(_is_proper(w, first=False, proper=proper) or _bare(w) in proper for w in words)


def _has_content(subject: str) -> bool:
    return any(_is_content_word(w) for w in subject.split())


def _np_subject(words: list[str]) -> list[str]:
    """Noun phrase before the first verb ("The Concord" in "The Concord pays …")."""
    if not words or _bare(words[0]) in (_IMPERATIVES | _SUBORDINATORS | _ADVERBS | _PRONOUNS):
        return []
    for i in range(1, min(len(words), _MAX_SUBJECT_WORDS + 1)):
        w = _bare(words[i])
        looks_verb = w in _AUX_VERBS or (
            words[i][:1].islower()
            and len(w) > 3
            and w.endswith(("s", "ed"))
            and w not in _DANGLING_END
            and w not in _ADVERBS
        )
        if not looks_verb:
            continue
        head = words[:i]
        if any(x.endswith((",", ";", ":")) for x in head[:-1]):
            return []  # a clause boundary inside the "subject" = a fragment
        if any(_bare(x) in (_PRONOUNS | _SUBORDINATORS | _AUX_VERBS | _IMPERATIVES) for x in head[1:]):
            return []  # a verb or connector inside: that is a clause, not a subject
        for k, x in enumerate(head):
            if k and _bare(x) in _PREPOSITIONS:
                head = head[:k]  # "A reading above 5 metres" -> "A reading"
                break
        return _trim_edges(head)
    return []


def _runs(words: list[str], keep, *, limit: int) -> list[list[str]]:
    """Maximal runs of ``keep(i, word)`` tokens, split at punctuation, capped at ``limit`` words."""
    out: list[list[str]] = []
    cur: list[str] = []
    for i, w in enumerate(words):
        if keep(i, w):
            cur.append(w.strip(",;:.!?\u201c\u201d\"'()"))
            if w.endswith((",", ";", ":", ".", "!", "?")):
                out.append(cur[:limit])
                cur = []
        else:
            if cur:
                out.append(cur[:limit])
            cur = []
    if cur:
        out.append(cur[:limit])
    return [r for r in out if r]


def subject_candidates(sentence: str, proper: frozenset[str] = frozenset()) -> list[str]:
    """Candidate subjects, best first: leading noun phrase, names, "the …" phrases, numbers."""
    s = re.sub(r"\s+", " ", sentence or "").strip().rstrip(".!?")
    words = s.split()
    cands: list[list[str]] = [_np_subject(words)]
    # names: capitalised runs (a sentence-initial word needs evidence it is a name)
    names = _runs(words, lambda i, w: _is_proper(w, first=i == 0, proper=proper), limit=_MAX_SPAN_WORDS)
    for run in sorted(names, key=lambda r: -len(r)):
        # a lone name takes the plain noun(s) after it ("MAX line", "Aurora citric descaler")
        ext = list(run)
        if len(run) == 1:
            end = s.find(run[0]) + len(run[0])
            tail = s[end:].split()[:3]
            nouns: list[str] = []
            for k, w in enumerate(tail):
                plain = (_is_content_word(w) and not _verbish(w) and not _bare(w).endswith("s")
                         and _bare(w) not in _IMPERATIVES and not w[:1].isupper())
                if not plain:
                    break
                nouns.append(w.strip(",;:.!?"))
                nxt = tail[k + 1] if k + 1 < len(tail) else ""
                if w.endswith((",", ";", ":", ".")) or not nxt or _bare(nxt) in (_DANGLING_END | _PREPOSITIONS):
                    ext += nouns  # the noun run closes cleanly: the last absorbed word is a noun
                    break
        cands.append(ext)
    # "the …" noun phrases
    for i, w in enumerate(words[:-1]):
        if _bare(w) == "the":
            run = _runs(words[i + 1:], lambda _j, x: _is_content_word(x) or "-" in x, limit=3)
            if run:
                head: list[str] = []
                for k, x in enumerate(run[0]):
                    if k and _bare(x).endswith("s") and not _bare(x).endswith("ss"):
                        break  # "the warranty covers": a verb, not part of the phrase
                    head.append(x)
                head = _trim_edges(head)
                if head:
                    cands.append(["the", *head])
    # numbers with their unit/noun
    for i, w in enumerate(words):
        if any(ch.isdigit() for ch in w):
            tail = [w.strip(",;:.!?")]
            for nxt in words[i + 1:i + 3]:
                if _is_content_word(nxt) or re.fullmatch(r"[A-Za-z]{1,3}|%", _bare(nxt) or ""):
                    tail.append(nxt.strip(",;:.!?"))
                else:
                    break
                if nxt.endswith((",", ";", ":", ".")):
                    break
            cands.append(_trim_edges(tail) or tail[:1])
    seen: set[str] = set()
    out: list[str] = []
    for c in cands:
        text = " ".join(c).strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            out.append(text)
    return out


def _render_subject(subject: str, proper: frozenset[str]) -> str:
    words = subject.split()
    if not words:
        return ""
    first = _bare(words[0])
    if first in _DETERMINERS or first in _GENERIC_NOUNS or not _is_proper(words[0], first=True, proper=proper):
        words[0] = words[0][:1].lower() + words[0][1:]
    if len(words) > 1 and _bare(words[0]) in {"a", "an", "each", "every", "any", "some"}:
        words.pop(0)
    return " ".join(words).strip(" ,;:")


# ── question assembly + gate ─────────────────────────────────────────────

def _subject_in(question: str) -> str:
    m = re.search(r"(?:about|regarding) (.+)\?$", question)
    return m.group(1) if m else question


def question_issues(question: str, *, scope: str = "", proper: frozenset[str] = frozenset()) -> list[str]:
    """Why ``question`` is not self-contained/specific (empty list = acceptable).

    Used both as the emission gate and as the quality metric in tests/tools.
    """
    issues: list[str] = []
    subject = _subject_in(question)
    words = subject.split()
    if len(subject.strip()) < _MIN_SUBJECT_CHARS or not words:
        return ["subject_too_short"]
    bare_first = _bare(words[0])
    if bare_first in (_PRONOUNS | _SUBORDINATORS | _IMPERATIVES | _ADVERBS):
        issues.append("pronoun_or_clause_start")
    if _bare(words[-1]) in (_DANGLING_END | _PRONOUNS | _NUMBER_WORDS | _ADVERBS):
        issues.append("dangling_end")
    if len(words) > _MAX_SUBJECT_WORDS + 1:
        issues.append("subject_too_long")
    if "…" in question or "..." in question or "|" in question:
        issues.append("truncated_or_tabular")
    if len(words) == 1 and not _is_distinctive(subject, proper):
        issues.append("single_common_word")
    if not _has_content(subject):
        issues.append("no_content_word")
    elif not scope and not _is_distinctive(subject, proper):
        issues.append("generic_without_scope")
    elif scope and content_tokens(subject) <= content_tokens(scope):
        issues.append("repeats_scope")
    if re.search(r"\b(?:it|they|this|that|these|those)\b", subject, re.IGNORECASE) and len(words) > 1:
        issues.append("embedded_pronoun")
    return issues


def answer_is_standalone(sentence: str) -> bool:
    """False when the sentence leans on context outside itself ("It boils …", "However, …").

    The extractive answer is the sentence verbatim, so a pronoun/connector
    opener would reach training data with an antecedent nobody can see.
    """
    first = _bare(sentence.split()[0]) if sentence.split() else ""
    return first not in (_PRONOUNS | _SUBORDINATORS | {"another", "such", "both", "either", "neither"})


def build_question(
    sentence: str,
    *,
    scope: str,
    variant: int,
    proper: frozenset[str] = frozenset(),
    avoid: Iterable[str] = (),
) -> str | None:
    """Self-contained question for ``sentence`` or ``None`` (caller drops the pair).

    Tries each subject candidate in order and returns the first question that
    passes `question_issues` and is not a near-duplicate of ``avoid``.
    """
    if not answer_is_standalone(sentence):
        return None
    templates = _TEMPLATES_SCOPED if scope else _TEMPLATES_BARE
    taken = list(avoid)
    for cand in subject_candidates(sentence, proper):
        subject = _render_subject(cand, proper).rstrip("?:,;")
        if len(subject) < _MIN_SUBJECT_CHARS:
            continue
        q = templates[variant % len(templates)].format(scope=scope, subject=subject)
        if not question_issues(q, scope=scope, proper=proper) and not is_near_duplicate(q, taken):
            return q
    return None


def is_near_duplicate(question: str, others: Iterable[str]) -> bool:
    """True when ``question`` asks about the same subject as one of ``others``.

    Compares the subject phrase only (the shared scope prefix would make every
    scoped question look alike): exact match or token-set Jaccard.
    """
    norm = normalize_question(_subject_in(question))
    toks = content_tokens(_subject_in(question))
    for other in others:
        if normalize_question(_subject_in(other)) == norm:
            return True
        o = content_tokens(_subject_in(other))
        if toks and o and len(toks & o) / len(toks | o) >= _DUP_JACCARD:
            return True
    return False
