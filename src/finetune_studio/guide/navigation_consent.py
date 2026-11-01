"""Which pages the user has allowed the guide to open this turn.

WHAT THIS FILE DOES
===================
``navigate`` changes the page the user is looking at, so the guide may only call it when the user
asked for it. :func:`allowed_pages` reads the conversation and returns the page keys the user
consented to, or an empty set:

* an explicit request names its destination: "open the Files page", "show me Smart Brain",
  "let's go to Training", "can you take me to Settings?" → the named page(s);
* a bare yes to the guide's own offer ("Want me to open the Files page?" → "Yes please",
  "Sure!", "Go ahead", "Yes, take me there", "Yes — open it") → only the page that was offered;
* anything else is a question, not a request: "Open files: what formats are supported?",
  "OK, open source models - which work best?", "Can you open-source the model?", or a refusal
  ("Don't, please. Open nothing.") → nothing.

A verb needs a destination right after it: matching the verb alone let "open source" and
"open-weight" switch pages, and reading only the last message lost the offer a "Yes" answers.
"""
from __future__ import annotations

import re

from finetune_studio.guide.registry import PAGES

#: Extra names users call a page by (the page titles are added automatically).
_EXTRA_ALIASES: dict[str, tuple[str, ...]] = {
    "dashboard": ("home", "home page", "start page"),
    "projects": ("project list", "projects list"),
    "models_library": ("model library", "models library", "hugging face", "hf models", "model browser"),
    "my_models": ("my models", "models", "downloaded models"),
    "inference": ("inference page",),
    "settings": ("settings", "app settings", "global settings"),
    "overview": ("project overview", "overview", "project home", "project page", "home"),
    "wizard": ("quick work", "wizard", "quick-work"),
    "files": ("files", "file page", "documents", "uploads", "upload page", "file library"),
    "pairs": ("pairs", "q&a", "q&a pairs", "question-answer pairs", "questions", "qa pairs", "data prep"),
    "brain": ("smart brain", "brain", "rag"),
    "training": ("training", "train page", "training page"),
    "testing": ("testing", "tests", "test page"),
    "compare": ("compare", "comparison"),
    "benchmarks": ("benchmarks", "benchmark"),
    "export": ("export", "exports", "export page"),
    "chat": ("chat",),
    "project_settings": ("project settings", "settings"),
    "project_models": ("project models", "trained models", "models"),
}

_WORD_FILLERS = r"(?:(?:up|the|my|our|this|that|a)\s+)*"
_VERBS = (
    r"open|go to|go over to|head to|navigate to|switch to|jump to|take me to|bring me to|show me"
    r"|show|bring up|pull up|let'?s go to|let us go to|i want to go to|i'd like to go to|i would like to go to"
    r"|i want to see|i'd like to see|take me"
)
#: A navigation verb at the start of a sentence (after an optional "yes," / "please"), or in a
#: polite request ("can you …"). The verb must be followed by whitespace: "open-source" is not one.
_REQUEST = re.compile(
    r"(?:^|(?<=[.!?;—–-])\s*|(?<=,)\s*)"
    r"(?P<polite>(?:(?:yes|yeah|yep|sure|ok|okay|alright|great)\b[\s,;:!—–-]*)*(?:please\s+)?"
    r"(?:(?:can|could|would|will) you\s+(?:please\s+)?)?)"
    rf"(?P<verb>{_VERBS})\s+(?P<rest>.*)",
    re.IGNORECASE,
)
#: Words that point back at whatever the guide just offered.
_BACK_REFERENCE = re.compile(r"^(?:it|there|that(?: page| one)?|this(?: page| one)?|them)\b", re.IGNORECASE)
_REFUSAL = re.compile(
    r"^\s*(?:no|nope|nah|not now|not yet|don'?t|do not|never|cancel|stop)\b"
    r"|\b(?:don'?t|do not|never|without)\s+(?:please\s+)?(?:open|navigate|go|switch|take|bring|show)\b"
    r"|\b(?:open|show|go to)\s+nothing\b",
    re.IGNORECASE,
)
_AFFIRMATIVE_WORDS = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "alright", "please", "do", "it", "go", "ahead",
    "that", "would", "be", "great", "thanks", "thank", "you", "sounds", "good", "absolutely", "definitely",
    "fine", "let's", "lets",
}
_AFFIRMATIVE_CORE = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "alright", "please", "ahead",
                     "absolutely", "definitely"}
#: The guide offering a page switch: "Want me to open …", "Shall I take you to …", "I can open …".
_OFFER = re.compile(
    r"\b(?:want me to|shall i|should i|would you like me to|do you want me to|i can|i could|let me"
    r"|happy to|want to (?:go|head|jump|switch))\b[^.?!]*?"
    r"\b(?:open|take you|bring you|navigate|switch|show you|go to|head to|jump to)\b(?P<rest>[^?!\n]*)",
    re.IGNORECASE,
)


def _aliases() -> list[tuple[str, frozenset[str]]]:
    """Every alias → the page keys it may mean, longest alias first (so "project settings" wins)."""
    table: dict[str, set[str]] = {}
    for key, page in PAGES.items():
        names = {page.title.lower(), *(a.lower() for a in _EXTRA_ALIASES.get(key, ()))}
        for name in names:
            table.setdefault(name, set()).add(key)
    return sorted(((a, frozenset(k)) for a, k in table.items()), key=lambda item: -len(item[0]))


_ALIASES = _aliases()


def _pages_at_start(text: str) -> set[str]:
    """Pages named at the very start of ``text`` (after "the", "my", "up" …)."""
    rest = re.sub(rf"^{_WORD_FILLERS}", "", text.strip(), flags=re.IGNORECASE).lower()
    for alias, keys in _ALIASES:
        if re.match(rf"{re.escape(alias)}(?![\w&-])", rest):
            return set(keys)
    return set()


def _first_page_named(text: str) -> set[str]:
    """The page an offer names first in ``text`` (later names are context, not the offer)."""
    low = text.lower()
    best: tuple[int, int, frozenset[str]] | None = None
    for alias, keys in _ALIASES:
        m = re.search(rf"(?<![\w&-]){re.escape(alias)}(?![\w&-])", low)
        if m and (best is None or (m.start(), -len(alias)) < (best[0], best[1])):
            best = (m.start(), -len(alias), keys)
    return set(best[2]) if best else set()


def _offered_pages(assistant_text: str) -> set[str]:
    pages: set[str] = set()
    for m in _OFFER.finditer(assistant_text or ""):
        pages |= _first_page_named(m.group("rest"))
    return pages


def _sentences(text: str) -> list[str]:
    return [s for s in re.findall(r"[^.!?]+[.!?]?", text) if s.strip()]


def _is_bare_affirmative(text: str) -> bool:
    words = re.findall(r"[a-z']+", text.lower())
    return bool(words) and set(words) <= _AFFIRMATIVE_WORDS and bool(set(words) & (_AFFIRMATIVE_CORE | {"do"}))


def allowed_pages(messages: list[dict]) -> set[str]:
    """Page keys the latest user turn allows ``navigate`` to open (empty: none)."""
    last = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    if last < 0:
        return set()
    user = str(messages[last].get("content") or "").strip()
    if not user or _REFUSAL.search(user):
        return set()
    previous = next((str(m.get("content") or "") for m in reversed(messages[:last])
                     if m.get("role") == "assistant"), "")
    offered = _offered_pages(previous)
    if _is_bare_affirmative(user):
        return offered
    allowed: set[str] = set()
    for sentence in _sentences(user):
        m = _REQUEST.search(sentence.strip())
        if not m:
            continue
        polite = bool(re.search(r"\b(?:can|could|would|will) you\b", m.group("polite"), re.IGNORECASE))
        if sentence.rstrip().endswith("?") and not polite:
            # "Open files: what formats are supported?" asks about a thing; it is not a request.
            continue
        rest = m.group("rest")
        if _BACK_REFERENCE.match(re.sub(rf"^{_WORD_FILLERS}", "", rest.strip(), flags=re.IGNORECASE)) \
                or m.group("verb").lower() == "take me" and not _pages_at_start(rest):
            allowed |= offered
            continue
        allowed |= _pages_at_start(rest)
    return allowed
