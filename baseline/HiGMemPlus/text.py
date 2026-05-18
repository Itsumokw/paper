"""Small deterministic text utilities used by HiGMemPlus smoke experiments."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Iterable


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "him",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


TEMPORAL_TERMS = {
    "after",
    "before",
    "earlier",
    "later",
    "then",
    "today",
    "tomorrow",
    "tonight",
    "yesterday",
    "last",
    "next",
    "ago",
    "date",
    "time",
    "first",
    "second",
    "finally",
}


CAUSAL_TERMS = {"because", "since", "so", "therefore", "caused", "reason", "why"}
PREFERENCE_TERMS = {"like", "love", "hate", "prefer", "favorite", "want", "need", "wish"}
SOCIAL_TERMS = {"friend", "mother", "father", "brother", "sister", "boss", "wife", "husband", "date"}


def stable_id(prefix: str, *parts: object) -> str:
    text = "||".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9']*", text or "")]


def content_tokens(text: str) -> list[str]:
    return [tok for tok in tokenize(text) if len(tok) > 2 and tok not in STOPWORDS]


def top_terms(text: str, limit: int = 12) -> list[str]:
    counts = Counter(content_tokens(text))
    return [term for term, _count in counts.most_common(limit)]


def lexical_score(query: str, text: str) -> float:
    q = content_tokens(query)
    if not q:
        return 0.0
    q_counts = Counter(q)
    t_counts = Counter(content_tokens(text))
    overlap = sum((q_counts & t_counts).values())
    phrase_bonus = 0.0
    lowered = (text or "").lower()
    for term in set(q):
        if len(term) >= 5 and term in lowered:
            phrase_bonus += 0.05
    return overlap / max(1, len(q)) + phrase_bonus


def rank_by_lexical(query: str, items: Iterable[tuple[str, object]], limit: int) -> list[tuple[float, str, object]]:
    scored = [(lexical_score(query, text), text, item) for text, item in items]
    scored.sort(key=lambda row: row[0], reverse=True)
    return [row for row in scored[:limit] if row[0] > 0]


def split_atomic_sentences(text: str) -> list[str]:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []
    pieces = re.split(r"(?<=[.!?])\s+|;\s+", cleaned)
    return [piece.strip() for piece in pieces if piece.strip()]


def extract_entities(text: str) -> list[str]:
    candidates = re.findall(r"\b[A-Z][A-Za-z']+(?:\s+[A-Z][A-Za-z']+)?\b", text or "")
    seen = []
    for candidate in candidates:
        if candidate.lower() in STOPWORDS:
            continue
        if candidate not in seen:
            seen.append(candidate)
    return seen[:8]


def relation_type(text: str) -> str:
    toks = set(content_tokens(text))
    if toks & TEMPORAL_TERMS:
        return "temporal"
    if toks & CAUSAL_TERMS:
        return "causal"
    if toks & PREFERENCE_TERMS:
        return "preference/attribute"
    if toks & SOCIAL_TERMS:
        return "social"
    if len(extract_entities(text)) >= 2:
        return "entity"
    return "event-update"


def infer_predicate(text: str) -> str:
    toks = tokenize(text)
    for tok in toks:
        if tok.endswith("ed") or tok.endswith("ing"):
            return tok
    for tok in toks:
        if tok not in STOPWORDS:
            return tok
    return ""


def extract_time_expr(text: str) -> str:
    lowered = (text or "").lower()
    for expr in [
        "yesterday",
        "tomorrow",
        "tonight",
        "today",
        "last week",
        "next week",
        "last month",
        "next month",
        "last night",
        "this morning",
        "this afternoon",
        "this evening",
    ]:
        if expr in lowered:
            return expr
    match = re.search(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", lowered)
    if match:
        return match.group(0)
    match = re.search(r"\b\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?)?\b", lowered)
    if match:
        return match.group(0)
    return ""


def normalize_relative_time(expr: str, mentioned_at: str) -> str:
    expr = (expr or "").lower()
    if not expr or not mentioned_at:
        return ""
    try:
        base = datetime.strptime(mentioned_at, "%B %d, %Y")
    except ValueError:
        return ""
    if expr == "yesterday" or expr == "last night":
        return (base - timedelta(days=1)).date().isoformat()
    if expr == "tomorrow":
        return (base + timedelta(days=1)).date().isoformat()
    if expr == "today" or expr == "tonight":
        return base.date().isoformat()
    if expr == "last week":
        return (base - timedelta(days=7)).date().isoformat()
    if expr == "next week":
        return (base + timedelta(days=7)).date().isoformat()
    if expr == "last month":
        return (base - timedelta(days=30)).date().isoformat()
    if expr == "next month":
        return (base + timedelta(days=30)).date().isoformat()
    return ""


def question_family(question: str, metadata: dict[str, object] | None = None) -> str:
    metadata = metadata or {}
    raw_type = str(metadata.get("question_type") or metadata.get("category") or "").lower()
    raw_hop = str(metadata.get("hop_type") or "").lower()
    text = " ".join(content_tokens(question))
    if "unans" in raw_type or str(metadata.get("answerability_label", "")).lower() == "unanswerable":
        return "adversarial"
    if "temporal" in raw_type or "temporal" in raw_hop or any(term in text.split() for term in TEMPORAL_TERMS):
        return "temporal"
    if "two_hop" in raw_hop or "multi" in raw_type:
        return "multi-hop"
    if "open" in raw_type or any(term in text for term in ["why", "reason", "should", "would", "could"]):
        return "open-domain"
    if metadata.get("dataset") == "longdialqa":
        return "longdialogue"
    return "single-hop"
