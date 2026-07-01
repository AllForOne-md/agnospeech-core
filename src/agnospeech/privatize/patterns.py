"""Shared PII / identifier patterns used by L1 redaction and by the PII-recall
companion attack. Typed placeholders keep the text readable and keep the harm
span intact: we replace the identifier, not the sentence around it.

These regex rules are the CPU-only spine. The event drop-in adds Presidio's
recognizers + a GLiNER NER pass for orgs/locations/persons behind the same
``redact`` call; the interface and placeholder vocabulary do not change.
"""

from __future__ import annotations

import re

# Order matters: earlier patterns win on overlap (urls before handles, etc.).
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("URL", re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # reddit / social handles: /u/name, u/name, @name
    ("HANDLE", re.compile(r"(?<!\w)(?:/?u/|@)[A-Za-z0-9_]{2,}")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?\d[\d\-\s().]{7,}\d)(?!\w)")),
    # standalone long digit runs (ids, zips, card-ish), 4+ digits
    ("NUMBER", re.compile(r"(?<!\w)\d{4,}(?!\w)")),
    ("MONEY", re.compile(r"[$£€]\s?\d[\d,]*(?:\.\d+)?")),
]

# Lightweight gazetteer NER stand-in for the spine. The event GLiNER pass
# supersedes this; kept tiny and obviously heuristic on purpose.
_PERSON_TITLES = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Mx)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"
)
_LOCATION_HINT = re.compile(
    r"\b(?:in|from|at|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
)


def find_pii_spans(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, type) PII spans, non-overlapping, leftmost-longest."""
    spans: list[tuple[int, int, str]] = []
    taken = [False] * (len(text) + 1)
    for ptype, pat in PATTERNS:
        for m in pat.finditer(text):
            s, e = m.start(), m.end()
            if any(taken[i] for i in range(s, e)):
                continue
            for i in range(s, e):
                taken[i] = True
            spans.append((s, e, ptype))
    for m in _PERSON_TITLES.finditer(text):
        s, e = m.start(), m.end()
        if not any(taken[i] for i in range(s, e)):
            for i in range(s, e):
                taken[i] = True
            spans.append((s, e, "PERSON"))
    spans.sort()
    return spans
