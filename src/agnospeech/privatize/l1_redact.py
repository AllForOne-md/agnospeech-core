"""L1 Redact: replace direct identifiers with typed placeholders.

Conservative, high-precision PII removal. The harm-bearing content is left
untouched so the detector still works; only identifiers become ``[TYPE]``.
This is the cheapest, most reliable privacy level and the floor every other
level builds on.

Spine implementation: regex + a tiny gazetteer (see patterns.py). Event drop-in:
Presidio Replace operator + GLiNER NER, same ``apply`` signature and placeholder
vocabulary, so nothing downstream changes.
"""

from __future__ import annotations

from .base import Privatizer
from .patterns import find_pii_spans


class L1Redact(Privatizer):
    level = "L1"
    name = "redact_regex_gazetteer"

    def apply(self, text: str) -> str:
        spans = find_pii_spans(text)
        if not spans:
            return text
        out = []
        cursor = 0
        for s, e, ptype in spans:
            out.append(text[cursor:s])
            out.append(f"[{ptype}]")
            cursor = e
        out.append(text[cursor:])
        return "".join(out)
