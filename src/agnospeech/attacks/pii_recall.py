"""PII-recall companion column (never substituted into the leaderboard TO).

Measures how much direct PII survives privatization: of the identifier spans
found in the raw text, what fraction still appears verbatim in the output. Lower
leakage is better. Reported alongside TO as a scorecard column and used by the
victim-side privacy readout (S11), never folded into the scored trade-off.
"""

from __future__ import annotations

from ..privatize.patterns import find_pii_spans


def pii_leakage(raw: str, privatized: str) -> tuple[int, int]:
    """Return (surviving, total) PII spans. ``surviving`` = raw identifiers still
    present verbatim in the privatized text."""
    spans = find_pii_spans(raw)
    total = len(spans)
    if total == 0:
        return 0, 0
    surviving = sum(1 for s, e, _ in spans if raw[s:e] in privatized)
    return surviving, total


def pii_removed_fraction(raws: list[str], privs: list[str]) -> float:
    surv = tot = 0
    for r, p in zip(raws, privs):
        s, t = pii_leakage(r, p)
        surv += s
        tot += t
    if tot == 0:
        return 1.0
    return 1.0 - surv / tot
