"""The trade-off score.

    TO = (Utility_protected / Utility_original) - (Privacy_protected / Privacy_original)

Range [-1, 1], higher is better. The utility ratio is majority-corrected; the
privacy ratio is the plain accuracy ratio. By construction the no-op baseline
(L0) scores TO = 0, so a useful operating point must beat both the no-op and the
degenerate maximum.

The default HEADLINE uses the ADAPTIVE attacker. The static attacker gives an
inflated, optimistic TO that the S1 honesty toggle exposes.
"""

from __future__ import annotations


def to_score(utility_ratio: float, privacy_ratio: float) -> float:
    return float(utility_ratio - privacy_ratio)
