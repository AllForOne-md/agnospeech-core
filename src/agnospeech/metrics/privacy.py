"""Privacy = authorship-attribution attack accuracy (lower is better).

The scored privacy ratio is the plain accuracy ratio Pp/Po, matching the
Meisenbacher's RG. The chance-corrected (skill-corrected) term is reported as an extra
column only, never substituted into the leaderboard TO.

    privacy_ratio   = Privacy_protected / Privacy_original
    chance_corrected(acc, N) = (acc - 1/N) / (1 - 1/N)

All numbers are closed-world N-candidate attribution, an upper-bias (optimistic)
estimate of real-world protection. Open-world re-identification is a named,
stronger threat we do not build into the product surface.
"""

from __future__ import annotations


def privacy_ratio(p_protected: float, p_original: float) -> float:
    if p_original <= 1e-9:
        return 0.0
    return float(p_protected / p_original)


def chance_corrected(accuracy: float, n_candidates: int) -> float:
    """Skill above random guessing among N candidates. Companion column only."""
    chance = 1.0 / n_candidates
    denom = 1.0 - chance
    if denom <= 1e-9:
        return 0.0
    return float((accuracy - chance) / denom)
