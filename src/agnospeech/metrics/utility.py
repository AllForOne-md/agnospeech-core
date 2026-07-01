"""Utility = HSD macro-F1, with majority-class baseline correction.

The corrected utility ratio anchors the majority-class baseline at 0, so that a
privatization which collapses the detector down to "always predict the majority
class" scores a utility ratio of 0, not a flatteringly high raw ratio. This is
the honesty move that keeps the TO comparable to Meisenbacher's baseline-corrected
RG.

    utility_ratio = clip( (F1_protected - F1_majority) / (F1_raw - F1_majority), 0, 1 )
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def majority_baseline_f1(y_true) -> float:
    """Macro-F1 of always predicting the majority class on this label set."""
    y_true = np.asarray(y_true)
    if len(y_true) == 0:
        return 0.0
    vals, counts = np.unique(y_true, return_counts=True)
    majority = vals[int(np.argmax(counts))]
    y_pred = np.full_like(y_true, majority)
    return macro_f1(y_true, y_pred)


def utility_ratio(f1_protected: float, f1_raw: float, f1_majority: float) -> float:
    denom = f1_raw - f1_majority
    if denom <= 1e-9:
        return 0.0
    r = (f1_protected - f1_majority) / denom
    return float(min(1.0, max(0.0, r)))
