"""S16: paired-bootstrap dominance test over the per-instance logs.

Resample the test indices B times (paired across levels, so the same resample
hits every level), recompute TO per level each time, and report a CI per level
plus the probability that the recommended operating point beats each alternative
and the no-op. This is a visible step beyond a point estimate: it answers "is the
recommended point's TO advantage real, or noise?".
"""

from __future__ import annotations

import numpy as np

from .to_score import to_score
from .utility import macro_f1, majority_baseline_f1, utility_ratio


def paired_bootstrap(
    y_true: np.ndarray,
    hsd_pred: dict[str, np.ndarray],
    auth_correct: dict[str, np.ndarray],
    levels: list[str],
    l0: str = "L0",
    b: int = 2000,
    seed: int = 0,
) -> dict[str, dict]:
    """Return per-level {mean, lo, hi, samples}. ``samples`` is the (B,) TO
    vector, kept so dominance probabilities are computed paired."""
    rng = np.random.default_rng(seed)
    m = len(y_true)
    y_true = np.asarray(y_true)
    samples = {lv: np.empty(b) for lv in levels}
    for k in range(b):
        idx = rng.integers(0, m, m)
        yt = y_true[idx]
        f1_maj = majority_baseline_f1(yt)
        f1_raw = macro_f1(yt, hsd_pred[l0][idx])
        p_orig = float(np.mean(auth_correct[l0][idx]))
        for lv in levels:
            f1 = macro_f1(yt, hsd_pred[lv][idx])
            ur = utility_ratio(f1, f1_raw, f1_maj)
            p_acc = float(np.mean(auth_correct[lv][idx]))
            pr = p_acc / p_orig if p_orig > 1e-9 else 0.0
            samples[lv][k] = to_score(ur, pr)
    out: dict[str, dict] = {}
    for lv in levels:
        s = samples[lv]
        out[lv] = {
            "mean": float(np.mean(s)),
            "lo": float(np.percentile(s, 2.5)),
            "hi": float(np.percentile(s, 97.5)),
            "samples": s,
        }
    return out


def dominance(boot: dict[str, dict], recommended: str, others: list[str]) -> dict:
    """P(TO_recommended > TO_other) for each other level and vs the no-op (0),
    plus a one-sided paired p-value against the best alternative."""
    rec = boot[recommended]["samples"]
    probs = {}
    for o in others:
        probs[o] = float(np.mean(rec > boot[o]["samples"]))
    probs["no_op"] = float(np.mean(rec > 0.0))
    # one-sided p-value vs the strongest alternative (incl. no-op at TO=0)
    alt_means = {o: boot[o]["mean"] for o in others}
    alt_means["no_op"] = 0.0
    strongest = max(alt_means, key=alt_means.get)
    if strongest == "no_op":
        p_value = float(np.mean(rec <= 0.0))
    else:
        p_value = float(np.mean(rec <= boot[strongest]["samples"]))
    return {"win_prob": probs, "strongest_alt": strongest, "p_value_one_sided": p_value}
