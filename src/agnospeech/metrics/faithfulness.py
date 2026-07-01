"""Accuracy-measuring strategies beyond the single TO point estimate.

Four complementary reads, each closing a specific hole the headline F1/TO leaves:

1. ERASER comprehensiveness & sufficiency (DeYoung et al., ACL 2020): is the L2
   rationale FAITHFUL, i.e. does the detector actually rely on the kept tokens?
   Comprehensiveness = P(hate|full) - P(hate|rationale removed)  (higher = the
   kept tokens carry the signal). Sufficiency = P(hate|full) - P(hate|rationale
   only)  (lower / ->0 = the kept tokens alone suffice). Sufficiency is gameable
   (Hsia et al. 2023), so it is MEASURED and reported next to comprehensiveness,
   never asserted.

2. Board form: Score = HS_accuracy - author_accuracy in [-1, 1], the privhsd.com
   server's grading reduced to our local proxy. Robust to the same-head F1
   inflation because accuracy, not the head's own confidence, is what is scored.

3. Dual-detector utility: a char-only head (no word channel) measures utility on
   the L2 output. The learned L2 attributes spans with the primary head's WORD
   channel, so measuring utility with that same head inflates F1 (the
   sufficiency-gaming Goodhart effect the spike found above keep_frac ~0.6). The
   char-only head shares no features with the word-channel keep decision, so its
   utility ratio is Goodhart-attenuated. It is NOT fully independent of the primary
   (its char basis is a subset of the primary's char channel, proba corr ~0.98), so
   it attenuates rather than rescues the negative ERASER sufficiency. The fully
   independent judge is a pretrained transformer head (event P1).

4. Subgroup F1: macro-F1 sliced by a grouping (here, text-length tertile, with a
   hook for dialect / target-group slices). A single corpus-level F1 hides a
   utility collapse on one slice; the per-slice spread surfaces it. The dialectal
   (AAE) fairness audit needs an external labeled slice (HateCheck functional
   classes), flagged where it is not available locally.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score

from .utility import macro_f1


def _join_spans(text: str, spans: list[tuple[int, int]]) -> str:
    return " ".join(text[s:e] for s, e in spans).strip()


def _delete_spans(text: str, spans: list[tuple[int, int]]) -> str:
    keep, cursor = [], 0
    for s, e in sorted(spans):
        keep.append(text[cursor:s])
        cursor = e
    keep.append(text[cursor:])
    out = " ".join(part.strip() for part in keep if part.strip())
    return out.strip()


def eraser_scores(hsd, originals: list[str],
                  rationale_spans: list[list[tuple[int, int]]]) -> dict:
    """ERASER comprehensiveness & sufficiency for a token-rationale extractor.

    ``originals`` are the (already-L1-redacted) texts the offsets index into;
    ``rationale_spans[i]`` are the kept (start, end) spans for ``originals[i]``.
    Texts with an empty rationale are skipped (no rationale to score).
    """
    full, only, removed = [], [], []
    for text, spans in zip(originals, rationale_spans):
        if not spans:
            continue
        full.append(text)
        only.append(_join_spans(text, spans))
        removed.append(_delete_spans(text, spans))
    n = len(full)
    n_total = len(originals)
    if n == 0:
        return {"comprehensiveness": 0.0, "sufficiency": 0.0, "n_scored": 0,
                "n_skipped": n_total, "note": "no non-empty rationales to score"}
    p_full = hsd.proba(full)
    p_only = hsd.proba(only)
    p_removed = hsd.proba(removed)
    comp = float(np.mean(p_full - p_removed))
    suff = float(np.mean(p_full - p_only))
    return {
        "comprehensiveness": round(comp, 4),
        "sufficiency": round(suff, 4),
        "n_scored": n,
        "n_skipped": n_total - n,
        "note": "comprehensiveness higher = kept tokens carry the signal; "
                "sufficiency near 0 = kept tokens alone suffice (gameable, "
                "reported not asserted). Conditional on a NON-EMPTY rationale: the "
                "n_skipped empty-rationale (most over-redacted) posts are excluded.",
    }


def board_score(y_true, hsd_pred, attacker_acc: float) -> dict:
    """Score = HS_accuracy - author_accuracy, the privhsd.com server proxy."""
    hs_acc = float(accuracy_score(y_true, hsd_pred))
    return {
        "hs_accuracy": round(hs_acc, 4),
        "author_accuracy": round(attacker_acc, 4),
        "board_score": round(hs_acc - attacker_acc, 4),
    }


def dual_detector_utility(dual_hsd, y_true, level_texts: dict[str, list[str]],
                          f1_raw_dual: float, f1_maj: float) -> dict:
    """Goodhart-attenuated utility: a char-only head (no word channel) reads each
    level's output, so the L2 word-channel keep decision cannot inflate it.

    ``f1_raw_dual`` is the dual head's own macro-F1 on the raw test set (its own
    baseline), so the ratio is anchored to the dual head, not the primary head.
    Not a fully independent detector (proba corr ~0.98 with the primary).
    """
    from .utility import utility_ratio
    out = {}
    for code, texts in level_texts.items():
        f1 = macro_f1(y_true, dual_hsd.predict(texts))
        out[code] = {
            "macro_f1_dual": round(f1, 4),
            "utility_ratio_dual": round(utility_ratio(f1, f1_raw_dual, f1_maj), 4),
        }
    return out


def length_tertile_slices(texts: list[str]) -> list[np.ndarray]:
    """Index masks for short / medium / long raw posts (a corpus-agnostic slice).

    A generic robustness probe: utility that holds corpus-wide can still collapse
    on the short slice (least context). Returns three boolean index arrays.
    """
    lengths = np.array([len(t) for t in texts])
    if len(lengths) < 3:
        return [np.ones(len(lengths), dtype=bool)]
    q1, q2 = np.quantile(lengths, [1 / 3, 2 / 3])
    return [lengths <= q1, (lengths > q1) & (lengths <= q2), lengths > q2]


def subgroup_f1(y_true, y_pred, slices: list[np.ndarray],
                names: list[str]) -> dict:
    """Macro-F1 per slice plus the spread (max - min). A wide spread flags a
    utility collapse hidden by the corpus-level number."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    per = {}
    vals = []
    for mask, name in zip(slices, names):
        if mask.sum() == 0:
            continue
        f1 = macro_f1(y_true[mask], y_pred[mask])
        per[name] = {"n": int(mask.sum()), "macro_f1": round(f1, 4)}
        vals.append(f1)
    spread = round(max(vals) - min(vals), 4) if len(vals) > 1 else 0.0
    return {"per_slice": per, "spread": spread,
            "note": "spread = max-min macro-F1 across slices; large = uneven "
                    "utility. Dialect (AAE) fairness needs an external slice "
                    "(HateCheck), not available on reddit_25."}
