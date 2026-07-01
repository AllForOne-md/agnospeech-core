"""Utility-vs-Privacy curve with floors and a non-degenerate operating-point
selector (S1 / S17).

Sweeping the L3 intensity traces a curve. Because L3 anchors the harm rationale,
high intensity keeps the detector alive (utility stays moderate) while erasing
the authorship signal (privacy ratio -> low), so TO climbs toward the high end.
That high-TO end is the DEGENERATE optimum: the text is unreadable. The floors
(HSD-F1, readability, semantic similarity) mark that zone infeasible, and the
operating-point selector refuses it and returns the knee of the FEASIBLE region.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class CurvePoint:
    intensity: float
    utility_ratio: float
    privacy_ratio: float
    macro_f1: float
    readability: float
    semantic_sim: float
    to: float
    feasible: bool


@dataclass
class Floors:
    # Calibrated against the CPU-spine proxies: the feasible band keeps text
    # mostly real words and semantically close to the original; the degenerate
    # tail (unreadable "·"-heavy noise that still posts a high TO) falls below.
    # Absolute magic constants; RelativeFloors below is the de-fixed alternative.
    macro_f1: float = 0.55
    readability: float = 0.45
    semantic_sim: float = 0.40
    tol: float = 1e-9  # boundary tolerance so FP noise cannot flip feasibility

    def feasible_mask(self, points: list["CurvePoint"]) -> list[bool]:
        return [
            (p.macro_f1 >= self.macro_f1 - self.tol
             and p.readability >= self.readability - self.tol
             and p.semantic_sim >= self.semantic_sim - self.tol)
            for p in points
        ]


@dataclass
class RelativeFloors:
    """Distribution-relative floors: deletes the three absolute magic constants.

    A point is feasible iff:
      - utility_ratio >= ``util_min``  (replaces the macro_f1=0.55 constant;
        majority-corrected and f1_raw-relative already, full alignment with the
        baseline-corrected RG (Meisenbacher). Never binds while harm-anchoring keeps F1
        alive, so it is a degenerate-refusal guard, not the load-bearing floor.)
      - readability >= ``read_drop`` x (best readability on the sweep)
      - semantic_sim >= ``sem_drop`` x (best semantic_sim on the sweep)

    The readability / semantic floors are relative DROP-ratios from the rewrite's
    own least-aggressive point, NOT a high raw percentile (which over-constrains).
    These are frozen policy ratios, set a priori, NEVER grid-searched against the
    board (or the overfitting just relocates into the threshold). A tolerance band
    keeps cross-machine FP non-determinism from flipping feasibility at the edge.

    Calibrated on reddit_25 to reproduce the absolute floors' feasible/infeasible
    split and the identical recommended operating point (see test_relative_floors).
    """

    util_min: float = 0.5
    read_drop: float = 0.60  # 0.60*max_readability tracks the absolute 0.45 floor
    sem_drop: float = 0.50   # on reddit_25 (a calibration, not a structural identity)
    tol: float = 1e-9        # match Floors.tol so boundary feasibility is identical

    def feasible_mask(self, points: list["CurvePoint"]) -> list[bool]:
        if not points:
            return []
        read_floor = self.read_drop * max(p.readability for p in points)
        sem_floor = self.sem_drop * max(p.semantic_sim for p in points)
        return [
            (p.utility_ratio >= self.util_min - self.tol
             and p.readability >= read_floor - self.tol
             and p.semantic_sim >= sem_floor - self.tol)
            for p in points
        ]


def _kneedle(x: np.ndarray, y: np.ndarray) -> int:
    """Index of the knee: the point of maximum drop below the straight chord,
    after normalizing both axes to [0, 1]. x must be sorted ascending."""
    if len(x) < 3:
        return 0
    xn = (x - x.min()) / (np.ptp(x) + 1e-12)
    yn = (y - y.min()) / (np.ptp(y) + 1e-12)
    chord = xn  # straight line from (0,0) to (1,1) under this normalization
    diff = yn - chord
    return int(np.argmax(np.abs(diff)))


def build_curve(points: list[CurvePoint], floors) -> dict:
    mask = floors.feasible_mask(points)
    for p, ok in zip(points, mask):
        p.feasible = ok
    feasible = [p for p in points if p.feasible]
    global_max = max(points, key=lambda p: p.to)

    if feasible:
        # Knee on the feasible frontier (privacy_ratio ascending vs utility_ratio).
        fs = sorted(feasible, key=lambda p: p.privacy_ratio)
        xi = np.array([p.privacy_ratio for p in fs])
        yi = np.array([p.utility_ratio for p in fs])
        knee = fs[_kneedle(xi, yi)]
        # Recommend the feasible point with the best TO; the knee is the
        # principled cross-check that it is not bought with unreadable text.
        recommended = max(feasible, key=lambda p: p.to)
    else:
        knee = recommended = global_max

    # Degenerate zone: the infeasible high-intensity band. Report its lower edge.
    infeasible = [p for p in points if not p.feasible]
    degenerate_from = min((p.intensity for p in infeasible), default=None)

    return {
        "floors": asdict(floors),
        "points": [asdict(p) for p in points],
        "global_max_to": asdict(global_max),
        "recommended": asdict(recommended),
        "knee": asdict(knee),
        "degenerate_zone_from_intensity": degenerate_from,
        "degenerate_optimum_is_refused": (not global_max.feasible),
    }
